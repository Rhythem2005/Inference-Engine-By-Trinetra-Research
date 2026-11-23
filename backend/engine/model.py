"""KAAL-304M decoder-only Transformer architecture with exact KV-caching."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEFAULT_MODEL_CONFIG, ModelConfig

KVCacheLayer = Tuple[torch.Tensor, torch.Tensor]
KVCache = List[KVCacheLayer]


class LayerNorm(nn.Module):
    """Custom LayerNorm with scale/shift parameters computed in FP32 for stability."""

    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.layer_norm(
            x.float(),
            (x.shape[-1],),
            self.scale.float(),
            self.shift.float(),
            self.eps,
        )
        return out.to(x.dtype)


class GELU(nn.Module):
    """Tanh-approximated GELU activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention with fused QKV projection and exact KV-cache."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError(f"d_out ({d_out}) must be divisible by num_heads ({num_heads})")
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCacheLayer] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCacheLayer]]:
        batch_size, num_tokens, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        past_len = 0
        if kv_cache is not None:
            past_k, past_v = kv_cache
            past_len = past_k.shape[2]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present: Optional[KVCacheLayer] = (k, v) if use_cache else None
        dropout_p = self.dropout if self.training else 0.0

        if num_tokens == 1:
            # Single-token prefill or KV-cached decode fast path on Apple Silicon MPS.
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=False,
            )
        elif kv_cache is None:
            # Explicit boolean lower-triangular causal mask to avoid PyTorch MPS FP16 is_causal=True bug.
            causal_mask = torch.ones(
                (num_tokens, num_tokens), dtype=torch.bool, device=q.device
            ).tril()
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=causal_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
        else:
            # Multi-token continuation with existing KV-cache.
            total_len = past_len + num_tokens
            q_positions = torch.arange(past_len, total_len, device=x.device).unsqueeze(1)
            k_positions = torch.arange(total_len, device=x.device).unsqueeze(0)
            causal_mask = k_positions <= q_positions
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=causal_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )

        context = context.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        return self.out_proj(context), present


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.ffn_dim),
            GELU(),
            nn.Linear(cfg.ffn_dim, cfg.emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer decoder block."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias,
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim, eps=cfg.layer_norm_eps)
        self.norm2 = LayerNorm(cfg.emb_dim, eps=cfg.layer_norm_eps)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCacheLayer] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCacheLayer]]:
        shortcut = x
        att_out, present = self.att(self.norm1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = shortcut + self.drop_shortcut(att_out)

        shortcut = x
        x = self.ff(self.norm2(x))
        x = shortcut + self.drop_shortcut(x)
        return x, present


class KAALModel(nn.Module):
    """KAAL-304M decoder-only Transformer with weight-tied output projection."""

    def __init__(self, cfg: ModelConfig = DEFAULT_MODEL_CONFIG) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = LayerNorm(cfg.emb_dim, eps=cfg.layer_norm_eps)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

        # Tie input token embeddings and output language modeling head weights.
        self.out_head.weight = self.tok_emb.weight

    def forward(
        self,
        idx: torch.Tensor,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, KVCache]:
        _, seq_len = idx.shape
        past_len = past_kv[0][0].shape[2] if past_kv is not None else 0
        if past_len + seq_len > self.cfg.context_length:
            raise ValueError(
                f"Sequence length {past_len + seq_len} exceeds context_length {self.cfg.context_length}"
            )

        positions = torch.arange(past_len, past_len + seq_len, device=idx.device, dtype=torch.long)
        x = self.tok_emb(idx) + self.pos_emb(positions)
        x = self.drop_emb(x)

        next_kv: Optional[KVCache] = [] if use_cache else None
        for layer_idx, block in enumerate(self.trf_blocks):
            layer_cache = past_kv[layer_idx] if past_kv is not None else None
            x, present = block(x, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache and present is not None:
                assert next_kv is not None
                next_kv.append(present)

        x = self.final_norm(x)
        logits = self.out_head(x)
        if use_cache:
            assert next_kv is not None
            return logits, next_kv
        return logits


def count_unique_parameters(model: nn.Module) -> int:
    """Count unique trainable/non-trainable parameters deduplicated by storage pointer."""
    seen_ptrs: set[int] = set()
    total = 0
    for param in model.parameters():
        ptr = param.data_ptr()
        if ptr not in seen_ptrs:
            seen_ptrs.add(ptr)
            total += param.numel()
    return total
