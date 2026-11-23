# KAAL-304M Local

Open-source, zero-config local AI chat application and Apple Silicon inference engine for **KAAL-304M** — a custom 304M-parameter decoder-only PyTorch LLM trained from scratch.

Runs natively on **Apple Silicon (M1 / M2 / M3 / M4)** using **PyTorch MPS in FP16 (~610 MB unified memory)** with exact per-layer KV-caching and a graceful CPU fallback.

## 3-Step Apple Silicon Quickstart

1. **Clone the repository**
   ```bash
   git clone <repo-url> kaal-304m-local
   cd kaal-304m-local
   ```

2. **Place the model weights in `./weights/`**
   Drop both required files into `./weights/`:
   - `weights/best_model.pt` (FP32 ~1.35 GB or FP16 ~610 MB checkpoint)
   - `weights/tokenizer.json` (Hugging Face BPE tokenizer)

3. **Run the one-command launcher**
   ```bash
   ./start.sh
   ```
   `start.sh` verifies macOS/Apple Silicon, Python 3.10+, and Node 18+, sets up `.venv` and `frontend/node_modules` automatically on first run, starts FastAPI (`127.0.0.1:8000`) and Vite (`127.0.0.1:5173`), waits for `/api/health`, and opens `http://localhost:5173` in your browser. Press `Ctrl+C` to stop both servers cleanly.

## Architecture Specifications

| Parameter | Value |
| :--- | :--- |
| Model Name | KAAL-304M |
| Unique Parameters | `304,355,328` (weight-tied `tok_emb` / `out_head`) |
| Vocabulary Size | `49,152` (`<\|endoftext\|>` EOS ID = `0`) |
| Context Length | `2,048` (learned positional embeddings) |
| Layers | `20` pre-norm Transformer blocks |
| Embedding Dim | `1,024` |
| Attention Heads | `16` (`head_dim = 64`, fused QKV with bias) |
| Feed-Forward Dim | `4,096` (tanh-approximated GELU) |
| LayerNorm | Custom FP32-computed `scale` / `shift` (`eps = 1e-5`) |
| Runtime Precision | FP32 on Apple Silicon MPS (~1.2 GB) / FP32 on CPU |

## Strictly Locked Decoding

KAAL-304M uses a single, immutable decoding profile (`backend/engine/config.py`) with zero UI sliders and strict API schema validation (`extra="forbid"`):

- `do_sample = True`
- `temperature = 0.45`
- `top_k = 35`
- `top_p = 0.90`
- `repetition_penalty = 1.18` (sign-aware on generated tokens)
- `no_repeat_ngram_size = 3` (hard 3-gram loop blocker)
- `max_new_tokens = 128` (stops immediately on `<|endoftext|>`)

## API Endpoints

- `GET /api/health` — Live model status, device, dtype, parameter count, and `missing_files`.
- `POST /api/chat/stream` — Server-Sent Events (`text/event-stream`) streaming `{ "message": str, "context": str }` (`extra="forbid"`).
- `POST /api/chat` — Non-streaming JSON endpoint with the same locked schema.

## Running Tests

```bash
.venv/bin/pytest -v
```

Verifies exact unique parameter count (`304,355,328`), tied embedding/output head weights, KV-cache logit parity (`< 1e-4`), canonical prompt formatting, locked decoding immutability, API schema enforcement, and strict checkpoint loading.

## License

Released under the [MIT License](LICENSE).
