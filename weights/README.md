# KAAL-304M Weights Directory

Place the two required KAAL-304M model files in this directory before starting the app:

1. `best_model.pt` — PyTorch checkpoint (`payload["model"]` or raw `state_dict`, FP32 ~1.35 GB or FP16 ~610 MB)
2. `tokenizer.json` — Hugging Face `tokenizers` JSON file (`vocab_size = 49,152`, `<|endoftext|>` EOS token)

Both `*.pt` and `*.json` files in this folder are git-ignored so model weights are never accidentally committed.

Once both files are in place, run `./start.sh` from the repository root.
