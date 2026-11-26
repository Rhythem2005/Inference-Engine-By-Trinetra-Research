# KAAL-304M Inference Engine

KAAL-304M is an experimental language model developed by Trinetra AI Research. This repository contains its local PyTorch inference engine, HTTP API, and browser chat. For model architecture, training statistics, and evaluation, visit the [Trinetra AI Research website](https://trinetra-research.vercel.app/).

## Quick Start

| Operating system | Engine | Device selection |
| --- | --- | --- |
| macOS 14 or newer on Apple Silicon (M-series) | `backend/engine/` | Apple Silicon MPS when available, otherwise CPU |
| Windows 64-bit (x86-64) | `backend/windows/` | Supported NVIDIA CUDA when available, otherwise CPU |

Copy the GitHub clone URL from **Code → HTTPS** and run `git clone URL kaal-304m`, replacing `URL` with the copied address. Then run `cd kaal-304m`. If you already downloaded the source, open a terminal in the folder containing `backend/`, `frontend/`, and `weights/`. The public clone URL is pending release approval.

Both engines use the same model files. A verified public download is pending. If you already have the matching checkpoint and tokenizer from Trinetra AI Research, place them in the root `weights/` folder as shown below. The checkpoint may be about 1.4 GB. Loading and running it requires additional memory. The MIT license covers the inference software. Model artifacts have separate release and licensing requirements.

```text
weights/
├── best_model.pt
└── tokenizer.json
```

### macOS — Apple Silicon

Install native Apple Silicon builds of Python 3.11 and Node.js 22 LTS (includes npm). Open Terminal in the repository root and run the following. The launcher starts both local servers and opens the chat. Press **Ctrl+C** to stop both.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -r backend/requirements.txt
npm --prefix frontend ci
./start.sh
```

The macOS engine uses FP32 on MPS or CPU. Check the selected device and model status from a second terminal.

```sh
curl -fsS http://127.0.0.1:8000/api/health
```

### Windows — GPU/CPU

Install 64-bit Python 3.11 with the Python launcher and Node.js 22 LTS (includes npm). Open PowerShell in the repository root.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools
```

Install **one** PyTorch build. Use the CPU build if your computer has AMD, Intel, or integrated graphics only, or if you prefer CPU inference.

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.15.0" --index-url https://download.pytorch.org/whl/cpu
```

For a compatible NVIDIA GPU, update your NVIDIA driver and use this CUDA build instead. If you need another CUDA build, choose Windows, Pip, and a CUDA version supported by your GPU and driver. Use a PyTorch release that meets the repository's minimum of 2.15.0. [PyTorch installation selector](https://pytorch.org/get-started/locally/).

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.15.0" --index-url https://download.pytorch.org/whl/cu132
```

Install the remaining dependencies and start the backend in the same PowerShell window.

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
npm.cmd --prefix frontend ci
.\.venv\Scripts\python.exe -m uvicorn backend.windows.main:app --host 127.0.0.1 --port 8000
```

Open a second PowerShell window in the same folder and start the browser app. Press **Ctrl+C** in each window when you are done.

```powershell
npm.cmd --prefix frontend run dev
```

The backend logs its selected device at startup. CUDA and CPU both use FP32. If CUDA is unavailable or fails during model setup, the backend switches to CPU. If GPU memory runs out during generation, the backend switches to CPU and retries automatically if no output has been generated. If a reply was already streaming, resend the message after the app reports the switch.

To force CPU, stop the backend and run these commands in its PowerShell window. To restore automatic selection, set `$env:KAAL_DEVICE = "auto"` before restarting.

```powershell
$env:KAAL_DEVICE = "cpu"
.\.venv\Scripts\python.exe -m uvicorn backend.windows.main:app --host 127.0.0.1 --port 8000
```

Check the device and model status from another PowerShell window.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

### Open the chat

Both servers bind to the loopback interface. Select **Get started** in the chat, enter a name or nickname, and choose a mascot. Type a prompt such as “How do plants produce food?” and press **Enter**. **Stop** ends generation. The status badge displays the active device when the model is ready. [Local chat](http://localhost:5173).

## Inference modes

- **Browser chat** streams replies, supports a stop button, and lets you attach reference text with **Add Reference Context**. The browser uses recent turns as context within the model's token budget.
- **HTTP API** accepts a JSON object with `message` and optional `context`. Use `POST /api/chat` for a complete JSON response or `POST /api/chat/stream` for Server-Sent Events. Interactive API docs are available at [localhost port 8000](http://127.0.0.1:8000/docs).
- **Python** callers can import `KAALEngine` from `backend.engine` on macOS or `backend.windows.engine` on Windows, call `load()`, then use `generate()` or iterate over `stream()`.

On macOS, run this from another terminal while the app is up.

```sh
curl -fsS http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Summarize this sentence.","context":"Plants use sunlight to make sugars."}'
```

The equivalent in Windows PowerShell uses the same request body.

```powershell
$body = @{
    message = 'Summarize this sentence.'
    context = 'Plants use sunlight to make sugars.'
} | ConvertTo-Json -Compress
Invoke-RestMethod http://127.0.0.1:8000/api/chat -Method Post -ContentType 'application/json' -Body $body
```

To see events as they arrive on macOS, change the endpoint to `/api/chat/stream` and add curl's `-N` option. In PowerShell, use the same `$body` variable with `curl.exe`.

```powershell
$body | curl.exe -fsS -N http://127.0.0.1:8000/api/chat/stream -H 'Content-Type: application/json' --data-binary '@-'
```

For API-only use on macOS, activate the virtual environment and run `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`. On Windows, run the backend command from the Windows setup.

## Generation settings

Both engines and their API endpoints use the same server-wide profile. Set `KAAL_DECODING` to a JSON object when starting the server to override individual fields. On macOS, for example, run the following.

```sh
KAAL_DECODING='{"do_sample":true,"max_new_tokens":128}' ./start.sh
```

On Windows, set the variable before running the backend command.

```powershell
$env:KAAL_DECODING = '{"do_sample":true,"max_new_tokens":128}'
```

| Settings | Default |
| --- | --- |
| `do_sample` | `false` (greedy decoding) |
| `max_new_tokens` | `256` (valid range 1–2047) |
| `repetition_penalty`, `no_repeat_ngram_size` | `1.18`, `3` |
| `temperature`, `top_k`, `top_p`, `min_p` | `0.45`, `35`, `0.90`, `0.0` (sampling only) |
| `frequency_penalty`, `presence_penalty` | `0.0`, `0.0` |

For Python callers, pass a `DecodingConfig` as `decoding_config` when constructing the engine. The 2,048-token window includes the prompt, reference context, and reserved reply tokens. Long inputs can lose text from the beginning.

## Troubleshooting

- **Weights missing** — Check both filenames under `weights/` and restart the server. The API can start without the files and reports them as missing.
- **Checkpoint or tokenizer fails to load** — Use the matching files from Trinetra. The loader expects a tensor state dictionary, either at the top level or under a `model` key. It uses restricted checkpoint loading and checks that model parameters and tokenizer IDs match.
- **Port 8000 or 5173 is in use** — Stop the process using that port and rerun the launch command for your operating system. On Windows, `Get-NetTCPConnection -LocalPort 8000,5173 -State Listen` shows the process IDs using those ports.
- **macOS uses CPU** — Run `uname -m` and check for `arm64`. If your terminal is running under Rosetta, open a native terminal and recreate the virtual environment with Apple Silicon Python. MPS requires a supported macOS and PyTorch build.
- **Windows blocks npm.ps1 or environment activation** — Use `npm.cmd` and `.\.venv\Scripts\python.exe` as shown above. These commands work without activating the environment.
- **Windows uses CPU on NVIDIA** — Check the backend log and verify CUDA with `.\.venv\Scripts\python.exe -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"`. If the output is `None False`, install the CUDA build in a fresh virtual environment. If CUDA is installed but unavailable, update the NVIDIA driver or use CPU.
- **Windows runs low on GPU or system memory** — Close other memory-heavy apps. Use `KAAL_DEVICE=cpu` as shown above if GPU memory is limited. CPU inference still needs enough system RAM for the checkpoint and generation.
- **Unexpected answers** — KAAL-304M can get facts wrong, lose context, and misunderstand instructions. Try a short prompt with one clear task and verify important outputs.

## Development and contributions

| Path | Purpose |
| --- | --- |
| `backend/engine/` | macOS inference engine and shared architecture and generation configuration |
| `backend/main.py` | macOS FastAPI entry point |
| `backend/windows/` | Windows CUDA/CPU inference engine and FastAPI entry point |
| `backend/tests/` | API, engine, and inference parity tests |
| `frontend/` | React chat and its Vite development server |
| `weights/` | Local model files, excluded from source control |

For changes, keep pull requests focused and include reproduction steps or test results. Run the checks below from the repository root after setup. Tests that require the KAAL-304M files skip when they are missing.

```sh
python -m pytest -q
npm --prefix frontend run build
```

In Windows PowerShell, use `.\.venv\Scripts\python.exe -m pytest -q` and `npm.cmd --prefix frontend run build`.

See [SECURITY.md](SECURITY.md) for reporting security issues. The inference software uses the [MIT License](LICENSE). KAAL-304M is developed by Trinetra AI Research and the software uses PyTorch, Hugging Face Tokenizers, FastAPI, React, and Vite. Dependencies carry their own license notices, including the SIL Open Font License for the frontend fonts. Keep the applicable notices if you redistribute a built app.
