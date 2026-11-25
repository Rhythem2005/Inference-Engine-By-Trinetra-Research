#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=== KAAL-304M Local Server Launcher ==="

# 1. macOS check
if [ "$(uname -s)" != "Darwin" ]; then
    echo "Warning: KAAL-304M is optimized for macOS with Apple Silicon. Running on $(uname -s)."
fi

# 2. Python check
PYTHON_BIN=""
if [ -x "$DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$DIR/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    echo "Error: Python 3 is not installed. Please install Python 3.11+."
    exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    echo "Error: Python 3.11 or newer is required."
    exit 1
fi

# 3. Node check
if ! command -v node &>/dev/null; then
    echo "Error: Node.js is not installed. Please install Node 18+."
    exit 1
fi
NODE_VER=$(node -v | sed 's/v//')
NODE_MAJOR=$(echo "$NODE_VER" | cut -d. -f1)
if [ "$NODE_MAJOR" -lt 18 ]; then
    echo "Error: Node 18+ is required (found v$NODE_VER)."
    exit 1
fi
echo "✓ Node.js v$NODE_VER detected"

# 4. Virtual environment setup
if [ ! -d "$DIR/.venv" ]; then
    echo "Creating virtual environment at .venv..."
    "$PYTHON_BIN" -m venv "$DIR/.venv"
fi
if ! "$DIR/.venv/bin/python" -c 'import fastapi, tokenizers, torch, uvicorn' 2>/dev/null; then
    "$DIR/.venv/bin/python" -m pip install --upgrade pip setuptools
    "$DIR/.venv/bin/python" -m pip install -r "$DIR/backend/requirements.txt"
fi

# 5. Frontend dependencies
if [ ! -x "$DIR/frontend/node_modules/.bin/vite" ]; then
    echo "Installing frontend dependencies..."
    (cd "$DIR/frontend" && npm ci)
fi

# 6. Weights verification
if [ ! -f "$DIR/weights/best_model.pt" ] || [ ! -f "$DIR/weights/tokenizer.json" ]; then
    echo "Notice: Model weights missing in ./weights/ (best_model.pt or tokenizer.json). Server will start in standby mode."
fi

for port in 8000 5173; do
    if curl -s --max-time 1 "http://127.0.0.1:$port/" >/dev/null 2>&1; then
        echo "Error: Port $port is already serving HTTP. Stop that server before starting KAAL-304M."
        exit 1
    fi
done

# Cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo "All servers stopped."
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 7. Start FastAPI Backend
echo "Starting FastAPI backend on http://127.0.0.1:8000..."
"$DIR/.venv/bin/python" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# 8. Start Vite Frontend
echo "Starting Vite frontend on http://127.0.0.1:5173..."
(cd "$DIR/frontend" && exec ./node_modules/.bin/vite) &
FRONTEND_PID=$!

# 9. Wait for Backend health check
echo "Waiting for servers to be ready..."
RETRIES=30
while [ $RETRIES -gt 0 ]; do
    if ! kill -0 "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null; then
        echo "Error: A server exited during startup. Check the output above."
        exit 1
    fi
    if curl -fs --max-time 1 http://127.0.0.1:8000/api/health >/dev/null &&
       curl -fs --max-time 1 http://127.0.0.1:5173/ >/dev/null; then
        break
    fi
    sleep 1
    RETRIES=$((RETRIES - 1))
done

if [ $RETRIES -eq 0 ]; then
    echo "Error: Servers did not become ready. Check the output above."
    exit 1
fi

# 10. Open browser
if command -v open &>/dev/null; then
    open "http://localhost:5173" 2>/dev/null || true
fi

echo ""
echo "KAAL-304M is ready at http://localhost:5173."
echo "API docs: http://127.0.0.1:8000/docs"
echo "Press Ctrl+C to stop both servers."
echo ""

wait
