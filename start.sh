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
    echo "Error: Python 3 is not installed. Please install Python 3.10+."
    exit 1
fi

PY_VER=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON_BIN -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON_BIN -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "Error: Python 3.10+ is required (found $PY_VER)."
    exit 1
fi
echo "✓ Python $PY_VER detected ($PYTHON_BIN)"

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
    python3 -m venv "$DIR/.venv"
    "$DIR/.venv/bin/pip" install --upgrade pip
    "$DIR/.venv/bin/pip" install -r "$DIR/backend/requirements.txt"
fi
UVICORN_EXEC="$DIR/.venv/bin/uvicorn"

# 5. Frontend dependencies
if [ ! -d "$DIR/frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    (cd "$DIR/frontend" && npm install)
fi

# 6. Weights verification
if [ ! -f "$DIR/weights/best_model.pt" ] || [ ! -f "$DIR/weights/tokenizer.json" ]; then
    echo "Notice: Model weights missing in ./weights/ (best_model.pt or tokenizer.json). Server will start in standby mode."
fi

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
trap cleanup SIGINT SIGTERM EXIT

# 7. Start FastAPI Backend
echo "Starting FastAPI backend on http://127.0.0.1:8000..."
"$UVICORN_EXEC" backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# 8. Start Vite Frontend
echo "Starting Vite frontend on http://127.0.0.1:5173..."
(cd "$DIR/frontend" && npm run dev) &
FRONTEND_PID=$!

# 9. Wait for Backend health check
echo "Waiting for backend to be ready..."
RETRIES=30
while [ $RETRIES -gt 0 ]; do
    if curl -s http://127.0.0.1:8000/api/health > /dev/null 2>&1; then
        echo "✓ Backend is healthy!"
        break
    fi
    sleep 1
    RETRIES=$((RETRIES - 1))
done

if [ $RETRIES -eq 0 ]; then
    echo "Warning: Backend took longer than expected to initialize. Check logs above."
fi

# 10. Open browser
if command -v open &>/dev/null; then
    open "http://localhost:5173" 2>/dev/null || true
fi

echo ""
echo "=========================================================="
echo "  🚀 KAAL-304M Local is running!"
echo "  • Frontend: http://localhost:5173 (or http://127.0.0.1:5173)"
echo "  • Backend:  http://127.0.0.1:8000 (API docs at /docs)"
echo "  Press Ctrl+C to stop both servers."
echo "=========================================================="
echo ""

wait
