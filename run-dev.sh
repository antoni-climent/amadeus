#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ENV="$ROOT_DIR/.amadeus_env"
TTS_ENV="$ROOT_DIR/.tts_env"
FRONTEND_DIR="$ROOT_DIR/frontend"

if [[ ! -x "$BACKEND_ENV/bin/python" ]]; then
  echo "Missing backend env: $BACKEND_ENV"
  exit 1
fi

if [[ ! -x "$TTS_ENV/bin/python" ]]; then
  echo "Missing TTS env: $TTS_ENV"
  exit 1
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Missing frontend dependencies. Run: cd frontend && npm install"
  exit 1
fi

cleanup() {
  jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export AMADEUS_TTS_PYTHON="$TTS_ENV/bin/python"
export AMADEUS_TTS_URL="${AMADEUS_TTS_URL:-http://127.0.0.1:8001}"

(
  cd "$ROOT_DIR"
  source "$TTS_ENV/bin/activate"
  exec python -m backend.tts_worker
) &

source "$TTS_ENV/bin/activate"
for _ in $(seq 1 180); do
  if python - <<'PY'
from urllib.request import urlopen
try:
    with urlopen("http://127.0.0.1:8001/health", timeout=1) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
  then
    break
  fi
  sleep 1
done

(
  cd "$ROOT_DIR"
  source "$BACKEND_ENV/bin/activate"
  exec python -m backend.api
) &

(
  cd "$FRONTEND_DIR"
  exec npm run dev -- --host 127.0.0.1
) &

wait
