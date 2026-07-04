#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHONPATH="$ROOT" HF_HOME="${HF_HOME:-$HOME/ai-models/hf-cache}" \
  "$ROOT/.venv/bin/python" -m uvicorn ui.backend.app:app --port "${PORT:-8723}" "$@"
