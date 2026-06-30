# orka compression-journey analysis engine (layer-1)

Backend for the visualizer. Turns an HF model name into one journey JSON.

## Run

    PYTHONPATH=<repo-root> HF_HOME=~/ai-models/hf-cache \
      <repo>/.venv/bin/python -m uvicorn ui.backend.app:app --reload --port 8723

or `ui/backend/run.sh`.

## Endpoints

- `GET  /analyze?model=<id>&bpw=3.0&keep_head=true&lattice=false` - instant static (estimated)
- `POST /pack {"model": "<id>"}` -> `{job_id}` - queued GPU job (measured)
- `GET  /jobs/{id}` - status (+ journey when done)
- `GET  /jobs/{id}/stream` - SSE progress, then the measured journey

## Notes

- Static = config + safetensors header only (no weights). Numbers labeled `estimated`.
- Live = single GPU job at a time, 10GB cap, validated pack config. Numbers `measured`.
- `trusted` rides on the reliable-eval hardening; may be null until that lands.
- Gated/private models: set `HF_TOKEN` in the environment (never logged).
