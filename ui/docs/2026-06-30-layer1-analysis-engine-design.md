# Layer-1 spec: Analysis Engine + Journey Data Contract

The data spine for the orka Compression-Journey Visualizer. Turns an HF model name into a
single **journey JSON** that every UI layer renders: the model's architecture, the
step-by-step compression pipeline, and each trick's effect on ratio + perplexity. Two
modes behind one contract - **static** (instant, config + header only, *estimated*
numbers) and **live** (queued GPU job, real `pack` + `eval`, *measured* numbers). Layer-1
is backend-only and shell-agnostic (browser or Electron both consume the same JSON).

## Goals / non-goals

| In scope (layer-1) | Out of scope |
|---|---|
| `GET /analyze` - instant static journey (any HF model, no weights) | Any UI rendering (layers 2/3) |
| `POST /pack` + job queue + SSE progress - live GPU journey | Trained effect-predictor (v1 estimator is a transparent heuristic) |
| The **journey JSON schema** (the contract everything renders) | Auth, multi-user, public hosting (local single-GPU tool) |
| Static estimator (ratio+ppl per trick, from measured RD anchors, arch-gated) | Editing/saving artifacts from the UI; model training |
| Live runner wrapping orka `pack_checkpoint` + `eval_artifact` + per-stage telemetry | Electron packaging (layer-2 decision) |
| FastAPI app under `ui/backend/`, importing orka as a library | Changes to orka's core compression code |

## Architecture

```
ui/
  backend/
    app.py            FastAPI: routes, CORS, static mount (later serves the SPA)
    schema.py         Pydantic models = THE journey contract (single source of truth)
    fetch.py          HF config.json + safetensors header (range GET, no weights)
    arch.py           model meta + ArchProfile -> architecture section (uses orka.quant)
    estimator.py      static per-trick ratio+ppl estimate (heuristic from RD anchors)
    journey.py        assemble the static journey JSON from arch + estimator
    jobs.py           single-GPU serial job queue + SSE progress bus
    live.py           run orka pack_checkpoint + eval_artifact, capture telemetry
    settings.py       config: GPU cap (10GB), HF cache dir, model size ceiling
    tests/
  docs/               this spec
```

**Dependency direction:** `ui/backend` → `orka` (one-way). orka stays the compression
engine, untouched; the backend is a consumer that builds viz-data from orka primitives
(`ArchProfile`, `pack_checkpoint`, `eval_artifact`, `classify_tensor_family`,
`core._checkpoint._tensor_shapes/_read_config_value`). No orka code change required.

**SRP boundaries:** `fetch` knows HTTP/HF; `arch` knows model structure; `estimator` knows
the RD heuristics; `live` knows the GPU pipeline; `jobs` knows scheduling; `schema` is the
only place the contract is defined. Each is independently testable.

## The Journey Data Contract (the deliverable)

One schema, produced identically by static and live paths; fields carry a `source` tag
(`"estimated"` | `"measured"`) so the UI can show provenance and never present an estimate
as a fact. Top-level shape:

```jsonc
{
  "model": {
    "name": "Qwen/Qwen2.5-0.5B",
    "params_total": 494032768,
    "dtype": "bfloat16",
    "vocab_size": 151936,
    "tie_word_embeddings": true,
    "fp16_bytes": 988065536
  },
  "architecture": {                      // from ArchProfile + classify_tensor_family
    "arch_class": "dense" | "moe" | "mamba_hybrid" | "conv_hybrid",
    "flags": { "tied_head": true, "has_moe": false, "has_ssm": false },
    "param_breakdown": [                 // for the architecture view + 3D param histogram
      { "family": "embedding", "params": 136118272, "pct": 27.6, "role": "head+embed(tied)" },
      { "family": "attention", "params": ..., "pct": ... },
      { "family": "mlp",       "params": ..., "pct": ... }
    ],
    "layers": [                          // per-layer block list for the architecture diagram
      { "index": 0, "modules": [ { "name": "self_attn.q_proj", "shape": [512,896],
        "family": "attention", "role": "attn.q",
        "treatment": "quantize" | "keep_fp16" | "skip_error_comp" } ] }
    ]
  },
  "pipeline": [                          // the ordered journey stepper
    { "id": "load",      "title": "Load checkpoint", "summary": "...", "io": {...} },
    { "id": "transform", "title": "Normalize / rotate", ... },
    { "id": "allocate",  "title": "Bit allocation (uniform 3bpw)", ... },
    { "id": "codebook",  "title": "Learn RVQ codebooks", "sample_tensor": {...} },
    { "id": "quantize",  "title": "Assign indices + residual", ... },
    { "id": "strategies","title": "Post-assign (error-comp / em-aq / mse-scale)", ... },
    { "id": "pack",      "title": "Write index planes + manifest", ... }
  ],
  "tricks": [                            // the Trick Lab catalogue
    { "id": "bpw", "label": "Bits per weight", "kind": "scalar", "range": [2.5,4.0],
      "default": 3.0, "applies": true, "why": "uniform beats per-tensor on <1.5B" },
    { "id": "keep_head_fp16", "kind": "toggle", "default": true,
      "applies": true, "gated_by": "tied_head",
      "why": "tied head IS the logit projection; quantizing it explodes ppl" },
    { "id": "lattice", "kind": "toggle", "default": false, "applies": true,
      "warn": "Pareto-loses to VQ on hybrid archs" },
    { "id": "error_comp", "kind": "toggle", "default": false,
      "applies_per_tensor": "skipped on head + recurrent (ArchProfile)" }
    // + rvq_stages, group_size, em_aq, mse_scale, hessian, outliers, rotation
  ],
  "result": {                            // headline outcome for the current config
    "source": "estimated" | "measured",
    "bpw": 3.0, "ratio": 4.3, "fp16_mb": 988, "orka_mb": 230,
    "ppl_base": 20.9, "ppl_orka": 28.3, "ppl_ratio": 1.35,
    "trusted": true, "trust_reason": null,     // measured path only (see Live)
    "notes": ["tied head kept fp16 (auto)", "estimate from RD anchors"]
  }
}
```

The Pydantic models in `schema.py` are authoritative; this JSON is illustrative. Versioned
with a `schema_version` field so the UI can guard.

## Endpoints

| Method | Path | Mode | Returns |
|---|---|---|---|
| GET | `/analyze?model=<id>&bpw=&tricks=` | static, sync | full journey JSON, `result.source="estimated"` |
| POST | `/pack` `{model, config}` | live, async | `{ job_id }` |
| GET | `/jobs/{id}` | - | `{ status, journey? }` (journey present when done) |
| GET | `/jobs/{id}/stream` | - | SSE: stage-progress events, then final journey |

`/analyze` is the default fast path - hit on every model search and every Trick-Lab toggle
(recompute estimates client-trigger, server-compute). `/pack` is opt-in ("Run for real").

## Static estimator (the honest heuristic)

Given the param breakdown + config + the chosen trick config, compute ratio + ppl **without
weights**, from measured anchors (memory: `orka-compression-frontier-by-model`):

- **ratio** = `fp16_bytes / (quantized_bytes + fp16_passthrough_bytes)`, where the quantized
  fraction = body params at `bpw` (≈`bpw/16` of fp16) + per-stage codebook overhead; the
  passthrough fraction = head/embed kept fp16 when `tied_head` (vocab-width via ArchProfile).
  This is arithmetic, not a guess - the only estimated part is the codebook/overhead constant.
- **ppl_ratio** = piecewise from the measured RD curve (2.5→~2.2x, 2.75→~1.6x, 3.0→~1.35x,
  3.5→~1.47... interpolated), shifted by arch flags: untied→best, tied huge-vocab→worse,
  MoE→good, lattice-on-hybrid→penalty, sub-3bpw→broken band. Every output labeled
  `source:"estimated"` with the anchor it came from in `notes`.
- **applicability** of each trick is decided by `ArchProfile` (e.g. `error_comp` shows
  "skipped on head+recurrent"; `lattice` shows the hybrid warning) - never hardcoded names.

The estimator is a pure function `(arch, config) -> result` - trivially unit-testable, and
clearly upgradeable to a fitted predictor later (out of scope for v1).

## Static shape fetch (no weight download)

Static mode needs per-tensor shapes but not weights. Fetch:
1. `config.json` (KB) via `huggingface_hub.hf_hub_download`.
2. The **safetensors header** via HTTP range GET on the resolve URL - the first 8 bytes give
   the header length, then the header JSON gives every tensor name+shape. KB, not GB.
   Sharded models: read each shard's header (enumerate via `model.safetensors.index.json`).
3. Build `ArchProfile.from_shapes(shapes, vocab)` + the param breakdown.

Gated/private models use the user's `HF_TOKEN` (env, never logged - reuse this session's
handling). Non-safetensors (.bin) models: header read unavailable → return arch from config
only (layer list without exact shapes), flagged `partial: true`.

## Live pack runner

`POST /pack` enqueues a job; a **single-GPU serial queue** (the crash lesson - never two GPU
jobs at once) runs it:
1. Resolve + download the model (HF cache).
2. `pack_checkpoint(..., codebook_sizes=[4096,4096], em_aq_passes=3, keep_head_fp16="auto",
   awq_calibration=prompts, max_gpu_mem_gb=10, backend="torch", device="cuda")` - the
   validated config; capture per-stage progress via the existing `--progress-file` hook →
   SSE events.
3. `eval_artifact(...)` → real `ratio`, `ppl_base/orka`, and the **`trusted` flag**.
   *(Dependency: this rides on the reliable-eval hardening - the separate brainstorm. Until
   that lands, `trusted` may be null; the UI shows "unverified" rather than a false number.)*
4. Emit the same journey JSON with `result.source="measured"`.

**Guards:** refuse models above a param ceiling that won't fit the 10GB cap (return a clear
"too large for live mode, static estimate only" error); one job at a time; cancel on
disconnect.

## Data flow

```
ANALYZE (sync):  GET /analyze ─► fetch(config+header) ─► arch(ArchProfile) ─►
                 estimator(config) ─► journey JSON (estimated)         [<1s]

PACK (async):    POST /pack ─► enqueue ─► [GPU serial] download ─► pack ──SSE──►
                 eval ─► journey JSON (measured, trusted?)            [minutes]
```

## Error handling

| Case | Behaviour |
|---|---|
| Model id not found / 404 | 404 with message; no partial journey |
| Gated/private, no token | 403 with "needs HF_TOKEN" hint |
| Non-safetensors weights | static journey with `partial:true` (config-only arch) |
| Live: model too big for 10GB | reject job with "static only" reason |
| Live: pack/eval raises | job → `failed` with the error; never a fake number |
| Live: eval untrusted (NaN/degenerate) | journey returned with `trusted:false` + reason |
| Two live requests | second queues behind the first (serial) |

## Testing

- **schema**: round-trip the example journey through the Pydantic models; `schema_version` present.
- **estimator** (pure, no network/GPU): known breakdowns → expected ratio arithmetic; arch
  flags shift ppl band correctly; trick applicability matches ArchProfile.
- **fetch**: header parse from a captured safetensors header fixture; sharded index handling;
  .bin → partial.
- **arch**: a synthetic config+shapes → expected param_breakdown + flags (reuse the meta-build
  trick from the brutal-audit: instantiate on `meta`, no weights).
- **endpoints**: `/analyze` happy path + 404/403 (mock `fetch`); `/pack` enqueues + SSE shape
  (mock `live`).
- **live smoke** (GPU, opt-in, not CI): one small model end-to-end → measured journey.

## Dependencies + open questions

- **Depends on:** orka (`ArchProfile`, `pack_checkpoint`, `eval_artifact`); the reliable-eval
  hardening for the `trusted` flag (separate spec - layer-1 degrades gracefully without it).
- **Open (resolve in plan):** exact codebook-overhead constant for the ratio estimate (fit
  from the 5 models we already packed); SSE vs WebSocket for progress (SSE is simpler, lean
  SSE); FastAPI run via `uvicorn` standalone vs later Electron sidecar (layer-2).

## Out of scope / future

Layers 2 (visualizer) + 3 (3D) consume this contract. A trained effect-predictor replaces the
heuristic estimator later. Electron shell is a layer-2 packaging choice. The contract is the
stable interface across all of it.
