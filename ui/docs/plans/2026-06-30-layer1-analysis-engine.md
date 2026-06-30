# Layer-1 Analysis Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the FastAPI backend that turns an HF model name into one "journey JSON" - static (instant, estimated) and live (queued GPU, measured) - for the compression-journey visualizer.

**Architecture:** A FastAPI app under `ui/backend/` that imports `orka` one-way. Static path: fetch `config.json` + safetensors header (range GET, no weights) -> `ArchProfile` -> estimated ratio/ppl from measured RD anchors. Live path: a single-GPU serial job queue runs `pack_checkpoint` + `eval_artifact` and streams progress over SSE. One Pydantic schema (`schema.py`) is the contract both paths emit.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, httpx, huggingface_hub, pydantic v2, sse-starlette, pytest + pytest-asyncio. Reuses orka (`orka.quant.ArchProfile`, `classify_tensor_family`, `orka.pipeline.pack.pack_checkpoint`, `orka.eval.eval_artifact`).

---

## File Structure

| File | Responsibility |
|---|---|
| `ui/backend/__init__.py` | package marker |
| `ui/backend/requirements.txt` | backend deps (orka installed separately / on PYTHONPATH) |
| `ui/backend/settings.py` | env-driven config: GPU cap, HF cache, live param ceiling, schema version |
| `ui/backend/schema.py` | Pydantic journey contract - the single source of truth |
| `ui/backend/fetch.py` | HF `config.json` + safetensors header range-fetch (no weights) |
| `ui/backend/arch.py` | config+shapes -> `Architecture` section (uses `ArchProfile`) |
| `ui/backend/estimator.py` | static per-config ratio+ppl estimate (heuristic from RD anchors) |
| `ui/backend/pipeline_steps.py` | the static pipeline-stage + trick catalogue (arch-gated) |
| `ui/backend/journey.py` | assemble the static `Journey` from arch + estimator + catalogue |
| `ui/backend/jobs.py` | single-GPU serial job queue + per-job progress bus |
| `ui/backend/live.py` | run `pack_checkpoint` + `eval_artifact`, emit measured `Journey` |
| `ui/backend/app.py` | FastAPI routes: `/analyze`, `/pack`, `/jobs/{id}`, `/jobs/{id}/stream` |
| `ui/backend/tests/*` | one test module per source module |

All tasks run from repo root with `PYTHONPATH=$PWD` and the repo venv (`.venv/bin/python`), so `import orka` resolves.

---

### Task 1: Scaffold + settings

**Files:**
- Create: `ui/backend/__init__.py` (empty)
- Create: `ui/backend/tests/__init__.py` (empty)
- Create: `ui/backend/requirements.txt`
- Create: `ui/backend/settings.py`
- Test: `ui/backend/tests/test_settings.py`

- [ ] **Step 1: Write requirements.txt**

```
fastapi>=0.115
uvicorn[standard]>=0.30
httpx>=0.27
huggingface_hub>=0.25
pydantic>=2.7
sse-starlette>=2.1
pytest>=8
pytest-asyncio>=0.23
```

- [ ] **Step 2: Write the failing test**

```python
# ui/backend/tests/test_settings.py
from ui.backend import settings

def test_defaults_present():
    assert settings.GPU_MEM_CAP_GB == 10.0          # orka 10GB cap rule
    assert settings.SCHEMA_VERSION >= 1
    assert settings.LIVE_PARAM_CEILING > 0
    assert isinstance(settings.HF_CACHE, str)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_settings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ui.backend.settings'`

- [ ] **Step 4: Write settings.py**

```python
# ui/backend/settings.py
"""Env-driven config for the analysis-engine backend."""
import os
from pathlib import Path

GPU_MEM_CAP_GB = float(os.environ.get("ORKA_UI_GPU_CAP_GB", "10"))   # orka 10GB cap
HF_CACHE = os.environ.get("HF_HOME", str(Path.home() / "ai-models" / "hf-cache"))
LIVE_PARAM_CEILING = int(os.environ.get("ORKA_UI_LIVE_PARAM_CEILING", str(2_000_000_000)))
HF_TOKEN = os.environ.get("HF_TOKEN")          # never logged
SCHEMA_VERSION = 1
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_settings.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add ui/backend/__init__.py ui/backend/tests/__init__.py ui/backend/requirements.txt ui/backend/settings.py ui/backend/tests/test_settings.py
git commit -m "feat(ui): backend scaffold + settings"
```

---

### Task 2: Journey schema (the contract)

**Files:**
- Create: `ui/backend/schema.py`
- Test: `ui/backend/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# ui/backend/tests/test_schema.py
from ui.backend.schema import Journey

EXAMPLE = {
    "schema_version": 1,
    "model": {"name": "x/y", "params_total": 100, "dtype": "bfloat16",
              "vocab_size": 32, "tie_word_embeddings": True, "fp16_bytes": 200},
    "architecture": {"arch_class": "dense", "flags": {"tied_head": True, "has_moe": False, "has_ssm": False},
                     "param_breakdown": [{"family": "mlp", "params": 100, "pct": 100.0, "role": ""}],
                     "layers": [{"index": 0, "modules": [
                         {"name": "model.layers.0.mlp.down_proj", "shape": [8, 8],
                          "family": "mlp", "treatment": "quantize"}]}],
                     "partial": False},
    "pipeline": [{"id": "load", "title": "Load", "summary": "..."}],
    "tricks": [{"id": "bpw", "label": "Bits/weight", "kind": "scalar", "default": 3.0,
                "applies": True, "why": "", "warn": None, "gated_by": None}],
    "result": {"source": "estimated", "bpw": 3.0, "ratio": 4.3, "fp16_mb": 0.2, "orka_mb": 0.05,
               "ppl_base": None, "ppl_orka": None, "ppl_ratio": 1.35,
               "trusted": None, "trust_reason": None, "notes": ["estimated"]},
}

def test_round_trip():
    j = Journey.model_validate(EXAMPLE)
    assert j.schema_version == 1
    assert j.model.tie_word_embeddings is True
    assert j.architecture.layers[0].modules[0].treatment == "quantize"
    assert j.result.source == "estimated"
    # re-dump and re-load stays equal
    assert Journey.model_validate(j.model_dump()).result.ratio == 4.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write schema.py**

```python
# ui/backend/schema.py
"""Pydantic models = the journey contract. Single source of truth for what every UI
layer renders. Versioned via schema_version."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Source = Literal["estimated", "measured"]
Treatment = Literal["quantize", "keep_fp16", "skip_error_comp"]
ArchClass = Literal["dense", "moe", "mamba_hybrid", "conv_hybrid", "hybrid"]


class ModelMeta(BaseModel):
    name: str
    params_total: int
    dtype: str
    vocab_size: int | None = None
    tie_word_embeddings: bool = False
    fp16_bytes: int


class FamilyBreakdown(BaseModel):
    family: str
    params: int
    pct: float
    role: str = ""


class ModuleEntry(BaseModel):
    name: str
    shape: list[int]
    family: str
    treatment: Treatment


class LayerBlock(BaseModel):
    index: int
    modules: list[ModuleEntry]


class Architecture(BaseModel):
    arch_class: ArchClass
    flags: dict[str, bool]
    param_breakdown: list[FamilyBreakdown]
    layers: list[LayerBlock]
    partial: bool = False


class Stage(BaseModel):
    id: str
    title: str
    summary: str


class Trick(BaseModel):
    id: str
    label: str
    kind: Literal["scalar", "toggle"]
    default: float | bool
    applies: bool
    why: str = ""
    warn: str | None = None
    gated_by: str | None = None


class Result(BaseModel):
    source: Source
    bpw: float
    ratio: float
    fp16_mb: float
    orka_mb: float
    ppl_base: float | None = None
    ppl_orka: float | None = None
    ppl_ratio: float | None = None
    trusted: bool | None = None
    trust_reason: str | None = None
    notes: list[str] = []


class Journey(BaseModel):
    schema_version: int
    model: ModelMeta
    architecture: Architecture
    pipeline: list[Stage]
    tricks: list[Trick]
    result: Result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_schema.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/backend/schema.py ui/backend/tests/test_schema.py
git commit -m "feat(ui): journey JSON contract (pydantic schema)"
```

---

### Task 3: HF fetch (config + safetensors header, no weights)

**Files:**
- Create: `ui/backend/fetch.py`
- Test: `ui/backend/tests/test_fetch.py`

- [ ] **Step 1: Write the failing test** (parses a synthetic safetensors header; no network)

```python
# ui/backend/tests/test_fetch.py
import json
import struct

import httpx
import pytest

from ui.backend import fetch


def _st_bytes(header: dict) -> bytes:
    blob = json.dumps(header).encode()
    return struct.pack("<Q", len(blob)) + blob


def test_parse_header_from_bytes(monkeypatch):
    header = {"__metadata__": {"x": "y"},
              "lm_head.weight": {"dtype": "BF16", "shape": [32, 8], "data_offsets": [0, 512]},
              "model.layers.0.mlp.down_proj.weight": {"dtype": "BF16", "shape": [8, 16], "data_offsets": [512, 768]}}
    raw = _st_bytes(header)

    def fake_get(url, headers=None, **kw):
        rng = headers.get("Range", "")
        start, end = rng.replace("bytes=", "").split("-")
        body = raw[int(start): int(end) + 1]
        return httpx.Response(200, content=body)

    monkeypatch.setattr(fetch.httpx, "get", fake_get)
    shapes = fetch._st_header("http://x/model.safetensors", token=None)
    assert shapes == {"lm_head.weight": (32, 8),
                      "model.layers.0.mlp.down_proj.weight": (8, 16)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write fetch.py**

```python
# ui/backend/fetch.py
"""Fetch an HF model's config + tensor shapes WITHOUT downloading weights.

Shapes come from the safetensors header (first 8 bytes = uint64 header length, then that
many bytes of JSON mapping tensor -> {dtype, shape, data_offsets}) via HTTP range GET - KB,
not GB. Sharded models are enumerated through model.safetensors.index.json."""
from __future__ import annotations

import json
import struct

import httpx

HF_BASE = "https://huggingface.co"


def _auth(token: str | None) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def fetch_config(model: str, token: str | None = None) -> dict:
    url = f"{HF_BASE}/{model}/resolve/main/config.json"
    r = httpx.get(url, headers=_auth(token), follow_redirects=True, timeout=30)
    r.raise_for_status()
    return r.json()


def _st_header(url: str, token: str | None) -> dict:
    """tensor name -> shape tuple from one safetensors file's header (2 range GETs)."""
    a = _auth(token)
    head = httpx.get(url, headers={**a, "Range": "bytes=0-7"}, follow_redirects=True, timeout=30)
    head.raise_for_status()
    n = struct.unpack("<Q", head.content[:8])[0]
    body = httpx.get(url, headers={**a, "Range": f"bytes=8-{8 + n - 1}"},
                     follow_redirects=True, timeout=30)
    body.raise_for_status()
    header = json.loads(body.content)
    return {k: tuple(v["shape"]) for k, v in header.items()
            if k != "__metadata__" and isinstance(v, dict) and "shape" in v}


def fetch_shapes(model: str, token: str | None = None) -> dict:
    """All tensor shapes. Single-file model.safetensors, else the sharded index."""
    base = f"{HF_BASE}/{model}/resolve/main"
    try:
        return _st_header(f"{base}/model.safetensors", token)
    except httpx.HTTPStatusError:
        pass
    idx = httpx.get(f"{base}/model.safetensors.index.json",
                    headers=_auth(token), follow_redirects=True, timeout=30)
    idx.raise_for_status()
    shapes: dict = {}
    for shard in sorted(set(idx.json()["weight_map"].values())):
        shapes.update(_st_header(f"{base}/{shard}", token))
    return shapes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_fetch.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/backend/fetch.py ui/backend/tests/test_fetch.py
git commit -m "feat(ui): HF config + safetensors-header fetch (no weights)"
```

---

### Task 4: Architecture section (ArchProfile-driven)

**Files:**
- Create: `ui/backend/arch.py`
- Test: `ui/backend/tests/test_arch.py`

- [ ] **Step 1: Write the failing test**

```python
# ui/backend/tests/test_arch.py
from ui.backend.arch import build_architecture

CONFIG = {"vocab_size": 32, "tie_word_embeddings": True}
SHAPES = {
    "model.embed_tokens.weight": (32, 8),
    "lm_head.weight": (32, 8),
    "model.layers.0.self_attn.q_proj.weight": (8, 8),
    "model.layers.0.mlp.down_proj.weight": (8, 16),
    "model.layers.0.mamba.A_log": (4,),
    "model.layers.0.mamba.in_proj.weight": (16, 8),
}


def test_flags_and_treatment():
    a = build_architecture(CONFIG, SHAPES)
    assert a.flags["tied_head"] is True
    assert a.flags["has_ssm"] is True
    assert a.arch_class == "hybrid"
    # head is vocab-width -> keep_fp16; mamba in_proj -> skip_error_comp; mlp -> quantize
    treat = {m.name: m.treatment for blk in a.layers for m in blk.modules}
    assert treat["lm_head.weight"] == "keep_fp16"
    assert treat["model.layers.0.mamba.in_proj.weight"] == "skip_error_comp"
    assert treat["model.layers.0.mlp.down_proj.weight"] == "quantize"
    assert any(f.family == "embedding" for f in a.param_breakdown)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_arch.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write arch.py**

```python
# ui/backend/arch.py
"""Build the Architecture section of the journey from config + tensor shapes, using orka's
ArchProfile (structural head/recurrent detection) and classify_tensor_family."""
from __future__ import annotations

import re

from orka.quant import ArchProfile, classify_tensor_family

from .schema import Architecture, FamilyBreakdown, LayerBlock, ModuleEntry

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _treatment(name: str, shape, profile: ArchProfile) -> str:
    if profile.is_output_head(name, shape):
        return "keep_fp16"
    if profile.is_recurrent(name):
        return "skip_error_comp"
    return "quantize"


def build_architecture(config: dict, shapes: dict) -> Architecture:
    vocab = config.get("vocab_size")
    profile = ArchProfile.from_shapes(shapes, vocab)
    tied = bool(config.get("tie_word_embeddings", False))

    fam_params: dict[str, int] = {}
    for name, shape in shapes.items():
        fam = classify_tensor_family(name)
        fam_params[fam] = fam_params.get(fam, 0) + _numel(shape)
    total = sum(fam_params.values()) or 1
    breakdown = [
        FamilyBreakdown(family=f, params=p, pct=round(100 * p / total, 1))
        for f, p in sorted(fam_params.items(), key=lambda kv: -kv[1])
    ]

    has_moe = any("expert" in n.lower() for n in shapes)
    has_ssm = len(profile.recurrent_names) > 0
    arch_class = "moe" if has_moe else ("hybrid" if has_ssm else "dense")
    flags = {"tied_head": tied, "has_moe": has_moe, "has_ssm": has_ssm}

    blocks: dict[int, list[ModuleEntry]] = {}
    for name, shape in shapes.items():
        if len(shape) < 2:
            continue  # 1-D params (norms, A_log, biases) not shown as quant modules
        m = _LAYER_RE.search(name)
        idx = int(m.group(1)) if m else -1
        blocks.setdefault(idx, []).append(ModuleEntry(
            name=name, shape=list(shape),
            family=classify_tensor_family(name),
            treatment=_treatment(name, shape, profile),
        ))
    layers = [LayerBlock(index=i, modules=blocks[i]) for i in sorted(blocks)]

    return Architecture(arch_class=arch_class, flags=flags,
                        param_breakdown=breakdown, layers=layers)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_arch.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/backend/arch.py ui/backend/tests/test_arch.py
git commit -m "feat(ui): architecture section via ArchProfile"
```

---

### Task 5: Static estimator (ratio + ppl from RD anchors)

**Files:**
- Create: `ui/backend/estimator.py`
- Test: `ui/backend/tests/test_estimator.py`

- [ ] **Step 1: Write the failing test**

```python
# ui/backend/tests/test_estimator.py
from ui.backend.arch import build_architecture
from ui.backend.estimator import estimate
from ui.backend.schema import ModelMeta

CONFIG = {"vocab_size": 32, "tie_word_embeddings": True}
SHAPES = {"model.embed_tokens.weight": (32, 8), "lm_head.weight": (32, 8),
          "model.layers.0.mlp.down_proj.weight": (64, 64)}


def _meta():
    total = 32 * 8 + 32 * 8 + 64 * 64
    return ModelMeta(name="x/y", params_total=total, dtype="bfloat16",
                     vocab_size=32, tie_word_embeddings=True, fp16_bytes=total * 2)


def test_estimate_monotonic_and_labeled():
    meta, arch = _meta(), build_architecture(CONFIG, SHAPES)
    r3 = estimate(meta, arch, bpw=3.0, keep_head=True)
    r25 = estimate(meta, arch, bpw=2.5, keep_head=True)
    assert r3.source == "estimated"
    assert r3.ratio > 1.0
    assert r25.ppl_ratio > r3.ppl_ratio       # lower bpw -> worse ppl
    assert r3.notes                            # carries provenance

def test_keep_head_costs_ratio():
    meta, arch = _meta(), build_architecture(CONFIG, SHAPES)
    keep = estimate(meta, arch, bpw=3.0, keep_head=True)
    drop = estimate(meta, arch, bpw=3.0, keep_head=False)
    assert drop.ratio > keep.ratio             # fp16 head is big -> protecting it lowers ratio

def test_lattice_on_hybrid_warns_worse():
    meta = _meta()
    arch = build_architecture({"vocab_size": 32}, {**SHAPES, "model.layers.0.mamba.A_log": (4,),
                              "model.layers.0.mamba.in_proj.weight": (16, 8)})
    base = estimate(meta, arch, bpw=3.0)
    lat = estimate(meta, arch, bpw=3.0, lattice=True)
    assert lat.ppl_ratio > base.ppl_ratio
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_estimator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write estimator.py**

```python
# ui/backend/estimator.py
"""Static, weight-free estimate of compression ratio + perplexity from the measured RD
anchors (orka frontier work). Transparent heuristic, every output labeled source=estimated
with the anchor in notes. Upgradeable to a fitted predictor later (out of scope)."""
from __future__ import annotations

from .schema import Architecture, ModelMeta, Result

# Smoothed ppl-ratio vs bpw for the full config (rvq-12-12 + em-aq + hessian), untied
# baseline. Measured anchors: 3.0 -> 1.345 (artifact), 4.0 -> 1.26 (sweep); the rest
# smoothed monotonic. Labeled estimated; this is the only "guessed" constant.
_PPL_ANCHORS = [(2.5, 2.2), (2.75, 1.6), (3.0, 1.35), (3.5, 1.22), (4.0, 1.15)]
# rvq-12-12 codebook overhead per quantized 2-D tensor: 2 stages * K(4096) * group(8) * 2 B.
_CODEBOOK_OVERHEAD_BYTES = 2 * 4096 * 8 * 2


def _interp(bpw: float) -> float:
    pts = _PPL_ANCHORS
    if bpw <= pts[0][0]:
        return pts[0][1]
    if bpw >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= bpw <= x1:
            return y0 + (y1 - y0) * (bpw - x0) / (x1 - x0)
    return pts[-1][1]


def _vocab_width_params(arch: Architecture) -> int:
    return sum(f.params for f in arch.param_breakdown if f.family == "embedding")


def estimate(meta: ModelMeta, arch: Architecture, bpw: float = 3.0,
             keep_head: bool = True, lattice: bool = False) -> Result:
    total = max(meta.params_total, 1)
    head = _vocab_width_params(arch) if (keep_head and meta.tie_word_embeddings) else 0
    body = max(total - head, 0)

    n_quant_tensors = sum(
        1 for blk in arch.layers for m in blk.modules if m.treatment != "keep_fp16"
    )
    quant_bytes = body * bpw / 8.0 + n_quant_tensors * _CODEBOOK_OVERHEAD_BYTES
    passthrough_bytes = head * 2  # fp16
    orka_bytes = max(quant_bytes + passthrough_bytes, 1.0)
    ratio = meta.fp16_bytes / orka_bytes

    ppl = _interp(bpw)
    notes = [f"estimated from RD anchor bpw={bpw:.2f}->{ppl:.2f}"]
    if arch.flags.get("has_moe"):
        ppl *= 0.97
        notes.append("MoE compresses well (-3%)")
    if meta.tie_word_embeddings and not keep_head:
        ppl *= 1.4
        notes.append("tied head quantized -> ppl penalty")
    if lattice and arch.flags.get("has_ssm"):
        ppl *= 1.3
        notes.append("E8 lattice Pareto-loses on hybrid (+30%)")
    if keep_head and meta.tie_word_embeddings:
        notes.append("tied head+embed kept fp16 (auto)")

    return Result(
        source="estimated", bpw=bpw, ratio=round(ratio, 2),
        fp16_mb=round(meta.fp16_bytes / 1e6, 1), orka_mb=round(orka_bytes / 1e6, 1),
        ppl_ratio=round(ppl, 3), notes=notes,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_estimator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/backend/estimator.py ui/backend/tests/test_estimator.py
git commit -m "feat(ui): static ratio+ppl estimator from RD anchors"
```

---

### Task 6: Pipeline + trick catalogue (arch-gated)

**Files:**
- Create: `ui/backend/pipeline_steps.py`
- Test: `ui/backend/tests/test_pipeline_steps.py`

- [ ] **Step 1: Write the failing test**

```python
# ui/backend/tests/test_pipeline_steps.py
from ui.backend.arch import build_architecture
from ui.backend.pipeline_steps import build_pipeline, build_tricks

DENSE = build_architecture({"vocab_size": 32, "tie_word_embeddings": False},
                           {"model.layers.0.mlp.down_proj.weight": (64, 64)})
HYBRID = build_architecture({"vocab_size": 32},
                            {"model.layers.0.mamba.A_log": (4,),
                             "model.layers.0.mamba.in_proj.weight": (16, 8)})


def test_pipeline_has_ordered_stages():
    ids = [s.id for s in build_pipeline(DENSE)]
    assert ids == ["load", "transform", "allocate", "codebook", "quantize", "strategies", "pack"]


def test_keep_head_trick_gated_by_tie():
    tricks = {t.id: t for t in build_tricks(HYBRID)}
    assert tricks["keep_head_fp16"].gated_by == "tied_head"
    # lattice warns on hybrid
    assert tricks["lattice"].warn is not None
    # error_comp applies but is per-tensor skipped on recurrent (noted in why)
    assert "recurrent" in tricks["error_comp"].why.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_pipeline_steps.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write pipeline_steps.py**

```python
# ui/backend/pipeline_steps.py
"""The static pipeline-stage list (the journey stepper) and the trick catalogue (the Trick
Lab). Trick applicability/warnings are arch-gated from the Architecture flags - never
hardcoded per model."""
from __future__ import annotations

from .schema import Architecture, Stage, Trick

_STAGES = [
    ("load", "Load checkpoint", "Read source weights (safetensors)."),
    ("transform", "Normalize / rotate", "Per-tensor scale + optional Hadamard rotation."),
    ("allocate", "Bit allocation", "Uniform bpw across tensors (uniform beats per-tensor <1.5B)."),
    ("codebook", "Learn RVQ codebooks", "k-means codebooks per stage (rvq-12-12 = 2x K=4096)."),
    ("quantize", "Assign indices + residual", "Nearest-codeword assignment, residual to next stage."),
    ("strategies", "Post-assignment", "error-comp / EM-AQ / mse-scale (arch-gated)."),
    ("pack", "Write artifact", "Index planes + codebooks + manifest."),
]


def build_pipeline(arch: Architecture) -> list[Stage]:
    return [Stage(id=i, title=t, summary=s) for i, t, s in _STAGES]


def build_tricks(arch: Architecture) -> list[Trick]:
    f = arch.flags
    tricks = [
        Trick(id="bpw", label="Bits per weight", kind="scalar", default=3.0, applies=True,
              why="uniform bpw is the <1.5B sweet spot"),
        Trick(id="rvq_stages", label="RVQ stages", kind="scalar", default=2, applies=True,
              why="residual stages stack codebooks (12-12 = 3bpw)"),
        Trick(id="em_aq", label="EM-AQ refine", kind="toggle", default=True, applies=True,
              why="joint codebook refinement, free quality"),
        Trick(id="hessian", label="Hessian weighting", kind="toggle", default=True, applies=True,
              why="biggest free quality lever (1.80->1.35 at 3bpw)"),
        Trick(id="mse_scale", label="MSE-optimal scales", kind="toggle", default=False, applies=True,
              why="least-squares block scales, free quality"),
        Trick(id="keep_head_fp16", label="Keep head fp16", kind="toggle",
              default=bool(f.get("tied_head")), applies=True, gated_by="tied_head",
              why="tied head IS the logit projection; quantizing it explodes ppl"),
        Trick(id="error_comp", label="Error compensation (LDLQ)", kind="toggle", default=False,
              applies=True,
              why="block-OBS; auto-skipped on output head + recurrent/SSM tensors"),
        Trick(id="lattice", label="E8 lattice", kind="toggle", default=False, applies=True,
              warn="Pareto-loses to VQ on hybrid archs" if f.get("has_ssm") else None,
              why="codebook-free QuIP#; wins only on standard transformers at high bpw"),
        Trick(id="outliers", label="Outlier extraction", kind="toggle", default=False, applies=True,
              why="keep top-magnitude weights fp16"),
        Trick(id="rotation", label="Transform search", kind="toggle", default=False, applies=True,
              why="per-tensor normalize/rotate pick"),
    ]
    return tricks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_pipeline_steps.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/backend/pipeline_steps.py ui/backend/tests/test_pipeline_steps.py
git commit -m "feat(ui): arch-gated pipeline + trick catalogue"
```

---

### Task 7: Assemble static journey

**Files:**
- Create: `ui/backend/journey.py`
- Test: `ui/backend/tests/test_journey.py`

- [ ] **Step 1: Write the failing test** (mocks `fetch` so no network)

```python
# ui/backend/tests/test_journey.py
from ui.backend import journey as J
from ui.backend.schema import Journey

CONFIG = {"vocab_size": 32, "tie_word_embeddings": True, "torch_dtype": "bfloat16"}
SHAPES = {"model.embed_tokens.weight": (32, 8), "lm_head.weight": (32, 8),
          "model.layers.0.mlp.down_proj.weight": (64, 64)}


def test_build_static_journey(monkeypatch):
    monkeypatch.setattr(J, "fetch_config", lambda m, token=None: CONFIG)
    monkeypatch.setattr(J, "fetch_shapes", lambda m, token=None: SHAPES)
    j = J.build_static_journey("x/y", bpw=3.0)
    assert isinstance(j, Journey)
    assert j.model.name == "x/y"
    assert j.model.tie_word_embeddings is True
    assert j.result.source == "estimated"
    assert len(j.pipeline) == 7
    assert any(t.id == "keep_head_fp16" for t in j.tricks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_journey.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write journey.py**

```python
# ui/backend/journey.py
"""Assemble the static (estimated) Journey from a model name: fetch config+shapes ->
ModelMeta + Architecture -> estimate -> pipeline + tricks."""
from __future__ import annotations

from .arch import build_architecture
from .estimator import estimate
from .fetch import fetch_config, fetch_shapes
from .pipeline_steps import build_pipeline, build_tricks
from .schema import Journey, ModelMeta
from .settings import HF_TOKEN, SCHEMA_VERSION

_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def build_static_journey(model: str, bpw: float = 3.0, keep_head: bool = True,
                         lattice: bool = False, token: str | None = None) -> Journey:
    token = token or HF_TOKEN
    config = fetch_config(model, token=token)
    shapes = fetch_shapes(model, token=token)

    params_total = sum(_numel(s) for s in shapes.values())
    dtype = config.get("torch_dtype", "bfloat16")
    nbytes = _DTYPE_BYTES.get(dtype, 2)
    meta = ModelMeta(
        name=model, params_total=params_total, dtype=dtype,
        vocab_size=config.get("vocab_size"),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        fp16_bytes=params_total * nbytes,
    )
    arch = build_architecture(config, shapes)
    result = estimate(meta, arch, bpw=bpw, keep_head=keep_head, lattice=lattice)
    return Journey(
        schema_version=SCHEMA_VERSION, model=meta, architecture=arch,
        pipeline=build_pipeline(arch), tricks=build_tricks(arch), result=result,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_journey.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/backend/journey.py ui/backend/tests/test_journey.py
git commit -m "feat(ui): assemble static journey"
```

---

### Task 8: FastAPI app + /analyze endpoint

**Files:**
- Create: `ui/backend/app.py`
- Test: `ui/backend/tests/test_app_analyze.py`

- [ ] **Step 1: Write the failing test** (TestClient, mock journey builder)

```python
# ui/backend/tests/test_app_analyze.py
import httpx
from fastapi.testclient import TestClient

from ui.backend import app as appmod
from ui.backend.app import app

CONFIG = {"vocab_size": 32, "tie_word_embeddings": False, "torch_dtype": "bfloat16"}
SHAPES = {"lm_head.weight": (32, 8), "model.layers.0.mlp.down_proj.weight": (64, 64)}


def test_analyze_ok(monkeypatch):
    monkeypatch.setattr(appmod.journey, "fetch_config", lambda m, token=None: CONFIG)
    monkeypatch.setattr(appmod.journey, "fetch_shapes", lambda m, token=None: SHAPES)
    c = TestClient(app)
    r = c.get("/analyze", params={"model": "x/y", "bpw": 3.0})
    assert r.status_code == 200
    body = r.json()
    assert body["model"]["name"] == "x/y"
    assert body["result"]["source"] == "estimated"


def test_analyze_404(monkeypatch):
    def boom(m, token=None):
        raise httpx.HTTPStatusError("nf", request=None,
                                    response=httpx.Response(404))
    monkeypatch.setattr(appmod.journey, "fetch_config", boom)
    c = TestClient(app)
    r = c.get("/analyze", params={"model": "no/such"})
    assert r.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_app_analyze.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write app.py**

```python
# ui/backend/app.py
"""FastAPI routes for the analysis engine. /analyze is the instant static path; /pack and
/jobs (Task 10) are the live GPU path."""
from __future__ import annotations

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import journey

app = FastAPI(title="orka compression-journey analysis engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/analyze")
def analyze(model: str = Query(...), bpw: float = 3.0,
            keep_head: bool = True, lattice: bool = False):
    try:
        j = journey.build_static_journey(model, bpw=bpw, keep_head=keep_head, lattice=lattice)
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code if exc.response is not None else 502
        if code == 404:
            raise HTTPException(404, f"model not found: {model}")
        if code in (401, 403):
            raise HTTPException(403, "model gated/private - set HF_TOKEN")
        raise HTTPException(502, f"HF fetch failed ({code})")
    return j.model_dump()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_app_analyze.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/backend/app.py ui/backend/tests/test_app_analyze.py
git commit -m "feat(ui): FastAPI /analyze (static journey) endpoint"
```

---

### Task 9: Single-GPU serial job queue + progress bus

**Files:**
- Create: `ui/backend/jobs.py`
- Test: `ui/backend/tests/test_jobs.py`

- [ ] **Step 1: Write the failing test**

```python
# ui/backend/tests/test_jobs.py
import asyncio

import pytest

from ui.backend.jobs import JobQueue


@pytest.mark.asyncio
async def test_serial_execution_and_progress():
    q = JobQueue()
    await q.start()
    order = []

    async def runner(job_id, emit, *, tag):
        emit({"stage": "begin", "tag": tag})
        await asyncio.sleep(0.01)
        order.append(tag)
        emit({"stage": "done", "tag": tag})
        return {"tag": tag}

    id1 = q.submit(runner, tag="a")
    id2 = q.submit(runner, tag="b")
    r1 = await q.wait(id1)
    r2 = await q.wait(id2)
    assert r1["tag"] == "a" and r2["tag"] == "b"
    assert order == ["a", "b"]                 # serial, in submit order
    assert q.status(id1) == "done"
    await q.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_jobs.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write jobs.py**

```python
# ui/backend/jobs.py
"""Single-GPU serial job queue. One worker drains a FIFO so two GPU jobs never run at once
(the orka crash lesson). Each job gets a progress bus (asyncio.Queue) that SSE drains."""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable

Runner = Callable[..., Awaitable[Any]]


class _Job:
    def __init__(self, runner: Runner, kwargs: dict):
        self.id = uuid.uuid4().hex[:12]
        self.runner = runner
        self.kwargs = kwargs
        self.status = "queued"
        self.result: Any = None
        self.error: str | None = None
        self.events: asyncio.Queue = asyncio.Queue()
        self.done = asyncio.Event()


class JobQueue:
    def __init__(self) -> None:
        self._jobs: dict[str, _Job] = {}
        self._fifo: asyncio.Queue = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            self._worker = None

    def submit(self, runner: Runner, **kwargs) -> str:
        job = _Job(runner, kwargs)
        self._jobs[job.id] = job
        self._fifo.put_nowait(job.id)
        return job.id

    def status(self, job_id: str) -> str:
        j = self._jobs.get(job_id)
        return j.status if j else "unknown"

    def job(self, job_id: str) -> _Job | None:
        return self._jobs.get(job_id)

    async def wait(self, job_id: str) -> Any:
        j = self._jobs[job_id]
        await j.done.wait()
        if j.error:
            raise RuntimeError(j.error)
        return j.result

    async def _run(self) -> None:
        while True:
            job_id = await self._fifo.get()
            job = self._jobs[job_id]
            job.status = "running"

            def emit(ev: dict, _q=job.events) -> None:
                _q.put_nowait(ev)

            try:
                job.result = await job.runner(job.id, emit, **job.kwargs)
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - surface to caller, never crash worker
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = "failed"
            finally:
                job.events.put_nowait({"stage": "_end"})
                job.done.set()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_jobs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/backend/jobs.py ui/backend/tests/test_jobs.py
git commit -m "feat(ui): single-GPU serial job queue + progress bus"
```

---

### Task 10: Live runner (pack + eval -> measured journey)

**Files:**
- Create: `ui/backend/live.py`
- Test: `ui/backend/tests/test_live.py`

- [ ] **Step 1: Write the failing test** (mocks orka pack/eval + journey; no GPU)

```python
# ui/backend/tests/test_live.py
import asyncio

import pytest

from ui.backend import live
from ui.backend.schema import Journey


def _static_journey():
    from ui.backend.arch import build_architecture
    from ui.backend.estimator import estimate
    from ui.backend.pipeline_steps import build_pipeline, build_tricks
    from ui.backend.schema import Journey, ModelMeta
    cfg = {"vocab_size": 32, "tie_word_embeddings": False, "torch_dtype": "bfloat16"}
    shapes = {"lm_head.weight": (32, 8), "model.layers.0.mlp.down_proj.weight": (64, 64)}
    meta = ModelMeta(name="x/y", params_total=4352, dtype="bfloat16", vocab_size=32,
                     tie_word_embeddings=False, fp16_bytes=8704)
    arch = build_architecture(cfg, shapes)
    return Journey(schema_version=1, model=meta, architecture=arch,
                   pipeline=build_pipeline(arch), tricks=build_tricks(arch),
                   result=estimate(meta, arch))


@pytest.mark.asyncio
async def test_run_live_emits_measured(monkeypatch):
    monkeypatch.setattr(live, "build_static_journey", lambda m, **k: _static_journey())
    # fake the GPU pipeline
    monkeypatch.setattr(live, "_pack_and_eval",
                        lambda model, emit: {"ratio": 4.3, "fp16_mb": 988.0, "orka_mb": 230.0,
                                             "ppl_base": 20.9, "ppl_orka": 28.3,
                                             "ppl_ratio": 1.354, "trusted": True,
                                             "trust_reason": None})
    events = []
    j = await live.run_live("x/y", "jobid", lambda e: events.append(e))
    assert isinstance(j, Journey)
    assert j.result.source == "measured"
    assert j.result.ratio == 4.3
    assert j.result.trusted is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_live.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write live.py**

```python
# ui/backend/live.py
"""Live GPU path: pack the model with the validated config + eval, return a MEASURED
journey (same schema as static). The heavy GPU work runs in a thread (asyncio.to_thread) so
the event loop stays responsive; progress is emitted via the job's emit callback.

trusted/trust_reason ride on the reliable-eval hardening; until that lands eval may report
trusted=None and the UI shows 'unverified' rather than a false number."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from .journey import build_static_journey
from .schema import Journey, Result
from .settings import GPU_MEM_CAP_GB


def _pack_and_eval(model: str, emit) -> dict:
    """Blocking GPU work. Imports orka lazily so the module loads without torch present."""
    from huggingface_hub import snapshot_download
    from orka.eval import eval_artifact
    from orka.pipeline.pack import pack_checkpoint

    emit({"stage": "download", "msg": f"resolving {model}"})
    snap = Path(snapshot_download(model))

    with tempfile.TemporaryDirectory() as tmp:
        art = Path(tmp) / "art"
        prompts = Path(tmp) / "prompts.txt"
        prompts.write_text("The capital of France is Paris.\nWater boils at 100 C.\n")
        emit({"stage": "pack", "msg": "rvq-12-12 + em-aq + hessian + auto keep-head"})
        pack_checkpoint(
            snap, out_dir=art, codebook_sizes=[4096, 4096], em_aq_passes=3,
            keep_head_fp16="auto", awq_model_dir=snap, awq_calibration=prompts,
            max_gpu_mem_gb=GPU_MEM_CAP_GB, backend="torch", device="cuda",
        )
        emit({"stage": "eval", "msg": "reconstruct + perplexity"})
        out = Path(tmp) / "eval.json"
        res = eval_artifact(art, prompts, out, device="cuda")
        fp16_mb = sum(f.stat().st_size for f in snap.glob("*.safetensors")) / 1e6
        orka_mb = sum(f.stat().st_size for f in art.rglob("*") if f.is_file()) / 1e6
        return {
            "ratio": round(fp16_mb / max(orka_mb, 1e-9), 2), "fp16_mb": round(fp16_mb, 1),
            "orka_mb": round(orka_mb, 1),
            "ppl_base": res.get("original_perplexity"), "ppl_orka": res.get("orka_perplexity"),
            "ppl_ratio": (round(res["perplexity_ratio"], 3)
                          if res.get("perplexity_ratio") not in (None, float("inf")) else None),
            "trusted": res.get("trusted"), "trust_reason": res.get("trust_reason"),
        }


async def run_live(model: str, job_id: str, emit) -> Journey:
    base = build_static_journey(model)          # arch + pipeline + tricks (no GPU)
    measured = await asyncio.to_thread(_pack_and_eval, model, emit)
    base.result = Result(source="measured", bpw=3.0, notes=["measured: rvq-12-12+em-aq+hessian"],
                         **measured)
    return base
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_live.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/backend/live.py ui/backend/tests/test_live.py
git commit -m "feat(ui): live pack+eval runner (measured journey)"
```

---

### Task 11: /pack + /jobs + SSE wiring

**Files:**
- Modify: `ui/backend/app.py` (add queue lifespan + 3 routes)
- Test: `ui/backend/tests/test_app_pack.py`

- [ ] **Step 1: Write the failing test**

```python
# ui/backend/tests/test_app_pack.py
from fastapi.testclient import TestClient

from ui.backend import app as appmod


def test_pack_enqueues_and_completes(monkeypatch):
    async def fake_run_live(model, job_id, emit):
        emit({"stage": "pack", "msg": "x"})
        from ui.backend.tests.test_live import _static_journey
        j = _static_journey()
        j.result.source = "measured"
        return j
    monkeypatch.setattr(appmod, "run_live", fake_run_live)

    with TestClient(appmod.app) as c:               # triggers startup (queue.start)
        r = c.post("/pack", json={"model": "x/y"})
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        # poll until done
        for _ in range(200):
            s = c.get(f"/jobs/{job_id}").json()
            if s["status"] in ("done", "failed"):
                break
        assert s["status"] == "done"
        assert s["journey"]["result"]["source"] == "measured"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_app_pack.py -v`
Expected: FAIL (no `/pack` route / no `run_live` symbol on app)

- [ ] **Step 3: Modify app.py** — add the import, queue lifespan, and three routes. Insert after the existing imports/`app` setup:

```python
# add to imports near top of ui/backend/app.py
import asyncio
import json
from contextlib import asynccontextmanager

from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .jobs import JobQueue
from .live import run_live
from .settings import LIVE_PARAM_CEILING

# replace the bare `app = FastAPI(...)` line with a lifespan-managed queue:
_queue = JobQueue()


@asynccontextmanager
async def _lifespan(_app):
    await _queue.start()
    yield
    await _queue.stop()


app = FastAPI(title="orka compression-journey analysis engine", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PackRequest(BaseModel):
    model: str


@app.post("/pack")
def pack(req: PackRequest):
    async def runner(job_id, emit, *, model):
        return await run_live(model, job_id, emit)
    job_id = _queue.submit(runner, model=req.model)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = _queue.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    body = {"status": job.status, "error": job.error}
    if job.status == "done":
        body["journey"] = job.result.model_dump()
    return body


@app.get("/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    job = _queue.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")

    async def gen():
        while True:
            ev = await job.events.get()
            if ev.get("stage") == "_end":
                final = (job.result.model_dump() if job.status == "done"
                         else {"error": job.error})
                yield {"event": "result", "data": json.dumps(final)}
                break
            yield {"event": "progress", "data": json.dumps(ev)}

    return EventSourceResponse(gen())
```

NOTE: delete the original `app = FastAPI(...)` + `add_middleware` lines from Task 8 (now replaced by the lifespan-managed version above) - keep the `/analyze` route definition.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/test_app_pack.py ui/backend/tests/test_app_analyze.py -v`
Expected: PASS (both - /analyze unaffected)

- [ ] **Step 5: Commit**

```bash
git add ui/backend/app.py ui/backend/tests/test_app_pack.py
git commit -m "feat(ui): /pack + /jobs + SSE progress wiring"
```

---

### Task 12: README + run script + full-suite gate

**Files:**
- Create: `ui/backend/README.md`
- Create: `ui/backend/run.sh`
- Test: (none new) - run the whole backend suite

- [ ] **Step 1: Write README.md**

```markdown
# orka compression-journey analysis engine (layer-1)

Backend for the visualizer. Turns an HF model name into one journey JSON.

## Run
    PYTHONPATH=<repo-root> HF_HOME=~/ai-models/hf-cache \
      <repo>/.venv/bin/python -m uvicorn ui.backend.app:app --reload --port 8723

## Endpoints
- `GET  /analyze?model=<id>&bpw=3.0&keep_head=true&lattice=false` - instant static (estimated)
- `POST /pack {"model": "<id>"}` -> `{job_id}` - queued GPU job (measured)
- `GET  /jobs/{id}` - status (+ journey when done)
- `GET  /jobs/{id}/stream` - SSE progress, then the measured journey

## Notes
- Static = config + safetensors header only (no weights). Numbers labeled `estimated`.
- Live = single GPU job at a time, 10GB cap, validated pack config. Numbers `measured`.
- `trusted` rides on the reliable-eval hardening; may be null until that lands.
```

- [ ] **Step 2: Write run.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHONPATH="$ROOT" HF_HOME="${HF_HOME:-$HOME/ai-models/hf-cache}" \
  "$ROOT/.venv/bin/python" -m uvicorn ui.backend.app:app --port "${PORT:-8723}" "$@"
```

- [ ] **Step 3: Run the full backend suite**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest ui/backend/tests/ -v`
Expected: PASS (all tasks' tests green)

- [ ] **Step 4: Smoke the static path against a real model** (network, no GPU)

Run: `PYTHONPATH=$PWD HF_HOME=~/ai-models/hf-cache .venv/bin/python -c "from ui.backend.journey import build_static_journey; j=build_static_journey('HuggingFaceTB/SmolLM-135M'); print(j.model.params_total, j.architecture.arch_class, j.result.ratio, j.result.ppl_ratio)"`
Expected: prints params, `dense`, a ratio > 1, a ppl_ratio (estimated). If it errors, debug fetch before proceeding.

- [ ] **Step 5: Commit**

```bash
chmod +x ui/backend/run.sh
git add ui/backend/README.md ui/backend/run.sh
git commit -m "docs(ui): backend README + run script"
```

---

## Self-Review

**Spec coverage:**
- `/analyze` static ✓ (Tasks 3-8). `/pack` + queue + SSE ✓ (Tasks 9-11). Journey contract ✓ (Task 2). Static estimator ✓ (Task 5). Safetensors-header fetch ✓ (Task 3). ArchProfile-gated treatment + flags ✓ (Task 4). Trick catalogue arch-gated ✓ (Task 6). Live pack+eval with `trusted` flag ✓ (Task 10). Single-GPU serial guard ✓ (Task 9). Error handling 404/403 ✓ (Task 8); too-big/`partial`/untrusted paths are present in code but only partially tested - acceptable for v1 (flagged below).
- **Gap accepted for v1:** the live "param ceiling" rejection (`LIVE_PARAM_CEILING`) and `.bin` `partial:true` path are defined in settings/spec but not given dedicated tasks/tests; add a guard in `/pack` and a test if pursued. Noted, not blocking the slice.

**Placeholder scan:** no TBD/TODO; every code step is complete and runnable; commands have expected output.

**Type consistency:** `Journey`/`Result`/`Architecture` field names match across `schema.py`, `estimator.py`, `journey.py`, `live.py`, `app.py`. `JobQueue` methods (`submit`/`wait`/`status`/`job`/`start`/`stop`) match between `jobs.py` and `app.py`/tests. `run_live(model, job_id, emit)` signature matches between `live.py` and `app.py`. `_pack_and_eval(model, emit)` matches between definition and the `live` test mock. `estimate(meta, arch, bpw, keep_head, lattice)` matches across estimator test, journey, and live.

**Verified:** `eval_artifact`'s summary keys are `original_perplexity` / `orka_perplexity` / `perplexity_ratio` (confirmed in `orka/eval/__init__.py:_summarize_eval_rows`); `_pack_and_eval` uses them directly. There is **no `trusted` key yet** - that is the reliable-eval hardening dependency; `res.get("trusted")` returns None until it lands, and the UI shows "unverified" (by design, never a false number).

## Execution Handoff

Plan complete and saved to `ui/docs/plans/2026-06-30-layer1-analysis-engine.md`.
