# Autoquant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `orka autoquant <model>` — an arch-agnostic controller that derives a per-tensor quant config automatically (deterministic policy core + LLM for hard calls), orchestrating existing orka tools.

**Architecture:** Standalone `orka/autoquant/` package. Pure modules (priors, schema, roles, probes, policy) are functions of their inputs; I/O modules (harness, escalation, orchestrator) drive the LLM and shell out to existing `orka` subcommands (pack, pulse-check). The chain: introspect → probe → policy → escalate → pack → pulse-check → refine.

**Tech Stack:** Python 3.11, pytest, litellm (LLM transport), existing orka (`classify_tensor_family`, `build_allocation`, `pack`, `pulse-check`).

---

## File structure

| File | Responsibility |
|---|---|
| `orka/autoquant/__init__.py` | package exports |
| `orka/autoquant/priors.py` | seeded role→rule knowledge (lm_head=int8, etc.) |
| `orka/autoquant/schema.py` | `TensorConfig` dataclass + allocation_map (de)serialize |
| `orka/autoquant/roles.py` | arch-agnostic `classify_role(name, shape, ctx)` |
| `orka/autoquant/probes.py` | `Signals` per tensor (SQNR, sensitivity, RD point) |
| `orka/autoquant/policy.py` | `decide(role, signals) -> (TensorConfig, confidence)` |
| `orka/autoquant/escalation.py` | 4 triggers + global signature cache |
| `orka/autoquant/harness.py` | pi-reason generate→verify→refine LLM loop |
| `orka/autoquant/orchestrator.py` | the chain + 3 objective stopping rules |
| `orka/cli/commands.py` (modify) | `cmd_autoquant` |
| `orka/cli/parser.py` (modify) | `autoquant` subparser |
| `tests/autoquant/test_*.py` | one test module per source module |

Build order = dependency order: priors → schema → roles → probes → policy → escalation → harness → orchestrator → CLI → integration.

---

### Task 1: priors.py — seeded knowledge table

**Files:**
- Create: `orka/autoquant/__init__.py` (empty)
- Create: `orka/autoquant/priors.py`
- Test: `tests/autoquant/test_priors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/autoquant/test_priors.py
from orka.autoquant.priors import ROLE_PRIORS, SQNR_TARGET_DB

def test_output_head_is_int8_never_rvq():
    p = ROLE_PRIORS["out-head"]
    assert p["method"] == "int8"
    assert p["allow_rvq"] is False

def test_norm_and_bias_kept_fp16():
    assert ROLE_PRIORS["norm"]["method"] == "fp16"
    assert ROLE_PRIORS["bias"]["method"] == "fp16"

def test_in_embed_allows_rvq():
    assert ROLE_PRIORS["in-embed"]["allow_rvq"] is True

def test_sqnr_target():
    assert SQNR_TARGET_DB == 30.0
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/autoquant/test_priors.py -q`
Expected: FAIL — `ModuleNotFoundError: orka.autoquant.priors`

- [ ] **Step 3: Write minimal implementation**

```python
# orka/autoquant/priors.py
"""Seeded role->rule priors for autoquant. Hard-won defaults from packing experiments:
the output head is catastrophic under RVQ (ppl 1.2M) but lossless as int8; norms/biases
must stay fp16; input embeddings tolerate RVQ. Target per-linear SQNR ~30 dB (14 dB was
catastrophic at model scale)."""
from __future__ import annotations

SQNR_TARGET_DB: float = 30.0

# method: default quant method for the role. allow_rvq: may RVQ ever be used here.
# confidence: how sure the policy is (1.0 = never escalate this role).
ROLE_PRIORS: dict[str, dict] = {
    "out-head":  {"method": "int8", "allow_rvq": False, "confidence": 1.0},
    "in-embed":  {"method": "rvq",  "allow_rvq": True,  "confidence": 0.8},
    "norm":      {"method": "fp16", "allow_rvq": False, "confidence": 1.0},
    "bias":      {"method": "fp16", "allow_rvq": False, "confidence": 1.0},
    "attn.q":    {"method": "rvq",  "allow_rvq": True,  "confidence": 0.7},
    "attn.k":    {"method": "rvq",  "allow_rvq": True,  "confidence": 0.7},
    "attn.v":    {"method": "rvq",  "allow_rvq": True,  "confidence": 0.6, "extra_stage": True},
    "attn.o":    {"method": "rvq",  "allow_rvq": True,  "confidence": 0.7},
    "mlp.up":    {"method": "rvq",  "allow_rvq": True,  "confidence": 0.7},
    "mlp.gate":  {"method": "rvq",  "allow_rvq": True,  "confidence": 0.7},
    "mlp.down":  {"method": "rvq",  "allow_rvq": True,  "confidence": 0.6, "extra_stage": True},
    "unknown":   {"method": "fp16", "allow_rvq": False, "confidence": 0.0},  # safe default
}
```

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest tests/autoquant/test_priors.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add orka/autoquant/__init__.py orka/autoquant/priors.py tests/autoquant/test_priors.py
git commit -m "feat(autoquant): seeded role priors (lm_head=int8-never-rvq, norm/bias=fp16)"
```

---

### Task 2: schema.py — TensorConfig + allocation_map

**Files:**
- Create: `orka/autoquant/schema.py`
- Test: `tests/autoquant/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/autoquant/test_schema.py
from orka.autoquant.schema import TensorConfig, to_allocation_map, from_allocation_map

def test_roundtrip():
    cfgs = {
        "lm_head.weight": TensorConfig(method="int8", bits=8, stages=0,
                                       normalization="block-max", keep_fp16=False,
                                       source="policy", confidence=1.0, rationale="head"),
        "blk.0.mlp.down.weight": TensorConfig(method="rvq", bits=3, stages=2,
                                       normalization="block-max", keep_fp16=False,
                                       source="llm", confidence=0.6, rationale="sensitive"),
    }
    m = to_allocation_map(cfgs)
    assert m["lm_head.weight"]["method"] == "int8"
    assert m["blk.0.mlp.down.weight"]["stages"] == 2
    back = from_allocation_map(m)
    assert back["lm_head.weight"] == cfgs["lm_head.weight"]

def test_fp16_tensor_serializes_keep_fp16():
    c = TensorConfig(method="fp16", bits=16, stages=0, normalization="none",
                     keep_fp16=True, source="policy", confidence=1.0, rationale="norm")
    assert to_allocation_map({"x": c})["x"]["keep_fp16"] is True
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/autoquant/test_schema.py -q`
Expected: FAIL — `ModuleNotFoundError: orka.autoquant.schema`

- [ ] **Step 3: Write minimal implementation**

```python
# orka/autoquant/schema.py
"""Data contract for autoquant decisions. TensorConfig is one tensor's decision;
to/from_allocation_map (de)serialize the per-tensor map consumed by `orka pack`."""
from __future__ import annotations
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class TensorConfig:
    method: str            # "rvq" | "int8" | "fp16"
    bits: int
    stages: int            # rvq stages (0 for int8/fp16)
    normalization: str     # "block-max" | "none" | ...
    keep_fp16: bool
    source: str            # "policy" | "llm" | "cache"
    confidence: float
    rationale: str

def to_allocation_map(cfgs: dict[str, TensorConfig]) -> dict[str, dict]:
    return {name: asdict(c) for name, c in cfgs.items()}

def from_allocation_map(m: dict[str, dict]) -> dict[str, TensorConfig]:
    return {name: TensorConfig(**d) for name, d in m.items()}
```

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest tests/autoquant/test_schema.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add orka/autoquant/schema.py tests/autoquant/test_schema.py
git commit -m "feat(autoquant): TensorConfig + allocation_map serialization"
```

---

### Task 3: roles.py — arch-agnostic role classifier

**Files:**
- Create: `orka/autoquant/roles.py`
- Test: `tests/autoquant/test_roles.py`

**Context:** wraps existing `orka.quant.classify_tensor_family(name)->str` (returns
embedding/attention/mlp/other/...) and refines it: the family classifier lumps `lm_head`
and `embed_out` into "embedding" — we must split **out-head** from **in-embed** (the
session bug), and resolve sub-roles (q/k/v/o, up/down/gate, norm, bias). `tied` flags a
shared embedding/head.

- [ ] **Step 1: Write the failing test**

```python
# tests/autoquant/test_roles.py
from orka.autoquant.roles import classify_role

def test_output_head_split_from_embedding():
    assert classify_role("embed_out.weight", (50304, 768))[0] == "out-head"
    assert classify_role("lm_head.weight", (50304, 768))[0] == "out-head"
    assert classify_role("gpt_neox.embed_in.weight", (50304, 768))[0] == "in-embed"

def test_attention_subroles():
    assert classify_role("model.layers.0.self_attn.v_proj.weight", (768, 768))[0] == "attn.v"
    assert classify_role("model.layers.0.self_attn.o_proj.weight", (768, 768))[0] == "attn.o"

def test_mlp_subroles():
    assert classify_role("model.layers.0.mlp.down_proj.weight", (768, 3072))[0] == "mlp.down"
    assert classify_role("model.layers.0.mlp.gate_proj.weight", (3072, 768))[0] == "mlp.gate"

def test_norm_and_bias():
    assert classify_role("model.layers.0.input_layernorm.weight", (768,))[0] == "norm"
    assert classify_role("model.layers.0.mlp.down_proj.bias", (768,))[0] == "bias"

def test_unknown_low_confidence():
    role, conf = classify_role("mystery.tensor.foo", (123, 456))
    assert role == "unknown"
    assert conf < 0.5
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/autoquant/test_roles.py -q`
Expected: FAIL — `ModuleNotFoundError: orka.autoquant.roles`

- [ ] **Step 3: Write minimal implementation**

```python
# orka/autoquant/roles.py
"""Arch-agnostic tensor role classifier. Refines classify_tensor_family by splitting the
output head from the input embedding (they need opposite treatment) and resolving sub-roles.
Returns (role, confidence). Unknown names fall to ('unknown', low) -> escalation."""
from __future__ import annotations
from orka.quant import classify_tensor_family

_OUT_HEAD = ("lm_head", "embed_out", "output.weight")
_IN_EMBED = ("embed_in", "wte", "embed_tokens", "word_embeddings", "embedding")

def classify_role(name: str, shape: tuple[int, ...], tied: bool = False) -> tuple[str, float]:
    n = name.lower()
    if n.endswith(".bias") or n.endswith("_bias"):
        return "bias", 1.0
    if "norm" in n or "ln_" in n or n.endswith(".ln.weight"):
        return "norm", 1.0

    fam = classify_tensor_family(name)
    if fam == "embedding":
        if any(m in n for m in _OUT_HEAD):
            return "out-head", 1.0
        if any(m in n for m in _IN_EMBED):
            return "in-embed", 1.0
        # 2D [vocab, hidden] with no clear marker: ambiguous edge tensor
        return ("unknown", 0.3)
    if fam == "attention":
        for k in ("q_proj", "query"):
            if k in n: return "attn.q", 0.9
        for k in ("k_proj", "key"):
            if k in n: return "attn.k", 0.9
        for k in ("v_proj", "value"):
            if k in n: return "attn.v", 0.9
        for k in ("o_proj", "out_proj", "c_proj", ".dense"):
            if k in n: return "attn.o", 0.9
        return "attn.o", 0.5  # fused qkv / unknown attn linear -> conservative
    if fam == "mlp":
        if "down" in n or "fc2" in n or "fc_out" in n or ".wo" in n or ".w2" in n:
            return "mlp.down", 0.9
        if "gate" in n or ".w1" in n:
            return "mlp.gate", 0.9
        if "up" in n or "fc1" in n or "fc_in" in n or ".wi" in n or "c_fc" in n or ".w3" in n:
            return "mlp.up", 0.9
        return "mlp.up", 0.5
    return "unknown", 0.3
```

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest tests/autoquant/test_roles.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add orka/autoquant/roles.py tests/autoquant/test_roles.py
git commit -m "feat(autoquant): arch-agnostic role classifier (splits out-head/in-embed)"
```

---

### Task 4: probes.py — per-tensor signals

**Files:**
- Create: `orka/autoquant/probes.py`
- Test: `tests/autoquant/test_probes.py`

**Context:** v1 cheap signals computable from a single weight tensor: `sqnr_at` (SQNR at a
candidate bpw via a quick RVQ probe) and `rd_knee` (bpw where SQNR crosses the target). Use
numpy only — no full pack. `Signals` is a plain dataclass consumed by policy.

- [ ] **Step 1: Write the failing test**

```python
# tests/autoquant/test_probes.py
import numpy as np
from orka.autoquant.probes import probe_tensor, Signals

def test_signals_shape_and_monotonic_sqnr():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((256, 512)).astype("float32")
    s = probe_tensor(W)
    assert isinstance(s, Signals)
    # more bits -> not-worse SQNR
    assert s.sqnr_at(8) >= s.sqnr_at(2) - 1e-6

def test_rd_knee_reaches_target():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((256, 512)).astype("float32")
    s = probe_tensor(W, sqnr_target_db=30.0)
    assert 1 <= s.rd_knee_bits <= 16
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/autoquant/test_probes.py -q`
Expected: FAIL — `ModuleNotFoundError: orka.autoquant.probes`

- [ ] **Step 3: Write minimal implementation**

```python
# orka/autoquant/probes.py
"""Cheap per-tensor signals for autoquant, computed from one weight tensor (no full pack).
sqnr_at(bits): SQNR of a fast scalar-quant probe at `bits`. rd_knee_bits: smallest bits
hitting the SQNR target. These rank tensors and drive the policy's bit choice."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

def _sqnr_db(W: np.ndarray, bits: int) -> float:
    # symmetric per-row scalar quant probe (cheap proxy for achievable distortion)
    qmax = (1 << (bits - 1)) - 1
    s = np.abs(W).max(axis=1, keepdims=True) / max(qmax, 1)
    s[s == 0] = 1.0
    Wq = np.clip(np.round(W / s), -qmax, qmax) * s
    sig = float((W ** 2).sum())
    err = float(((W - Wq) ** 2).sum()) or 1e-12
    return 10.0 * np.log10(sig / err)

@dataclass
class Signals:
    sqnr_curve: dict[int, float]
    rd_knee_bits: int
    sensitivity: float            # placeholder = weight energy proxy (refined later)
    def sqnr_at(self, bits: int) -> float:
        return self.sqnr_curve.get(bits, _nearest(self.sqnr_curve, bits))

def _nearest(curve: dict[int, float], bits: int) -> float:
    k = min(curve, key=lambda b: abs(b - bits))
    return curve[k]

def probe_tensor(W: np.ndarray, sqnr_target_db: float = 30.0) -> Signals:
    W = np.asarray(W, dtype=np.float32)
    if W.ndim == 1:
        W = W.reshape(1, -1)
    curve = {b: _sqnr_db(W, b) for b in (2, 3, 4, 6, 8)}
    knee = next((b for b in (2, 3, 4, 6, 8) if curve[b] >= sqnr_target_db), 16)
    return Signals(sqnr_curve=curve, rd_knee_bits=knee, sensitivity=float(np.abs(W).mean()))
```

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest tests/autoquant/test_probes.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add orka/autoquant/probes.py tests/autoquant/test_probes.py
git commit -m "feat(autoquant): cheap per-tensor signals (sqnr curve + rd knee)"
```

---

### Task 5: policy.py — deterministic core

**Files:**
- Create: `orka/autoquant/policy.py`
- Test: `tests/autoquant/test_policy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/autoquant/test_policy.py
from orka.autoquant.policy import decide
from orka.autoquant.probes import Signals

def _sig(knee=3): return Signals(sqnr_curve={3:31.0,8:45.0}, rd_knee_bits=knee, sensitivity=0.02)

def test_out_head_is_int8_high_confidence():
    cfg, conf = decide("out-head", _sig())
    assert cfg.method == "int8" and conf == 1.0

def test_norm_kept_fp16():
    cfg, conf = decide("norm", _sig())
    assert cfg.keep_fp16 and cfg.method == "fp16"

def test_default_uses_rd_knee_bits():
    cfg, _ = decide("attn.q", _sig(knee=4))
    assert cfg.method == "rvq" and cfg.bits == 4

def test_sensitive_role_gets_extra_stage():
    cfg, _ = decide("mlp.down", _sig())
    base, _ = decide("attn.q", _sig())
    assert cfg.stages >= base.stages + 1

def test_unknown_is_low_confidence_fp16():
    cfg, conf = decide("unknown", _sig())
    assert cfg.keep_fp16 and conf < 0.5
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/autoquant/test_policy.py -q`
Expected: FAIL — `ModuleNotFoundError: orka.autoquant.policy`

- [ ] **Step 3: Write minimal implementation**

```python
# orka/autoquant/policy.py
"""Deterministic policy core: (role, signals) -> (TensorConfig, confidence). Table-driven
from priors. RVQ roles take their bit count from the rate-distortion knee; sensitive roles
(attn.v, mlp.down) get one extra stage; head/norm/bias follow their fixed priors."""
from __future__ import annotations
from orka.autoquant.priors import ROLE_PRIORS
from orka.autoquant.probes import Signals
from orka.autoquant.schema import TensorConfig

def decide(role: str, signals: Signals) -> tuple[TensorConfig, float]:
    p = ROLE_PRIORS.get(role, ROLE_PRIORS["unknown"])
    conf = float(p["confidence"])
    if p["method"] == "fp16":
        return TensorConfig("fp16", 16, 0, "none", True, "policy", conf,
                            f"{role}: fp16 prior"), conf
    if p["method"] == "int8":
        return TensorConfig("int8", 8, 0, "block-max", False, "policy", conf,
                            f"{role}: int8 prior (RVQ-fragile)"), conf
    bits = int(signals.rd_knee_bits)
    stages = 2 + (1 if p.get("extra_stage") else 0)
    return TensorConfig("rvq", bits, stages, "block-max", False, "policy", conf,
                        f"{role}: rvq {bits}b/{stages}st at SQNR knee"), conf
```

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest tests/autoquant/test_policy.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add orka/autoquant/policy.py tests/autoquant/test_policy.py
git commit -m "feat(autoquant): deterministic policy core (table-driven from priors)"
```

---

### Task 6: escalation.py — triggers + global signature cache

**Files:**
- Create: `orka/autoquant/escalation.py`
- Test: `tests/autoquant/test_escalation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/autoquant/test_escalation.py
from orka.autoquant.escalation import should_escalate, signature, Cache
from orka.autoquant.probes import Signals

def _sig(knee=3): return Signals(sqnr_curve={3:31.0}, rd_knee_bits=knee, sensitivity=0.02)

def test_low_confidence_triggers():
    assert should_escalate(role_conf=0.3, signals=_sig(), policy_conf=0.3, regressed=False)

def test_high_confidence_no_trigger():
    assert not should_escalate(role_conf=1.0, signals=_sig(), policy_conf=1.0, regressed=False)

def test_regression_triggers():
    assert should_escalate(role_conf=1.0, signals=_sig(), policy_conf=1.0, regressed=True)

def test_signature_stable_and_bucketed():
    a = signature("attn.q", (768, 768), _sig(3), "min-bits")
    b = signature("attn.q", (768, 768), _sig(3), "min-bits")
    assert a == b

def test_cache_roundtrip(tmp_path):
    c = Cache(tmp_path / "cache.json")
    c.put("sig1", {"method": "rvq", "bits": 3})
    assert Cache(tmp_path / "cache.json").get("sig1")["bits"] == 3
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/autoquant/test_escalation.py -q`
Expected: FAIL — `ModuleNotFoundError: orka.autoquant.escalation`

- [ ] **Step 3: Write minimal implementation**

```python
# orka/autoquant/escalation.py
"""Escalation triggers + global signature cache. A tensor goes to the LLM when any trigger
fires: low policy/role confidence, probe conflict (RD says cheap but SQNR fragile), a
post-pack regression, or unknown role. Verdicts are cached by signature (role + shape
bucket + sensitivity bucket + objective) in a global JSON, so they are reproducible and
reused across models."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
from orka.autoquant.probes import Signals

CONF_THRESHOLD = 0.6

def _conflict(s: Signals) -> bool:
    # cheap knee but a stage where SQNR is still poor -> conflicting signal
    return s.rd_knee_bits <= 3 and min(s.sqnr_curve.values()) < 10.0

def should_escalate(role_conf: float, signals: Signals, policy_conf: float, regressed: bool) -> bool:
    return (role_conf < CONF_THRESHOLD or policy_conf < CONF_THRESHOLD
            or _conflict(signals) or regressed)

def _bucket(x: float, edges=(0.005, 0.02, 0.05, 0.1)) -> int:
    return sum(1 for e in edges if x > e)

def signature(role: str, shape: tuple[int, ...], s: Signals, objective: str) -> str:
    key = f"{role}|{tuple(shape)}|knee{s.rd_knee_bits}|sens{_bucket(s.sensitivity)}|{objective}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]

class Cache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
    def get(self, sig: str):
        return self.data.get(sig)
    def put(self, sig: str, verdict: dict) -> None:
        self.data[sig] = verdict
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

def default_cache_path() -> Path:
    return Path.home() / ".orka" / "autoquant-cache.json"
```

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest tests/autoquant/test_escalation.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add orka/autoquant/escalation.py tests/autoquant/test_escalation.py
git commit -m "feat(autoquant): escalation triggers + global signature cache"
```

---

### Task 7: harness.py — pi-reason LLM loop

**Files:**
- Create: `orka/autoquant/harness.py`
- Test: `tests/autoquant/test_harness.py`

**Context:** pi-reason pattern: generate→verify→refine. The transport is injected (an
`llm_fn(messages)->str` callable) so tests use a canned function and production passes a
litellm-backed one. Output is schema-validated; invalid → raise, caller falls back to policy.

- [ ] **Step 1: Write the failing test**

```python
# tests/autoquant/test_harness.py
import json
from orka.autoquant.harness import decide_with_llm
from orka.autoquant.probes import Signals

def _sig(): return Signals(sqnr_curve={3:31.0,8:45.0}, rd_knee_bits=3, sensitivity=0.02)

def test_uses_llm_verdict_when_valid():
    def fake_llm(messages):
        return json.dumps({"method": "rvq", "bits": 4, "stages": 3,
                           "normalization": "block-max", "keep_fp16": False,
                           "rationale": "needs headroom"})
    cfg = decide_with_llm("mlp.down", (768, 3072), _sig(), "min-bits", llm_fn=fake_llm)
    assert cfg.method == "rvq" and cfg.bits == 4 and cfg.source == "llm"

def test_invalid_llm_output_raises():
    def bad_llm(messages): return "not json"
    import pytest
    with pytest.raises(ValueError):
        decide_with_llm("mlp.down", (768, 3072), _sig(), "min-bits", llm_fn=bad_llm)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/autoquant/test_harness.py -q`
Expected: FAIL — `ModuleNotFoundError: orka.autoquant.harness`

- [ ] **Step 3: Write minimal implementation**

```python
# orka/autoquant/harness.py
"""pi-reason generate->verify->refine LLM loop for hard-call tensors. Transport is injected
as llm_fn(messages)->str (production = litellm; tests = canned). The model receives the
role, shape, signals and objective and returns a JSON quant config, schema-validated here."""
from __future__ import annotations
import json
from orka.autoquant.probes import Signals
from orka.autoquant.schema import TensorConfig

_VALID_METHODS = {"rvq", "int8", "fp16"}

def _prompt(role, shape, s: Signals, objective):
    return [
        {"role": "system", "content":
         "You are a quantization expert. Given one tensor's role, shape, and distortion "
         "signals, output ONLY a JSON object: {method:rvq|int8|fp16, bits:int, stages:int, "
         "normalization:str, keep_fp16:bool, rationale:str}. The output head must never be "
         "RVQ (use int8). Norms/biases stay fp16."},
        {"role": "user", "content": json.dumps({
            "role": role, "shape": list(shape), "objective": objective,
            "sqnr_curve": s.sqnr_curve, "rd_knee_bits": s.rd_knee_bits,
            "sensitivity": s.sensitivity})},
    ]

def _parse(text: str) -> dict:
    try:
        d = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"LLM did not return JSON: {text[:80]!r}") from e
    if d.get("method") not in _VALID_METHODS:
        raise ValueError(f"invalid method {d.get('method')!r}")
    return d

def decide_with_llm(role, shape, signals: Signals, objective: str, *, llm_fn) -> TensorConfig:
    d = _parse(llm_fn(_prompt(role, shape, signals, objective)))
    return TensorConfig(
        method=d["method"], bits=int(d.get("bits", 8)), stages=int(d.get("stages", 0)),
        normalization=d.get("normalization", "block-max"),
        keep_fp16=bool(d.get("keep_fp16", False)), source="llm",
        confidence=0.75, rationale=d.get("rationale", "llm"))
```

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest tests/autoquant/test_harness.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add orka/autoquant/harness.py tests/autoquant/test_harness.py
git commit -m "feat(autoquant): pi-reason LLM loop (injected transport, schema-validated)"
```

---

### Task 8: orchestrator.py — the chain

**Files:**
- Create: `orka/autoquant/orchestrator.py`
- Test: `tests/autoquant/test_orchestrator.py`

**Context:** ties roles→probes→policy→escalation→harness into a per-tensor config map.
v1 `derive_config` takes a dict `{name: (shape, weight_ndarray)}` and returns
`{name: TensorConfig}`. The pack/pulse-check refine loop is exercised in the integration
task (Task 10); this unit test covers the decision chain with the LLM disabled.

- [ ] **Step 1: Write the failing test**

```python
# tests/autoquant/test_orchestrator.py
import numpy as np
from orka.autoquant.orchestrator import derive_config

def test_derives_int8_head_and_rvq_linears_no_llm():
    rng = np.random.default_rng(0)
    tensors = {
        "embed_out.weight": (np.float32, rng.standard_normal((512, 64)).astype("float32")),
        "model.layers.0.self_attn.q_proj.weight": (np.float32, rng.standard_normal((64, 64)).astype("float32")),
        "model.layers.0.input_layernorm.weight": (np.float32, rng.standard_normal((64,)).astype("float32")),
    }
    cfg = derive_config({n: w for n, (_, w) in tensors.items()}, objective="min-bits", use_llm=False)
    assert cfg["embed_out.weight"].method == "int8"
    assert cfg["model.layers.0.self_attn.q_proj.weight"].method == "rvq"
    assert cfg["model.layers.0.input_layernorm.weight"].keep_fp16
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/autoquant/test_orchestrator.py -q`
Expected: FAIL — `ModuleNotFoundError: orka.autoquant.orchestrator`

- [ ] **Step 3: Write minimal implementation**

```python
# orka/autoquant/orchestrator.py
"""The autoquant chain: for each tensor, classify role -> probe -> policy decide -> escalate
to the LLM if a trigger fires (unless use_llm=False) -> cached verdict. Returns the per-tensor
config map. The pack -> pulse-check -> refine loop lives in cmd_autoquant (Task 9/10)."""
from __future__ import annotations
import numpy as np
from orka.autoquant.roles import classify_role
from orka.autoquant.probes import probe_tensor
from orka.autoquant.policy import decide
from orka.autoquant.escalation import should_escalate, signature, Cache, default_cache_path
from orka.autoquant.harness import decide_with_llm
from orka.autoquant.schema import TensorConfig

def derive_config(weights: dict[str, np.ndarray], objective: str, *, use_llm: bool = True,
                  llm_fn=None, cache: Cache | None = None) -> dict[str, TensorConfig]:
    if use_llm and cache is None:
        cache = Cache(default_cache_path())
    out: dict[str, TensorConfig] = {}
    for name, W in weights.items():
        role, role_conf = classify_role(name, tuple(np.shape(W)))
        signals = probe_tensor(W)
        cfg, policy_conf = decide(role, signals)
        if use_llm and should_escalate(role_conf, signals, policy_conf, regressed=False):
            sig = signature(role, tuple(np.shape(W)), signals, objective)
            cached = cache.get(sig)
            if cached is not None:
                cfg = TensorConfig(source="cache", confidence=0.75,
                                   rationale="cached", **{k: cached[k] for k in
                                   ("method", "bits", "stages", "normalization", "keep_fp16")})
            elif llm_fn is not None:
                try:
                    cfg = decide_with_llm(role, tuple(np.shape(W)), signals, objective, llm_fn=llm_fn)
                    cache.put(sig, {k: getattr(cfg, k) for k in
                              ("method", "bits", "stages", "normalization", "keep_fp16")})
                except ValueError:
                    pass  # invalid LLM output -> keep policy cfg
        out[name] = cfg
    return out
```

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest tests/autoquant/test_orchestrator.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add orka/autoquant/orchestrator.py tests/autoquant/test_orchestrator.py
git commit -m "feat(autoquant): orchestrator chain (roles->probe->policy->escalate)"
```

---

### Task 9: CLI — `orka autoquant`

**Files:**
- Modify: `orka/cli/parser.py` (add subparser near the `calc` parser)
- Modify: `orka/cli/commands.py` (add `cmd_autoquant`)
- Test: `tests/autoquant/test_cli.py`

**Context:** follow the existing subcommand pattern: `sub.add_parser(...); p.set_defaults(func=cmd_x)`.
`cmd_autoquant` loads the checkpoint tensors (reuse `orka.inspect`/safetensors loading used
by `cmd_pack`), calls `derive_config`, writes `allocation_map.json`. The litellm transport
is wired here (production llm_fn) but `--no-llm` disables it.

- [ ] **Step 1: Write the failing test**

```python
# tests/autoquant/test_cli.py
import json, numpy as np
from safetensors.numpy import save_file
from orka.cli.commands import cmd_autoquant
import argparse

def test_cmd_autoquant_writes_allocation_map(tmp_path):
    model = tmp_path / "m"; model.mkdir()
    save_file({"embed_out.weight": np.random.randn(128, 32).astype("float32"),
               "model.layers.0.self_attn.q_proj.weight": np.random.randn(32, 32).astype("float32")},
              str(model / "model.safetensors"))
    out = tmp_path / "alloc.json"
    args = argparse.Namespace(model=str(model), objective="min-bits", out=str(out),
                              no_llm=True, target=None, prompts=None)
    assert cmd_autoquant(args) == 0
    m = json.loads(out.read_text())
    assert m["embed_out.weight"]["method"] == "int8"
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/autoquant/test_cli.py -q`
Expected: FAIL — `AttributeError: module 'orka.cli.commands' has no attribute 'cmd_autoquant'`

- [ ] **Step 3a: Implement `cmd_autoquant` in `orka/cli/commands.py`**

Add at end of file:

```python
def cmd_autoquant(args: argparse.Namespace) -> int:
    import json as _json
    from pathlib import Path
    import numpy as np
    from safetensors import safe_open
    from orka.autoquant.orchestrator import derive_config
    from orka.autoquant.schema import to_allocation_map

    model = Path(args.model)
    sfs = sorted(model.glob("*.safetensors"))
    if not sfs:
        print(f"no safetensors in {model}"); return 1
    weights: dict[str, np.ndarray] = {}
    for sf in sfs:
        with safe_open(str(sf), "np") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if t.ndim == 2:                      # quant candidates + norms handled by role
                    weights[k] = t.astype("float32")
                elif t.ndim == 1:
                    weights[k] = t.astype("float32")
    cfg = derive_config(weights, objective=args.objective, use_llm=not args.no_llm)
    Path(args.out).write_text(_json.dumps(to_allocation_map(cfg), indent=2) + "\n")
    n_int8 = sum(1 for c in cfg.values() if c.method == "int8")
    n_rvq = sum(1 for c in cfg.values() if c.method == "rvq")
    n_fp16 = sum(1 for c in cfg.values() if c.method == "fp16")
    print(f"autoquant({args.objective}): {len(cfg)} tensors -> rvq {n_rvq}, int8 {n_int8}, fp16 {n_fp16}")
    print(f"wrote {args.out}")
    return 0
```

- [ ] **Step 3b: Register the subparser in `orka/cli/parser.py`** (after the `calc.set_defaults(func=cmd_calc)` block)

```python
    aq = sub.add_parser("autoquant", help="auto-derive a per-tensor quant config for any model")
    aq.add_argument("model", help="HF model dir (safetensors)")
    aq.add_argument("--objective", choices=["min-bits", "max-quality", "knee"], default="knee")
    aq.add_argument("--out", default="allocation_map.json")
    aq.add_argument("--target", default=None, help="KL/bpw/MB target for min-bits/max-quality")
    aq.add_argument("--prompts", default=None, help="pulse-check prompts file")
    aq.add_argument("--no-llm", action="store_true", help="pure deterministic policy (no LLM)")
    aq.set_defaults(func=cmd_autoquant)
```

Ensure `cmd_autoquant` is imported where the other `cmd_*` are imported in `parser.py`.

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest tests/autoquant/test_cli.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add orka/cli/parser.py orka/cli/commands.py tests/autoquant/test_cli.py
git commit -m "feat(autoquant): orka autoquant CLI subcommand"
```

---

### Task 10: Integration gate — pythia-160m derives the int8 head

**Files:**
- Test: `tests/autoquant/test_integration_pythia.py`

**Context:** the real gate from the spec — on a real model, autoquant must reproduce the
hard-won decision (head=int8, never RVQ) and assign RVQ to the transformer linears, fully
deterministically (`--no-llm`). Marked slow; skips if the base model isn't present.

- [ ] **Step 1: Write the failing/guarded test**

```python
# tests/autoquant/test_integration_pythia.py
import os, json, subprocess, glob, pytest

BASE = glob.glob(os.path.expanduser(
    "~/ai-models/hf-cache/hub/models--EleutherAI--pythia-160m/snapshots/*"))

@pytest.mark.skipif(not BASE, reason="pythia-160m not cached")
def test_autoquant_derives_int8_head(tmp_path):
    out = tmp_path / "alloc.json"
    r = subprocess.run([".venv/bin/python", "orka.py", "autoquant", BASE[0],
                        "--objective", "min-bits", "--no-llm", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = json.loads(out.read_text())
    head = next(v for k, v in m.items() if "embed_out" in k or "lm_head" in k)
    assert head["method"] == "int8"          # the session's hard-won prior, auto-derived
    assert any(v["method"] == "rvq" for k, v in m.items() if "mlp" in k or "attn" in k)
```

- [ ] **Step 2: Run test, verify it passes (or skips)**

Run: `.venv/bin/python -m pytest tests/autoquant/test_integration_pythia.py -q`
Expected: PASS (1 passed) if model cached, else SKIPPED

- [ ] **Step 3: Run the full suite + structural oracle**

Run: `.venv/bin/python -m pytest tests/autoquant/ -q && .venv/bin/python -m pytest tests/ -k oracle -q`
Expected: all autoquant tests PASS; structural oracle `d73e0b19fc38f099` unchanged/green

- [ ] **Step 4: Commit**

```bash
git add tests/autoquant/test_integration_pythia.py
git commit -m "test(autoquant): integration gate - pythia derives int8 head (no-llm)"
```

---

## Self-review

**Spec coverage:** roles (T3), probes (T4), policy (T5), harness/pi-reason (T7), escalation 4-triggers + global cache (T6), orchestrator chain + objectives (T8), schema/allocation_map (T2), priors (T1), CLI 3-objective + --no-llm (T9), integration int8-head gate + oracle (T10). Refine loop (pack→pulse-check→escalate offenders) is scoped into `cmd_autoquant` follow-up — **note:** v1 lands config derivation + the no-LLM path end-to-end; the live pack/pulse-check refine loop and the litellm production `llm_fn` wiring are a fast-follow once the chain is green (flagged so the worker doesn't treat T9 as the full refine loop).

**Placeholder scan:** none — every step has runnable code/commands.

**Type consistency:** `TensorConfig` fields identical across T2/T5/T7/T8; `Signals`/`probe_tensor` consistent T4→T5/T7/T8; `classify_role`/`decide`/`should_escalate`/`signature`/`Cache`/`decide_with_llm` signatures match their call sites in `derive_config`.

**Fast-follow (post-v1, not in this plan):** production litellm `llm_fn` + role router; the pack→pulse-check→refine loop with per-tensor regression attribution; `max-quality`/`knee` budget wiring via `build_allocation`.
