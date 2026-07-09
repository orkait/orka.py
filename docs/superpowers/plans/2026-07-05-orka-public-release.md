# Orka Compiler Public Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `orkait/orka-compiler` safe and ready to publish: installable package, Apache-2.0 licence, CI that enforces the structural pack-format gate, secret scanning that actually catches HuggingFace tokens, and comments trimmed to the ones that carry constraints.

**Architecture:** Purely additive. No module is moved, no function is split, `pipeline/pack.py` is untouched. Task 1 lands the structural gate first so every later task is verified against it. The only mutation to shared history, a force-push of the already-scrubbed refs, happens last.

**Tech Stack:** Python 3.11, hatchling, pytest, ruff, gitleaks, GitHub Actions.

---

## Context an engineer needs before starting

**The pack format is not byte-deterministic.** Codebook bytes shift under threaded BLAS. The guarantee is *structural*: `golden_oracle.py` packs a seeded synthetic model through 12 configurations and hashes a config-derived fingerprint of each manifest. The combined hash is **`d73e0b19fc38f099`**. If a change moves that hash, the change altered pack behaviour. That is the gate.

**Branch:** all work happens on `c-PE-orka-compiler-public-release-prep`, which already contains the design spec and the two WS0 security commits.

**Already done (do not redo):** history scrubbed via `git filter-repo` (393 commits, 0 leaks), `.gitleaks.toml` added, `orka_smol_kaggle.py` reads `_load_hf_token()`, the leaked token is revoked, backup mirror at `/home/kai/orka-compiler-PRE-SCRUB-backup.git`.

**Verify the gate before you touch anything:**

```bash
cd /mnt/storage/codespace/code/orkait/orka-compiler
.venv/bin/python /home/kai/ai-models/golden_oracle.py | tail -2
# Expected:
#   HASH d73e0b19fc38f099
#   DONE
```

## File structure

| File | Responsibility | Task |
|---|---|---|
| `tests/test_golden_oracle.py` | Structural gate: 12 pack configs, per-config + combined hash | 1 |
| `pyproject.toml` | Packaging metadata, deps, extras, console script | 2 |
| `LICENSE` | Apache-2.0 text | 2 |
| `orka.py` | **deleted** (superseded by console script) | 2 |
| `.gitignore` | secret filename patterns | 3 |
| `.pre-commit-config.yaml` | ruff, ruff-format, gitleaks | 3 |
| `.github/workflows/ci.yml` | pytest + oracle + ruff + gitleaks | 3 |
| `CONTRIBUTING.md` | dev setup, the gate rule, no-secrets rule | 4 |
| `orka/config.py` | typed accessors for the 7 env knobs | 5 |
| `orka/_runtime/limits.py`, `orka/core/_features.py`, `orka/qat/_core.py` | read config, not `os.environ` | 5 |
| `orka/**/*.py` | comment trim | 6 |
| `README.md` | rewritten | 7 |

---

## Task 1: Land the structural gate

Do this first. Every later task is verified by it.

**Files:**
- Create: `tests/test_golden_oracle.py`
- Source to port: `/home/kai/ai-models/golden_oracle.py`

- [ ] **Step 1: Write the test**

The source oracle hardcodes `sys.path.insert(0, "/mnt/storage/...")`. Drop that; pytest runs from the repo root. Keep the synthetic model and the fingerprint function verbatim, since changing either invalidates the recorded hash.

```python
# tests/test_golden_oracle.py
"""Structural gate for pack_checkpoint.

Codebook bytes are not reproducible under threaded BLAS, so this hashes a
config-derived fingerprint of each manifest instead. A change that moves
COMBINED_HASH changed pack behaviour.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from orka.pipeline.pack import pack_checkpoint

COMBINED_HASH = "d73e0b19fc38f099"

PER_CONFIG_HASHES = {
    "default": "5ad81ae4d56c38c4",
    "per-tensor": "5ad81ae4d56c38c4",
    "global": "e6370ee0fedd3cbc",
    "family": "6c1b65b21e0ed8f5",
    "blockmax": "0b77032fe42654ae",
    "slrq": "2a849fadae28809a",
    "chan-blockmax": "cfeade5b47afbe16",
    "multistage": "bcab98f95b147b12",
    "emaq": "bcab98f95b147b12",
    "mse-scale": "2e5c86dbe3daf9e8",
    "outliers": "5ad81ae4d56c38c4",
    "hadamard": "045b4f3c250701a8",
}

CONFIGS = {
    "default": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy"),
    "per-tensor": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy", codebook_mode="per-tensor"),
    "global": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy", codebook_mode="global"),
    "family": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy", codebook_mode="family"),
    "blockmax": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy", normalization="block-max"),
    "slrq": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy", normalization="slrq-block"),
    "chan-blockmax": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy", normalization="channel-block-max"),
    "multistage": dict(group_size=8, codebook_sizes=[16, 16], iterations=3, backend="numpy"),
    "emaq": dict(group_size=8, codebook_sizes=[16, 16], iterations=3, backend="numpy", em_aq_passes=2),
    "mse-scale": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy", normalization="block-max", mse_scale=True),
    "outliers": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy", outlier_frac=0.05),
    "hadamard": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy", rotation="hadamard", rotation_seed=7),
}


def _make_source(path: Path) -> None:
    random.seed(0)

    def rows(r: int, c: int):
        return [[random.gauss(0, 1) for _ in range(c)] for _ in range(r)]

    path.write_text(json.dumps({"tensors": {
        "model.embed_tokens.weight": rows(64, 32),
        "model.layers.0.self_attn.q_proj.weight": rows(32, 32),
        "model.layers.0.self_attn.k_proj.weight": rows(16, 32),
        "model.layers.0.mlp.up_proj.weight": rows(48, 32),
        "model.layers.0.mlp.down_proj.weight": rows(32, 48),
    }}))


def _fingerprint(artifact: Path) -> str:
    man = json.loads((artifact / "manifest.json").read_text())
    struct = {
        "n_stages": man.get("n_stages"),
        "codebook_mode": man.get("codebook_mode"),
        "normalization": man.get("normalization"),
        "tensor_count": man.get("tensor_count"),
        "mse_scale": man.get("mse_scale"),
        "rotation": man.get("rotation"),
        "tensors": sorted(
            [
                {
                    "name": t["name"], "group_size": t.get("group_size"), "shape": t.get("shape"),
                    "n_stages": t.get("n_stages"), "normalization": t.get("normalization"),
                    "index_bits": t.get("index_bits"), "has_outliers": bool(t.get("outlier_count")),
                    "has_salient": bool(t.get("salient_count")), "scale_count": t.get("scale_count"),
                }
                for t in man.get("tensors", [])
            ],
            key=lambda x: x["name"],
        ),
    }
    return hashlib.sha256(json.dumps(struct, sort_keys=True).encode()).hexdigest()[:16]


def _pack(name: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "m.json"
        _make_source(src)
        buf, old = io.StringIO(), sys.stderr
        sys.stderr = buf
        try:
            random.seed(12345)
            np.random.seed(12345)
            pack_checkpoint(src, root / "a.orka", **CONFIGS[name])
            return _fingerprint(root / "a.orka")
        finally:
            sys.stderr = old


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_config_fingerprint_unchanged(name: str) -> None:
    assert _pack(name) == PER_CONFIG_HASHES[name], (
        f"pack behaviour changed for config {name!r}. If deliberate, update "
        f"PER_CONFIG_HASHES and COMBINED_HASH and say why in the commit message."
    )


def test_combined_hash_unchanged() -> None:
    results = {name: _pack(name) for name in CONFIGS}
    combined = hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()[:16]
    assert combined == COMBINED_HASH
```

- [ ] **Step 2: Run it, confirm it passes against current code**

```bash
.venv/bin/python -m pytest tests/test_golden_oracle.py -q
```
Expected: `13 passed` (12 parametrized + 1 combined).

- [ ] **Step 3: Prove the gate actually catches a behaviour change**

A gate you never saw fail is not a gate. Temporarily change a default and confirm a red test:

```bash
sed -i 's/^EMBEDDING_MAX_GROUP_SIZE = 8/EMBEDDING_MAX_GROUP_SIZE = 4/' orka/pipeline/pack_config.py
.venv/bin/python -m pytest tests/test_golden_oracle.py -q 2>&1 | tail -3
```
Expected: failures naming the affected configs.

Revert it:
```bash
git checkout -- orka/pipeline/pack_config.py
.venv/bin/python -m pytest tests/test_golden_oracle.py -q
```
Expected: `13 passed`.

(`EMBEDDING_MAX_GROUP_SIZE = 8` is at `orka/pipeline/pack_config.py:24`. The point of this step is to observe red, then green: a gate never seen failing is not a gate.)

- [ ] **Step 4: Commit**

```bash
git add tests/test_golden_oracle.py
git commit -m "test: bring the pack structural gate into the repo

golden_oracle.py lived outside the repository, so no contributor could
verify that a change left pack_checkpoint's output structure intact. It now
runs under pytest: 12 configurations, per-config fingerprints plus the
combined hash d73e0b19fc38f099.

The guarantee is structural, not byte-for-byte. Codebook bytes shift under
threaded BLAS, so the fingerprint is derived from manifest metadata."
```

---

## Task 2: Packaging

**Files:**
- Create: `pyproject.toml`, `LICENSE`
- Delete: `orka.py`

- [ ] **Step 1: Confirm the console-script target exists**

```bash
.venv/bin/python -c "from orka.cli import main; print(main)"
```
Expected: `<function main at 0x...>`

- [ ] **Step 2: Write `pyproject.toml`**

`numpy` is the only hard dependency; the numpy backend is the deterministic reference path and must work without torch.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "orka-compiler"
version = "0.1.0"
description = "Vector-quantization compiler for large language model weights"
readme = "README.md"
requires-python = ">=3.10"
license = { file = "LICENSE" }
authors = [{ name = "Orkait" }]
keywords = ["quantization", "compression", "llm", "vector-quantization"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
torch = ["torch>=2.1", "triton>=2.1; platform_system == 'Linux'"]
hf = ["transformers>=4.40", "safetensors>=0.4", "huggingface_hub>=0.23", "datasets>=2.19"]
dev = ["pytest>=8.0", "ruff>=0.6"]

[project.scripts]
orka = "orka.cli:main"

[project.urls]
Homepage = "https://github.com/orkait/orka-compiler"
Source = "https://github.com/orkait/orka-compiler"

[tool.hatch.build.targets.wheel]
packages = ["orka"]

[tool.ruff]
line-length = 100
target-version = "py310"
exclude = [".venv", "llama.cpp", "ui", "deploy"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 3: Add the Apache-2.0 licence**

```bash
curl -sSL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE
head -2 LICENSE && wc -l LICENSE
```
Expected: the Apache header, 202 lines. If there is no network, copy the text from `https://www.apache.org/licenses/LICENSE-2.0.txt` manually.

- [ ] **Step 4: Install the package and check the entry point**

```bash
.venv/bin/python -m pip install -e '.[dev]' 2>&1 | tail -2
.venv/bin/orka --help 2>&1 | head -3
```
Expected: usage text. This proves `orka.py`'s `sys.path` hack is no longer needed.

- [ ] **Step 5: Delete the shim and confirm nothing referenced it**

```bash
rg -n 'orka\.py' --glob '!docs/**' --glob '!.venv/**' . | grep -v 'orka/.*\.py' || echo "no references"
git rm orka.py
.venv/bin/orka --help >/dev/null && echo "entry point still works"
```

- [ ] **Step 6: Run the gate and the suite**

```bash
.venv/bin/python -m pytest tests/test_golden_oracle.py -q
.venv/bin/python -m pytest -q 2>&1 | tail -3
```
Expected: oracle `13 passed`; full suite green.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml LICENSE
git commit -m "build: package as orka-compiler under Apache-2.0

PEP 621 metadata with hatchling. numpy is the only hard dependency so the
deterministic numpy backend installs without torch; torch, hf and dev are
extras. Adds the orka console script, which replaces the orka.py sys.path
shim."
```

---

## Task 3: Secret scanning, hooks, and CI

**Files:**
- Modify: `.gitignore`
- Create: `.pre-commit-config.yaml`, `.github/workflows/ci.yml`

- [ ] **Step 1: Extend `.gitignore` with credential filenames**

Append:

```gitignore
# credentials
*.token
kaggle.json
credentials.json
*.pem
.env.*
```

- [ ] **Step 2: Verify the custom gitleaks rules catch a planted token**

Stock gitleaks has no HuggingFace rule; this repo's `.gitleaks.toml` adds one. Prove it works before trusting it.

```bash
printf 'login(token="hf_%s")\n' "$(head -c 40 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 34)" > /tmp/leak_probe.py
gitleaks detect --source /tmp --config .gitleaks.toml --no-git --no-banner --redact 2>&1 | tail -2
rm -f /tmp/leak_probe.py
```
Expected: `leaks found: 2` (the `huggingface-token` and `huggingface-inline-login` rules both fire).

- [ ] **Step 3: Write `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.4
    hooks:
      - id: gitleaks
        args: ["--config", ".gitleaks.toml"]
```

- [ ] **Step 4: Write `.github/workflows/ci.yml`**

Two test jobs, for a reason. At least six test modules import `torch` at module scope (`test_lattice.py`, `test_ans.py`, `test_qat_memory.py`, `test_arch.py`, `test_build_vq_linear.py`, `test_trellis.py`), so a torch-free run cannot collect the full suite. The `gate` job therefore installs numpy only and proves two things that matter for users: the package installs without torch, and the structural oracle passes on the deterministic reference backend. The `test` job installs the torch extra and runs everything.

```yaml
name: ci

on:
  push:
    branches: [main]
  pull_request:

jobs:
  gate:
    name: structural gate (numpy only, no torch)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e .
      - name: Package works without torch
        run: |
          python -c "import torch" 2>/dev/null && { echo "torch unexpectedly present"; exit 1; } || true
          orka --help > /dev/null
      - name: Structural pack gate
        run: pip install pytest && pytest tests/test_golden_oracle.py -q

  test:
    name: full suite
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e '.[dev,torch]' --extra-index-url https://download.pytorch.org/whl/cpu
      - run: pytest -q

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff
      - run: ruff check orka tests

  secrets:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: gitleaks/gitleaks-action@v2
        env:
          GITLEAKS_CONFIG: .gitleaks.toml
```

`triton` is declared `platform_system == 'Linux'` in the torch extra, so the CPU wheel index resolves cleanly on the ubuntu runners.

- [ ] **Step 5: Confirm the gate job's premise locally**

Thirty module-scope `import torch` statements exist, in `orka/qat/`, `orka/inference/`, `orka/quant/{lattice,lattice_pack,trellis}.py` and `orka/integrations/`. That is fine: the lazy PEP 562 `orka/__init__.py` and the function-local imports on the pack path keep those modules off the numpy code path. Verify that rather than trusting it:

```bash
.venv/bin/python - <<'PY'
import builtins, sys
real = builtins.__import__
def blocked(name, *a, **k):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("No module named 'torch' (simulated)")
    return real(name, *a, **k)
builtins.__import__ = blocked
sys.path.insert(0, ".")
from orka.pipeline.pack import pack_checkpoint   # numpy pack path
import orka.cli                                   # console-script entry
print("orka imports cleanly without torch")
PY
```
Expected: `orka imports cleanly without torch`.

If this ever fails, a module-scope `import torch` has leaked onto the numpy path. Move it inside the function that needs it; do not add torch to the core dependencies.

- [ ] **Step 6: Commit**

```bash
git add .gitignore .pre-commit-config.yaml .github/workflows/ci.yml
git commit -m "ci: run the structural gate, lint and secret scan on every PR

Stock gitleaks reported a clean scan on this repository while a live
HuggingFace token sat in its history, so CI and the pre-commit hook both
use the repository's .gitleaks.toml rather than the default ruleset.

The gate job needs no GPU: the oracle uses the numpy backend."
```

---

## Task 4: CONTRIBUTING

**Files:**
- Create: `CONTRIBUTING.md`

- [ ] **Step 1: Write it**

```markdown
# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,torch,hf]'
pre-commit install
```

## The pack format gate

`pack_checkpoint` output is **not byte-reproducible** (codebook bytes move under
threaded BLAS), so the invariant we protect is *structural*.
`tests/test_golden_oracle.py` packs a seeded synthetic model through 12
configurations and hashes a fingerprint of each manifest. The combined hash is
`d73e0b19fc38f099`.

Any change under `orka/core/_format.py`, `orka/pipeline/`, `orka/codebook/`, or
`orka/transforms/` must keep that test green:

```bash
pytest tests/test_golden_oracle.py -q
```

If you intend to change pack behaviour, update `PER_CONFIG_HASHES` and
`COMBINED_HASH` in the same commit and explain in the commit message what moved
and why. A silent hash change is a bug, not a rebase artefact.

## Secrets

Never commit a credential. `pre-commit` runs `gitleaks` against `.gitleaks.toml`,
which declares HuggingFace and Kaggle rules that the stock ruleset lacks. Deploy
scripts must resolve tokens through `orka.deploy.kaggle._load_hf_token`, never a
literal.

## Comments

Keep comments that state a constraint or record a measurement
(`"FalconH1 4bpw 1.10 -> 1.50 with error-comp"`). Do not add comments that
narrate what the next line does, or that describe the code's own refactor
history.
```

- [ ] **Step 2: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs: contributing guide with the structural gate rule"
```

---

## Task 5: Centralise environment configuration

Seven knobs are read across four modules. Defaults must be transcribed exactly; a changed default is a behaviour change and the gate will catch it.

**Files:**
- Create: `orka/config.py`
- Modify: `orka/_runtime/limits.py`, `orka/core/_features.py`, `orka/qat/_core.py`

- [ ] **Step 1: Record the current defaults**

```bash
rg -n 'os\.environ|getenv' orka/_runtime/limits.py orka/core/_features.py orka/qat/_core.py orka/deploy/kaggle.py
```
Write down each name and default before editing. Expected knobs: `ORKA_PREFLIGHT_MIN_AVAIL_GB` (5.0), `ORKA_PREFLIGHT_MAX_SWAP_GB` (4.0), `ORKA_HARD_CEILING_GB` (25.0), `ORKA_KMEANS_ITERS` (caller-supplied), `ORKA_ENABLE_AWQ` (""), `HF_TOKEN` (None), `CUDA_VISIBLE_DEVICES` (None).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_config.py
import importlib

from orka import config


def test_defaults_match_documented_values(monkeypatch):
    for var in ("ORKA_PREFLIGHT_MIN_AVAIL_GB", "ORKA_PREFLIGHT_MAX_SWAP_GB", "ORKA_HARD_CEILING_GB"):
        monkeypatch.delenv(var, raising=False)
    importlib.reload(config)
    assert config.preflight_min_avail_gb() == 5.0
    assert config.preflight_max_swap_gb() == 4.0
    assert config.hard_ceiling_gb() == 25.0
    assert config.enable_awq() is False


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("ORKA_HARD_CEILING_GB", "12.5")
    monkeypatch.setenv("ORKA_ENABLE_AWQ", "1")
    assert config.hard_ceiling_gb() == 12.5
    assert config.enable_awq() is True
```

- [ ] **Step 3: Run it, confirm it fails**

```bash
.venv/bin/python -m pytest tests/test_config.py -q
```
Expected: `ModuleNotFoundError: No module named 'orka.config'`

- [ ] **Step 4: Write `orka/config.py`**

```python
"""Environment-driven runtime knobs, resolved in one place.

Values are read per call rather than cached at import, so tests and callers can
change the environment without reloading the module.
"""
from __future__ import annotations

import os

DEFAULT_PREFLIGHT_MIN_AVAIL_GB = 5.0
DEFAULT_PREFLIGHT_MAX_SWAP_GB = 4.0
DEFAULT_HARD_CEILING_GB = 25.0


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def preflight_min_avail_gb() -> float:
    return _float("ORKA_PREFLIGHT_MIN_AVAIL_GB", DEFAULT_PREFLIGHT_MIN_AVAIL_GB)


def preflight_max_swap_gb() -> float:
    return _float("ORKA_PREFLIGHT_MAX_SWAP_GB", DEFAULT_PREFLIGHT_MAX_SWAP_GB)


def hard_ceiling_gb() -> float:
    return _float("ORKA_HARD_CEILING_GB", DEFAULT_HARD_CEILING_GB)


def kmeans_iters(default: int) -> int:
    raw = os.environ.get("ORKA_KMEANS_ITERS")
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def enable_awq() -> bool:
    return bool(os.environ.get("ORKA_ENABLE_AWQ", ""))


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


def cuda_visible_devices() -> str | None:
    return os.environ.get("CUDA_VISIBLE_DEVICES")
```

- [ ] **Step 5: Run the test, confirm it passes**

```bash
.venv/bin/python -m pytest tests/test_config.py -q
```
Expected: `2 passed`

- [ ] **Step 6: Repoint the call sites**

In `orka/_runtime/limits.py`, replace each `float(os.environ.get("ORKA_PREFLIGHT_MIN_AVAIL_GB", "5.0"))` with `config.preflight_min_avail_gb()`, and likewise for the swap and ceiling knobs. In `orka/core/_features.py`, replace `os.environ.get("ORKA_ENABLE_AWQ", "")` with `config.enable_awq()` (note this changes a truthy string to a bool; check the call site treats it as a condition, not a value). In `orka/qat/_core.py`, replace `int(os.environ.get("ORKA_KMEANS_ITERS", iters))` with `config.kmeans_iters(iters)`.

Leave `orka/deploy/kaggle.py` alone: `_load_hf_token` deliberately layers dataset, Kaggle Secrets, then env, and centralising it would flatten that order.

- [ ] **Step 7: Run the gate and the suite**

```bash
.venv/bin/python -m pytest tests/test_golden_oracle.py -q
.venv/bin/python -m pytest -q 2>&1 | tail -3
```
Expected: oracle `13 passed`; suite green. A red oracle here means a default was transcribed wrong.

- [ ] **Step 8: Commit**

```bash
git add orka/config.py tests/test_config.py orka/_runtime/limits.py orka/core/_features.py orka/qat/_core.py
git commit -m "refactor: resolve environment knobs through orka.config

Seven variables were read inline across four modules with their defaults
duplicated at each site. Defaults are transcribed unchanged; the structural
pack gate confirms behaviour is identical."
```

---

## Task 6: Comment trim

721 comment lines. This is judgement work, not a regex. Comments cannot affect the gate, so a green oracle does **not** prove this task was done correctly. The diff review is the verification.

**Files:** `orka/**/*.py`

**Policy:**

| Action | Applies to | Example |
|---|---|---|
| Keep | measured facts, constraints, why-not-the-obvious | `Spearman -0.44, 0/10 top-1`; `FalconH1 4bpw 1.10 -> 1.50 with error-comp`; `P100 sm_60 incompatible` |
| Delete | narration of the next line; refactor scar tissue | `# Check for FP16 overflow just in case`; `# ...live in pack_helpers and are imported above`; `reads exactly as it did inline`; `follow-up; would not change semantics` |
| Rewrite | temporal framing wrapped around a real invariant | `(the old bug) under-counted planar specs by group_size (8x)` becomes `scalar stages cost bits * group_size per vector` |

- [ ] **Step 1: Work one subpackage per commit**

Order: `_runtime`, `core`, `transforms`, `codebook`, `quant`, `pipeline`, `inference`, `eval`, `artifact`, `integrations`, `autoquant`, `qat`, `cli`.

For each, list its comments first:

```bash
rg -n '^\s*#' orka/<subpackage> --type py
```

- [ ] **Step 2: For each comment, apply the table**

If you cannot tell whether a comment records a measurement, keep it. The cost of keeping a redundant comment is one line; the cost of deleting a regression guard is a silent quality regression that reappears months later.

- [ ] **Step 3: After each subpackage, verify and commit**

```bash
.venv/bin/python -m pytest tests/test_golden_oracle.py -q
git diff --stat
git add orka/<subpackage>
git commit -m "style: trim narration and refactor scars from orka/<subpackage>"
```

- [ ] **Step 4: Final review of the whole trim**

```bash
git diff main..HEAD -- orka | grep '^-' | grep -E '^\-\s*#' | less
```
Read every deleted comment line. Any that states a number, a measurement, or a "do not do X because Y" must be restored.

---

## Task 7: README

**Files:** Modify `README.md`

- [ ] **Step 1: Rewrite**

Follow the project README conventions: a centred hero block with the project name, tagline and badges; emoji on section headings; collapsible `<details>` for long option lists; tables over prose; no em dashes.

Required content, all of which must be true of the code as it exists:
- one-paragraph elevator pitch: a vector-quantization compiler for LLM weights
- install: `pip install orka-compiler`, extras `[torch]` and `[hf]`
- quickstart: `orka inspect <model>`, `orka allocate --source <model> --target-bpw 4.0 --out alloc.json`, `orka pack ...`
- a note that the numpy backend is the deterministic reference and torch is optional
- link to `CONTRIBUTING.md` and the structural gate

Do not claim benchmark numbers that are not reproducible from a committed script.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for the public release"
```

---

## Task 8: Final verification, then the single force-push

- [ ] **Step 1: Run everything**

```bash
.venv/bin/python -m pytest -q 2>&1 | tail -3
.venv/bin/python -m pytest tests/test_golden_oracle.py -q
ruff check orka tests
gitleaks detect --source . --config .gitleaks.toml --no-banner --redact 2>&1 | tail -2
```
Expected: suite green; oracle `13 passed`; ruff clean; `no leaks found`.

- [ ] **Step 2: Fresh-clone install smoke test**

```bash
rm -rf /tmp/orka-smoke && git clone -q . /tmp/orka-smoke && cd /tmp/orka-smoke
python -m venv .venv && .venv/bin/pip install -q -e . && .venv/bin/orka --help | head -2
cd - && rm -rf /tmp/orka-smoke
```
Expected: usage text, installed with numpy only (no torch).

- [ ] **Step 3: Open the PR**

```bash
git push -u origin c-PE-orka-compiler-public-release-prep
gh pr create --title "PE orka-compiler: public release preparation" --body-file docs/superpowers/specs/2026-07-05-orka-public-release-design.md
```

- [ ] **Step 4: The force-push - STOP and get explicit human confirmation**

This rewrites `orkait/orka-compiler` for everyone. All 106 branches change SHA. Every existing clone and fork breaks and must be re-cloned. The only undo is `/home/kai/orka-compiler-PRE-SCRUB-backup.git`.

Do not run this without the repository owner saying so in the moment.

```bash
git push --force --all origin
git push --force --tags origin
```

- [ ] **Step 5: Post-push verification**

```bash
rm -rf /tmp/orka-verify && git clone -q git@github.com:orkait/orka-compiler.git /tmp/orka-verify
cd /tmp/orka-verify && gitleaks detect --source . --config .gitleaks.toml --no-banner --redact 2>&1 | tail -2
cd - && rm -rf /tmp/orka-verify
```
Expected: `no leaks found` on the published history.

- [ ] **Step 6: Only then flip the repository to public**

Confirm first that the leaked token is revoked and that the Kaggle `hf-token-private` dataset has been re-seeded with a fresh one, since the old token is dead and the Ornith kernel reads from it.

---

## Out of scope

Deferred until this plan lands and the gate is green in CI:

- splitting `pipeline/pack.py` (776 lines)
- consolidating `pack_helpers` / `_util` into a single utils module
- resolving the `pipeline` and `eval` import cycle that is currently broken by deferred imports in `pipeline/sequential.py`
- removing the 18 back-compat root shims

Each is a module move. Each becomes safe only once `tests/test_golden_oracle.py` runs on every pull request.
