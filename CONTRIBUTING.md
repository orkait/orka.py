# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,torch,hf]'
pre-commit install
```

The `numpy` backend is the deterministic reference path and needs no torch:

```bash
pip install -e .    # numpy only
orka --help
```

## The pack format gate

`pack_checkpoint` output is **not byte-reproducible**. Codebook bytes move under
threaded BLAS. The invariant we protect is therefore *structural*.

`tests/test_golden_oracle.py` packs a seeded synthetic model through 12
configurations and hashes a fingerprint derived from each manifest: per-tensor
`group_size`, `shape`, `n_stages`, `index_bits`, `normalization`, outlier and
salient presence, and `scale_count`. The combined hash is `d73e0b19fc38f099`.

Any change under `orka/core/_format.py`, `orka/pipeline/`, `orka/codebook/`, or
`orka/transforms/` must keep it green:

```bash
pytest tests/test_golden_oracle.py -q
```

If you intend to change pack behaviour, update `PER_CONFIG_HASHES` and
`COMBINED_HASH` in the same commit and say in the commit message what moved and
why. A silent hash change is a bug, not a rebase artefact.

The gate runs on the numpy backend, so CI needs no GPU.

## Secrets

Never commit a credential. Deploy scripts must resolve tokens through
`orka.deploy.kaggle._load_hf_token`, which tries the mounted dataset, then
Kaggle Secrets, then `HF_TOKEN`. Never pass a literal to `login(token=...)`.

`pre-commit` and CI run `gitleaks` against `.gitleaks.toml` rather than the
default ruleset. This is not paranoia: scanning this repository's history with
stock gitleaks reports `no leaks found` while a live HuggingFace token is
present in `HEAD`. The repository rules declare HuggingFace and Kaggle patterns
explicitly.

## Comments

Keep a comment when it states a constraint or records a measurement:

```python
# scalar-quant proxy measured anti-correlated with full VQ (Spearman -0.44, 0/10 top-1)
# FalconH1 4bpw 1.10 -> 1.50 with error-comp on recurrent blocks
```

Delete a comment that narrates the next line, or that describes the code's own
refactor history:

```python
# Check for FP16 overflow just in case          <- says what the code says
# ...live in pack_helpers and are imported above <- talks to a past reviewer
```

If you cannot tell whether a comment records a measurement, keep it. A redundant
comment costs one line. A deleted regression guard costs a silent quality
regression that reappears months later.

## Lint

```bash
ruff check orka tests
```

`B905` (`zip(strict=)`) and `F841` (unused locals) are not enforced yet: the
former changes behaviour by raising on length mismatch and needs per-site review,
the latter has legacy sites that bind side-effecting calls. Both are a backlog to
burn down, not a licence to add more.
