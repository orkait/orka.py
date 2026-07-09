# Orka Compiler: Public Release Preparation

`orkait/orka-compiler` is currently private and carries a live HuggingFace token across 347 commits. This spec covers what must happen before the repository can be made public: purge the credential from history, add the packaging and CI infrastructure the project has never had, bring the byte-format guard into the repository so future refactors are verifiable, and trim comments that describe the code's own history instead of its constraints.

The module layout is deliberately left alone. A codebase audit scored the architecture 9/10 (13 focused subpackages, clean `cli -> pipeline -> {quant, transforms, codebook} -> core -> _runtime` layering, a real Strategy pattern, structural rather than name-based tensor detection). The gap is infrastructure, not organization. Module moves are deferred until the byte gate is in-tree and green.

## Severity

| # | Finding | Severity | Impact |
|---|---|---|---|
| 1 | Live HF token hardcoded at `deploy/kaggle/orka_smol_kaggle.py:65`, present in **347 commits** | CRITICAL | Publishing exposes the credential permanently. A later commit cannot remove it from history, forks, or GitHub caches. |
| 2 | Five further files call `login(token=...)` or read creds inline (`orka_qat_*_kaggle.py`, `orka_modal.py`) | HIGH | Same class of leak; recurrence risk |
| 3 | `golden_oracle.py` lives outside the repo (`/home/kai/ai-models/`) | HIGH | The `.orka` format is byte-non-deterministic. No contributor can verify a change did not corrupt it. |
| 4 | No `pyproject.toml`, `LICENSE`, CI, `CONTRIBUTING`, or pre-commit | HIGH | Not installable, not contributable, no guard against re-committing secrets |
| 5 | 721 comment lines include narration and refactor scar tissue | LOW | Noise; obscures the comments that do carry constraints |

## Goals and non-goals

| Goals | Non-goals |
|---|---|
| Purge secrets from all history | Splitting `pipeline/pack.py` (776 lines) |
| Make the package installable (`pip install orka-compiler`) | Moving modules between subpackages |
| Put the byte gate in CI | Consolidating `utils` |
| Prevent future secret commits | Changing any packed byte |
| Remove low-value comments | Removing measured-fact comments |

**Iron constraint:** no change in this spec may alter the bytes written by `pack_checkpoint`. Comments, packaging, and CI are byte-inert by construction. `orka/config.py` (WS4) touches only env-var reads, which do not participate in the pack format.

## Workstreams

### WS0. Security (blocks publication)

Ordered. Steps 0.1 to 0.4 are independent of the code work; step 0.5 runs last so history is rewritten exactly once.

| Step | Action | Owner |
|---|---|---|
| 0.1 | Revoke the leaked token at `huggingface.co/settings/tokens` | user |
| 0.2 | Replace the copy stored in the Kaggle `hf-token-private` dataset with a fresh token | user |
| 0.3 | `gitleaks detect` across all refs; triage every hit, not just the known token | agent |
| 0.4 | De-hardcode credentials in the 6 deploy scripts. Read from env or Kaggle Secrets. `orka/deploy/kaggle.py::_load_hf_token` already falls back to `UserSecretsClient`, so call sites only need to stop passing literals. | agent |
| 0.5 | `git filter-repo` to purge every discovered secret across all commits, then force-push | agent |
| 0.6 | Announce that all existing clones and forks must be re-cloned | user |

The token must be treated as compromised regardless of the rewrite: it has been readable in a private repo, and it was uploaded to the Kaggle `hf-token-private` dataset during the Ornith work.

### WS1. Packaging

- `pyproject.toml`, PEP 621 metadata, hatchling backend.
- Distribution name `orka-compiler`, import package `orka`.
- Core dependency: `numpy`. Extras:

| Extra | Packages | Why |
|---|---|---|
| `torch` | `torch`, `triton` | GPU pack path; `--backend numpy` must work without it |
| `hf` | `transformers`, `safetensors`, `huggingface_hub`, `datasets` | eval, export, model loading |
| `dev` | `pytest`, `ruff` | contributor tooling |

`gitleaks` is a Go binary, not a Python package. It is wired through the pre-commit hook and the CI action, not through `pip`.
- Console script `orka = orka.cli:main`. This replaces the 15-line `orka.py` `sys.path` shim, which is then deleted.
- `LICENSE`: Apache-2.0. Chosen over MIT for the explicit patent grant, which is relevant to a compression codec.

### WS2. Byte gate

The highest-value item. Today the format's only guard is a script outside the repository.

- Move `golden_oracle.py` into `tests/test_golden_oracle.py`.
- Pack a small committed fixture and assert the artifact hash equals `d73e0b19fc38f099`.
- Run it in CI on the `numpy` backend, which is the deterministic reference path.
- Document in `CONTRIBUTING.md`: any change touching `core/_format.py`, `pipeline/`, `codebook/`, or `transforms/` must keep this test green, or must deliberately bump the hash with justification.

This is what makes the deferred module reorg safe to attempt later.

### WS3. Repository infrastructure

| File | Contents |
|---|---|
| `.github/workflows/ci.yml` | `pytest` (numpy backend), `ruff check`, `gitleaks detect` |
| `.pre-commit-config.yaml` | `ruff`, `ruff-format`, `gitleaks` |
| `CONTRIBUTING.md` | dev setup, the byte-gate rule, no-secrets rule |
| `.gitignore` | add `*.token`, `kaggle.json`, `credentials.json`, `*.pem` |
| `README.md` | rewritten per the project README conventions |

### WS4. Comments and configuration

**Comment policy.** Three actions, applied to inline comments and docstrings alike:

| Action | Applies to | Example |
|---|---|---|
| Keep | measured facts, constraints, why-not-the-obvious rationale | `Spearman -0.44, 0/10 top-1`; `FalconH1 4bpw 1.10 -> 1.50 with error-comp`; `P100 sm_60 incompatible` |
| Delete | narration of the next line; refactor scar tissue | `# Check for FP16 overflow just in case`; `# ...live in pack_helpers and are imported above`; `reads exactly as it did inline`; `follow-up; would not change semantics` |
| Rewrite | temporal framing wrapped around a real invariant | `(the old bug) under-counted planar specs by group_size (8x)` becomes `scalar stages cost bits * group_size per vector` |

A comment that would let a future contributor re-introduce a measured regression is never "low value", regardless of age.

**Configuration.** Seven environment knobs are read across 13 call sites (`_runtime/limits.py`, `core/_features.py`, `qat/_core.py`, `deploy/kaggle.py`):

`ORKA_PREFLIGHT_MIN_AVAIL_GB`, `ORKA_PREFLIGHT_MAX_SWAP_GB`, `ORKA_HARD_CEILING_GB`, `ORKA_KMEANS_ITERS`, `ORKA_ENABLE_AWQ`, `HF_TOKEN`, `CUDA_VISIBLE_DEVICES`

Introduce `orka/config.py` exposing typed accessors with the existing defaults. Call sites read from it instead of `os.environ` directly. Defaults are preserved exactly, so behaviour is unchanged.

## Sequencing

```
branch: c-PE-orka-compiler-public-release-prep

  WS1 packaging ─┐
  WS2 byte gate ─┼─▶ CI green ─▶ WS4 comments + config ─▶ review ─▶ merge
  WS3 infra    ──┘                     │
                                       ▼
  WS0.1-0.2 revoke (user, anytime) ──▶ WS0.3 sweep ─▶ WS0.4 de-hardcode
                                                          │
                                                          ▼
                                              WS0.5 filter-repo (LAST)
                                                          │
                                                          ▼
                                                    flip to public
```

Rationale: the history rewrite must be the final mutation. Rewriting before the code changes land would force a second rewrite.

## Verification

| Claim | Command | Expected |
|---|---|---|
| No secrets in any ref | `gitleaks detect --no-banner` | 0 findings |
| Package installs clean | `pip install -e '.[dev]'` then `orka --help` | usage printed, no `sys.path` hack |
| Numpy path works without torch | `pip install -e .` in a torch-free venv, `orka pack --backend numpy ...` | succeeds |
| Bytes unchanged | `pytest tests/test_golden_oracle.py` | hash `d73e0b19fc38f099` |
| Full suite | `pytest` (38 test files) | green |
| Lint | `ruff check orka tests` | clean |
| No measured facts lost | `git diff` review of comment trim | every deleted line is narration, scar tissue, or a rewritten invariant |

Post-merge smoke: clone fresh from the public URL, `pip install orka-compiler`, `orka inspect` a small model.

## Risks

| Risk | Mitigation |
|---|---|
| `filter-repo` misses a secret | Run `gitleaks` over all refs before and after; treat the known token as rotated regardless |
| Comment trim deletes a regression guard | Every deletion reviewed in diff against the Keep/Delete/Rewrite table |
| Contributors' clones break after rewrite | Announce; the private repo remains as the pre-rewrite archive |
| `config.py` changes a default | Defaults transcribed literally; full suite plus byte gate must stay green |
