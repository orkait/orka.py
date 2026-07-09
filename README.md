<div align="center">

# 🐙 orka

**Squish LLM weights down to ~2 bits each with vector quantization, then run them.**

A residual-vector-quantization compressor for transformer weights: fit per-tensor codebooks, store indices as bit-planes, recover quality with QAT, and export to GGUF or vLLM. No magic, just codebooks and honest engineering.

[![python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![torch](https://img.shields.io/badge/PyTorch-optional-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![format](https://img.shields.io/badge/format-RVQ_+_12--bit_bit--planes-4f8ff7)](#-how-it-works)
[![bits](https://img.shields.io/badge/weights-~2_bits-7c3aed)](#-honest-benchmarks)
[![ci](https://github.com/orkait/orka.py/actions/workflows/ci.yml/badge.svg)](https://github.com/orkait/orka.py/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache_2.0-green)](#-license)

</div>

---

## 🧭 What this actually is

orka takes a model's linear weights and replaces each little group of 8 numbers with **an index into a learned codebook**. Store the tiny indices instead of the fat weights, and the file shrinks to roughly **2 bits per weight** (8x smaller than fp16). Then, because raw 2-bit quantization is rough, it can **fine-tune the codebooks back toward the original** (QAT) so the model stays smart.

That's the whole trick. It is not a new number format, not a kernel-fusion miracle, not "lossless." It's residual vector quantization (RVQ) applied carefully, with the boring-but-important parts (Hessian weighting, bit allocation, scale refinement, QAT) done properly.

## ⚡ Quick start

```bash
# core install: no torch. The numpy backend is the deterministic reference path.
pip install orka-compiler

# GPU packing and HF model loading are extras
pip install 'orka-compiler[torch,hf]'

# 1. compress a model -> a .orka artifact (~2 bits/weight)
orka pack  ./SmolLM-135M  --out model.orka \
    --quant-mode rvq-12-4 --normalization block-max --device cuda

# 2. check what it cost you (perplexity vs the original)
orka verify  model.orka

# 3. decode back to safetensors, or export to a runtime
orka reconstruct  model.orka --out recon.safetensors --format safetensors
orka export-vllm   model.orka --out ./model-vllm
```

Want orka to pick the settings for you?

```bash
# autoquant's --target is a KL budget against fp16, not a bit-width
orka autoquant ./model --objective min-bits --target 0.05 --out alloc.json

# for an explicit bit-width target, allocate is the command with --target-bpw
orka allocate ./model --target-bpw 4.0 --out alloc.json
```

`python -m orka <command>` works identically if you prefer not to rely on the console script.

## 🔬 How it works

```
weight group [8 values]
  ──fit────▶  per-tensor codebook (e.g. 4096 entries)        ← k-means, Hessian-weighted
  ──encode─▶  nearest-entry index per group                  ← ~2 bits/weight
  ──store──▶  indices as 12-bit BIT-PLANES (lo byte + packed hi)
  ──QAT────▶  fine-tune codebooks/weights back toward fp16    ← optional, recovers quality
  ──decode─▶  scale x sum(codebook[index]) (+ sparse correction)
```

<details>
<summary>📐 The pieces, and where they live</summary>

| Subpackage | Job |
|---|---|
| `orka.core` | private primitives - format, tensor, checkpointing |
| `orka.codebook` | k-means / k-means‖ codebook fitting |
| `orka.quant` | quant specs, bit **allocation** (water-filling), activation calibration |
| `orka.transforms` | normalization + rotations before VQ |
| `orka.pipeline` | the **pack** orchestration |
| `orka.inference` | the inference-time `VQLinear` + Triton/CUDA kernels |
| `orka.qat` | quantization-aware training (QATVQLinear, distillation) |
| `orka.autoquant` | probe, policy and escalation loop behind `orka autoquant` |
| `orka.artifact` | `.orka` ops - reconstruct, export, gguf, merge, correct |
| `orka.eval` | metrics, verify, sweeps, reports |
| `orka.integrations` | HF quantizer + vLLM hooks |
| `orka.deploy` | Kaggle / Modal bootstrap helpers |
| `orka.config` | the environment knobs, resolved in one place |

Full map: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
</details>

## 📊 Honest benchmarks

Real numbers, real caveats. Perplexity on wikitext-2, SmolLM-135M. Lower is better. We benchmarked against llama.cpp's k-quants because pretending the competition doesn't exist is how you fool yourself.

| method | bits/wt | ppl | vs fp16 | verdict |
|---|--:|--:|--:|---|
| fp16 | 16 | ~23.7 | - | the original |
| orka strong PTQ (vq-16) | 2.0* | ~30 | +28% | near-fp16, but big codebooks eat the savings |
| **orka planar + QAT** | 2.0 | ~42 | +79% | small + deployable, but not the quality leader |
| **llama.cpp Q2_K** | ~2.6 | ~23 | +23% | mature k-quants are good; on tiny models they win |
| llama.cpp Q4_K_M | ~4.5 | ~20 | +9% | more bits, better quality |

`*` "2-bit indices" but per-tensor codebooks add overhead - on a 135M model a `vq-16` artifact is barely smaller than fp16. The honestly-compressed orka config is the **planar** one (~5x smaller on disk).

<details>
<summary>How these numbers were measured, and what is missing</summary>

Sliding-window (non-overlapping, 512-token context) perplexity over the wikitext-2 `test` split, fp32 forward, one 12GB consumer GPU. The method is the `_wikitext_ppl` helper in `deploy/kaggle/orka_qat_hi05b_kaggle.py`.

**There is no single committed command that regenerates this table.** `orka eval` measures prompt-loss perplexity, not the wikitext-2 sweep. Treat the numbers as a record of what we measured, not as a reproducible benchmark, until a first-class harness lands. If you reproduce different numbers, we would rather hear it than not.
</details>

### 🫡 The honest takeaways

- **2 bits per weight, near-fp16 quality is real** with the full machinery (Hessian weighting + allocation + QAT) - on the right config.
- **On a 135M model, mature k-quants (Q2_K) still beat orka's small-file config on quality-per-byte.** We are not ahead here, and we will not pretend we are.
- **Codebook overhead amortizes with model size.** Tiny models are orka's worst case (the embedding table alone can be 20%+ of params). The interesting territory is larger models - that's where we expect VQ to pull ahead.
- **More QAT levers is not always better.** Training the full weights on a small calibration set *overfit* and did worse than light codebook-only tuning. Measured, not guessed.

## 🛠️ CLI

```
pack          compress a model -> .orka          allocate    water-fill bits per tensor
verify        decode + measure error             reconstruct .orka -> safetensors/json
distill       QAT fine-tune                       export-vllm .orka -> vLLM dir
correct       add low-rank/sparse correction     merge-orka  combine artifacts
autoquant     auto-pick the config               eval/sweep  benchmark configs
```

`orka <command> --help` for the details.

## ⚙️ Environment knobs

All of these resolve through [`orka/config.py`](orka/config.py).

| Env var | Default | Effect |
|---|---|---|
| `ORKA_KMEANS_FAISS` | off | `=1` swaps the unweighted CUDA k-means for faiss GPU Lloyd (~2x faster at equal reconstruction MSE, byte-deterministic per seed). Requires `pip install faiss-gpu-cu12`; falls back to the built-in torch path if faiss is missing or the tensor uses Hessian/importance weighting. Off by default so packs stay reproducible regardless of whether faiss is installed. Truthy values: `1`, `true`, `yes`. |
| `ORKA_ENABLE_AWQ` | off | `=1` enables the legacy AWQ path, which needs external calibration data. Truthy values: `1`, `true`, `yes`, `on`. |
| `ORKA_HARD_CEILING_GB` | `25.0` | Upper bound on the RAM cap, whatever the CLI asks for. Raise it on machines larger than 32GB. |
| `ORKA_PREFLIGHT_MIN_AVAIL_GB` | `5.0` | Refuse to start if `MemAvailable` is below `workload_budget + this`. |
| `ORKA_PREFLIGHT_MAX_SWAP_GB` | `4.0` | Refuse to start if swap in use exceeds this (the system is already thrashing-prone). |
| `ORKA_KMEANS_ITERS` | caller's value | Override the Lloyd iteration count, for quick validation runs. |

## 🧪 Development

```bash
pip install -e '.[dev,torch,hf]'
pre-commit install

pytest -q                            # full suite
pytest tests/test_golden_oracle.py   # the structural pack gate
```

`pack_checkpoint` output is **not byte-reproducible** (codebook bytes move under threaded BLAS), so the invariant we protect is structural: `tests/test_golden_oracle.py` packs a seeded model through 12 configurations and hashes a fingerprint of each manifest. Any change that moves the combined hash changed pack behaviour. See [CONTRIBUTING.md](CONTRIBUTING.md).

The package is organized by domain (see the table above). Old flat import paths (`orka.hf`, `orka.reconstruct`, ...) still work via compat shims while callers migrate to the canonical ones.

## 📄 License

[Apache License 2.0](LICENSE).

---

<div align="center">
<sub>Made with codebooks, a constrained GPU, and a refusal to fudge the benchmark numbers.</sub>
</div>
