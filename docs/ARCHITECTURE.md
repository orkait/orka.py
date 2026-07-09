# 🏗️ orka package architecture

The `orka/` package is organized by **domain**, so each concern lives in one place and can be optimized in isolation. Implementation moved out of the flat package root into the subpackages below; the old top-level module paths remain importable via thin compat shims (see [Compat shims](#-compat-shims)).

## 📦 Module map

| Subpackage | Responsibility | Key modules |
|---|---|---|
| `orka.core` | Private low-level primitives (no external API) | `_format`, `_tensor`, `_util`, `_checkpoint`, `_features` |
| `orka.codebook` | Codebook fitting | `kmeans` (Lloyd / k-means‖, weighted) |
| `orka.quant` | Quant specs, bit allocation, calibration | `spec`, `allocate`, `activations`, `compensation`, `family`, `semantic` |
| `orka.transforms` | Weight transforms before VQ | `normalize`, rotation |
| `orka.pipeline` | Pack orchestration | `pack`, `pack_pipeline`, `decode`, `strategies/` |
| `orka.inference` | Inference-time VQ layer + kernels | `vq_linear`, `triton_kernels`, `cuda_planes` |
| `orka.qat` | Quantization-aware training | `_core` (QATVQLinear), `train`, `distill` |
| `orka.artifact` | `.orka` artifact operations | `reconstruct`, `export`, `export_gguf`, `correct`, `merge` |
| `orka.eval` | Evaluation + reporting | `metrics`, `verify`, `sweep`, `report`, `hf`, `prompts` |
| `orka.integrations` | External framework hooks | `hf`, `hf_quantizer`, `vllm_quant`, `layers` |
| `orka.autoquant` | Auto config derivation | (arch-agnostic auto-config) |
| `orka.cli` | Command-line entry | `parser`, `commands` |
| `orka.deploy` | Deployment helpers | `kaggle` |
| `orka._runtime` | Runtime/env helpers | (memory caps, device) |
| `orka.data` | Bundled calibration corpus | `calibration.txt` |

## 🔌 Layering (who depends on whom)

```
cli ─┐
     ├─▶ pipeline ─▶ transforms ─▶ codebook ─┐
qat ─┤            ─▶ quant ───────────────────┼─▶ core
     ├─▶ artifact ─▶ inference ───────────────┤
eval ┘            integrations ───────────────┘
```

- `core` is the leaf - it depends on nothing else in `orka` and everything depends on it.
- `pipeline` is the pack orchestrator; `inference` is the decode/serve side.
- `integrations` and `cli` are the outer edges (entry points), depended on by nothing internal.

## 🩹 Compat shims

The historical flat paths (`orka.hf`, `orka.reconstruct`, `orka.qat_train`, `orka.metrics`, …) still import - each is a 2-line shim re-exporting from the new home. They preserve the public API and external/deploy scripts during the deprecation window. New code should import the canonical path (`orka.integrations.hf`, `orka.artifact.reconstruct`, …). Private internals (`orka.core._*`) have **no** shim - they were never public; all internal references were updated.

## ✅ Invariants

- **Tests are the structural gate.** `pytest` must stay green across any move. The pack is not byte-reproducible (codebook bytes shift under threaded BLAS), so the contract is `tests/test_golden_oracle.py`: it packs a seeded model through 12 configurations and hashes a manifest-derived fingerprint. A change that moves the combined hash changed pack behaviour.
- **`core` stays dependency-free** within `orka` - if a `core` module needs another subpackage, the boundary is wrong.
- **One concern per subpackage** - a file that mixes pack + eval + integration logic should be split along these lines.
