# Vendored research sources

Third-party code kept here for study, not built or imported by orka.

| dir | upstream | commit | fetched | license |
|---|---|---|---|---|
| `unsloth/` | https://github.com/unslothai/unsloth | `949b3466e03b6089e3e17ca91121e060025bf96b` | 2026-08-05 04:07:35 -0700 | Apache-2.0 (LICENSE retained in-tree) |

`.git` was removed at the user's request. The commit above is the provenance record;
LICENSE and COPYING are preserved unmodified inside the vendored tree.

## Why it is here

To understand how llama.cpp GGUF quants beat orka RVQ on the same model
(measured: Q4_K_M 1.674 GB / PPL 9.8191 vs orka 1.739 GB / PPL 10.9784).

| `llama-quantize/` | https://github.com/ggml-org/llama.cpp | `e031d9567` | 2026-08-05 | MIT |

Subset only: the five files that implement GGUF quantization
(`src/llama-quant.cpp`, `ggml/src/ggml-quants.{c,h}`, `ggml/src/ggml-common.h`,
`tools/quantize/quantize.cpp`). Taken from upstream/master, not from the orka fork.

## Why k-quants beat orka RVQ, in three mechanisms

1. **Asymmetric, searched scales.** `make_qkx2_quants` does an iterative search over
   `nstep` candidate (scale, min) pairs per sub-block - Q4_K uses 20 steps over
   [-1.0, +1.0] in 0.1 increments - minimising a weighted error. orka uses `block-max`:
   a single symmetric scale, no zero-point, no search.

2. **Importance weighting, even data-free.** The search minimises against
   `weights[l] = av_x + fabsf(x[l])`, so large-magnitude weights dominate the fit. With
   an imatrix it becomes `qw[l] * sqrt(sigma2 + x[l]^2)` - activation importance. orka
   weights all elements of a group equally.

3. **A per-tensor schedule, not one rate.** `use_more_bits(i_layer, n)` promotes attn_v
   and ffn_down to Q6_K on the first eighth, last eighth and every third layer, plus the
   output head. Our Q4_K_M: 148 tensors Q4_K, 19 tensors Q6_K.

`--token-embedding-type` / `--output-tensor-type` match on exact names
(`token_embd.weight`, `output.weight`) and override the schedule. That is the mechanism
unsloth's Q2_K_L preset uses, and the one orka lacks: orka protects the same tensors at
F16 (16 bits) where llama.cpp protects them at q8_0 or Q6_K.
