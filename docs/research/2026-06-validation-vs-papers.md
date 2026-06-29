# Validation of orka's compression techniques against the source papers

A full audit of every load-bearing compression trick in orka, checked against the
algorithm in its source paper and (where cheap) empirically on SmolLM-135M.
Triggered by finding that two of my *new* prototypes (trellis, lattice incoherence)
were under-implementation bugs, not weak techniques - so I re-checked everything.

## Headline
orka's **existing** quantization arsenal is correct and grounded in the literature.
The only bugs were in code I added this session (lattice + trellis incoherence),
both now fixed. No correctness issue found in the shipped pack/quant core.

## Full verdict table

CUDA column: ✅ = compute runs on GPU under the (default) torch backend;
🟡 = partial/has a CPU step; ⚪ = legitimately CPU (storage/entropy-coding/scalar
bookkeeping). `.cpu()` calls purely for disk serialization are not counted as CPU
compute.

| technique | orka location | source paper | verdict | CUDA |
|---|---|---|---|---|
| Residual VQ (RVQ stages) | `codebook/`, `spec` | Juang & Gray 1982 | ✅ correct | ✅ |
| k-means / Lloyd + assign | `_kmeans_torch`, `_assign_kernel` | Lloyd 1982 | ✅ correct | ✅ Triton fp16 argmin |
| block-max / channel-block-max scales | `transforms/normalize` | per-group absmax (universal) | ✅ correct | ✅ torch path |
| power-of-2 block scales (SLRQ) | `_normalize_tensor_slrq_block` | OCP Microscaling MX (E8M0 scale) | ✅ valid (shift-only dequant) | ✅ |
| salient-per-block + sensitive weights | SLRQ, `pillar_*` | SqueezeLLM 2306.07629, SpQR 2306.03078 | ✅ correct (keep top values fp16) | ✅ (`.cpu()` only to store) |
| outlier extraction (`w²·E[x²]`) | `transforms/outliers` | SpQR / SqueezeLLM sensitivity, OBQ | ✅ correct (output-impact) | ✅ topk on device |
| AWQ per-channel scaling (`W·E[\|x\|]^α`) | `_normalize_tensor_awq` | AWQ 2306.00978 | ⚠️ formula correct; **α fixed, not grid-searched** | ✅ |
| Hadamard incoherence (pack) | `transforms/rotate` | QuIP 2307.13304, QuIP# 2402.04396, QuaRot | ✅ `largest-pow2-block` | ✅ FWHT on device |
| orthogonal incoherence (pack) | `transforms/rotate` | QuIP | ✅ correct | 🟡 random Q via **CPU numpy QR** once/tensor (opt-in; hadamard default is GPU) |
| GPTQ / LDLQ error compensation | `compensated_assign` | GPTQ 2210.17323, OBQ, GPTVQ | ✅ correct block-OBS; +3.8 dB | ✅ `H=XᵀX`, cholesky on device |
| Hessian-diagonal (AWQ) weighting | `pack.py` `H_diag` | AWQ / OBQ | ✅ correct diagonal; +2.2 dB | ✅ **FIXED here** (was `as_tensor` on CPU → now `device=`) |
| EM-AQ joint refinement | `strategies/refinement` | Additive Quant (Babenko-Lempitsky 2014), AQLM 2401.06118 | ✅ correct coordinate descent | ✅ k-means on device |
| RD bit allocation | `quant/allocate` | Shoham & Gersho 1988 | ✅ correct (Lagrangian + greedy) | 🟡 distortion probes GPU; the λ-bisection / greedy solver is CPU scalar (negligible, not tensor work) |
| output-objective distill | `qat/distill` | GPTQ/QuIP# fine-tune | ✅ correct loss; +5-9 dB | ✅ GPU-only since #130 |
| codebook sharing (family/global) | `codebook_mode` | universal/shared codebooks | ✅ valid | ✅ |
| index entropy coding | `core/_format` (zlib) | ECVQ (Chou-Lookabaugh-Gray 1989) | ✅ adequate; ANS ~16% better at K=4096 | ⚪ CPU (entropy coder is inherently CPU; storage layer) |
| LS row-scale refine (`--mse-scale`) | `strategies/refinement` | least-squares scale | ✅ correct | ⚪ CPU by design (opt-in, tiny `[n_blocks]` vectors) |
| E8 lattice nearest-point | `quant/lattice` | QuIP# 2402.04396, Conway-Sloane | ✅ correct | ✅ |
| **lattice incoherence** | `quant/lattice` | QuIP# | ✅ **FIXED** (weak 8-D → full input-dim; +6% single-stage ppl) | ✅ |
| **trellis (TCQ)** | `quant/trellis` + proto | QTIP 2406.11235 | ✅ **corrected** (competitive with E8) | ✅ Viterbi on device |

### Device review summary
- **Hot compute is GPU-only** under the default torch backend: VQ/k-means/assign,
  all normalizations, outlier/salient scoring, Hadamard rotation, GPTQ/LDLQ, EM-AQ,
  distill, lattice, trellis, and the allocator's distortion probes.
- **Fixed here**: the pack's `E[x²]` Hessian-diagonal was computed on CPU
  (`as_tensor` without `device`); now on the pack device - same class as the distill
  CPU-sync bug fixed in #130.
- **Legitimately CPU** (not worth moving): zlib entropy coding, the opt-in
  `mse_scale` scale-refine (tiny vectors), the Lagrangian/greedy allocation solver
  (scalar arithmetic over the small RD table, not weight tensors), and all `.cpu()`
  calls that only serialize codebooks/indices to disk.
- **One opt-in CPU path remains**: orthogonal-rotation Q is generated via numpy QR
  on CPU (once per tensor). The default rotation is Hadamard, which is GPU. Could be
  moved to `torch.linalg.qr` on device if orthogonal rotation ever becomes hot.

## Bugs found by this validation (all in code I added this session)
1. **Trellis** - "dead" was wrong; missing incoherence + `randn` instead of 1MAD. Real impl beats memoryless 2-bit by +1.5 dB, competitive with E8 (10.8 vs 11.9 dB). (#132)
2. **lattice.py incoherence** - 8-D within-group instead of QuIP# full input-dim, plus a seed-keying bug (compress vs reconstruct). Fixed → single-stage ppl 1.197 → 1.121. (#133)

## Minor gaps (correct but simplified, documented not bugs)
- **AWQ α** is a user knob, not auto grid-searched per layer (the AWQ paper searches α∈[0,1]). A sweep is possible via the flag but not automatic.
- **Index entropy coding** is zlib, ~16% above entropy at high K - a real ANS coder would recover it (low priority; ~5% of total size).
- **AQLM joint assignment** - orka's RVQ is greedy/residual; AQLM beam-searches the M-tuple. orka's EM-AQ refinement narrows but doesn't close this.

## Method note
WebFetch returned only abstracts for several arXiv papers, so validation was by
reading orka's source against the known algorithm and confirming empirically
(output-SQNR / perplexity move in the predicted direction and magnitude). Single-
model (smol) probes - directional, not a multi-model benchmark.
