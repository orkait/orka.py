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

| technique | orka location | source paper | verdict |
|---|---|---|---|
| Residual VQ (RVQ stages) | `codebook/`, `spec` | Juang & Gray 1982 | ✅ correct |
| k-means / Lloyd | `_kmeans_torch` | Lloyd 1982 | ✅ correct |
| block-max / channel-block-max scales | `transforms/normalize` | per-group absmax (universal) | ✅ correct |
| power-of-2 block scales (SLRQ) | `_normalize_tensor_slrq_block` | OCP Microscaling MX (E8M0 scale) | ✅ valid (shift-only dequant) |
| salient-per-block + sensitive weights | SLRQ, `pillar_*` | SqueezeLLM 2306.07629, SpQR 2306.03078 | ✅ correct (keep top values fp16) |
| outlier extraction (`w²·E[x²]`) | `transforms/outliers` | SpQR / SqueezeLLM sensitivity, OBQ | ✅ correct (output-impact, not raw magnitude) |
| AWQ per-channel scaling (`W·E[\|x\|]^α`) | `_normalize_tensor_awq` | AWQ 2306.00978 | ⚠️ formula correct; **α is a fixed knob, not the paper's per-layer grid-search** |
| Hadamard / orthogonal incoherence (pack) | `transforms/rotate` | QuIP 2307.13304, QuIP# 2402.04396, QuaRot | ✅ adequate - `largest-pow2-block` (up to full width) |
| GPTQ / LDLQ error compensation | `compensated_assign` | GPTQ 2210.17323, OBQ, GPTVQ | ✅ correct block-OBS; empirical +3.8 dB |
| Hessian-diagonal (AWQ) weighting | `pack.py` `H_diag` | AWQ / OBQ | ✅ correct diagonal output-error; +2.2 dB |
| EM-AQ joint refinement | `strategies/refinement` | Additive Quantization (Babenko-Lempitsky 2014), AQLM 2401.06118 | ✅ correct coordinate descent |
| RD bit allocation | `quant/allocate` | Shoham & Gersho 1988 | ✅ correct (Lagrangian `argmin d+λb`, bisect λ, + greedy fill) |
| output-objective distill | `qat/distill` | GPTQ/QuIP# fine-tune | ✅ correct loss; +5-9 dB |
| codebook sharing (family/global) | `codebook_mode` | universal/shared codebooks | ✅ valid (amortizes codebook tax) |
| index entropy coding | `core/_format` (zlib) | ECVQ (Chou-Lookabaugh-Gray 1989) | ✅ adequate; ANS ~16% better at K=4096, ~1% at K=256 |
| E8 lattice nearest-point | `quant/lattice` | QuIP# 2402.04396, Conway-Sloane | ✅ correct |
| **lattice incoherence** | `quant/lattice` | QuIP# | ✅ **FIXED** this session (was weak 8-D → full input-dim; +6% single-stage ppl) |
| **trellis (TCQ)** | `quant/trellis` + proto | QTIP 2406.11235 | ✅ **corrected** (was buggy "dead"; real impl = competitive with E8) |

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
