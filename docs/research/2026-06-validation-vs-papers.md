# Validation of orka's compression tricks against the source papers

Prompted by finding that my trellis result was wrong (an under-implementation bug,
not the technique). This is a systematic re-check of every load-bearing compression
claim against the actual algorithm / paper, with empirical evidence on SmolLM-135M.

## Verdict table

| trick | paper | orka / my impl | verdict |
|---|---|---|---|
| GPTQ / LDLQ error feedback | GPTQ (2210.17323), GPTVQ | `compensated_assign`: `H=XᵀX/n`, 1% damp, dead-col fix, block-OBS `Wc[:,rest]-=E@(Hinv_bb⁻¹·Hinv_b,rest)` | ✅ **correct** - matches the block-OBS update exactly; empirically +3.8 dB output-SQNR |
| AWQ / Hessian importance | AWQ (2306.00978), OBQ | `E[x²]` per-column = diag(XᵀX), used to weight k-means distortion | ✅ correct as the **diagonal** output-error objective; +2.2 dB. (Naming is loose - true AWQ also does per-channel scaling, which orka has as a separate `awq` norm mode.) |
| E8 lattice nearest-point | QuIP# (2402.04396), Conway-Sloane | `nearest_e8` = min over D8 and D8+½ | ✅ correct - valid lattice points, beats integer rounding on average (packing gain) |
| Incoherence - **main pack** | QuIP/QuIP# | `_hadamard_block_size` = largest pow2 divisor of `cols` (2048→2048, 1536→512, 576→64) | ✅ adequate - near-full-width Hadamard, not the weak case |
| Incoherence - **my lattice.py** | QuIP# | 8-D within-group Hadamard only | ❌ **under-implemented** - full input-dim Hadamard is better (ppl ratio 1.144 vs 1.202 at *lower* bpw). FIXED. |
| Trellis (TCQ) | QTIP (2406.11235) | initial 4-state toy | ❌ was buggy → corrected: real bitshift trellis + incoherence + 1MAD = 10.8 dB @2bpw, competitive with E8 (see trellis doc) |
| Output-objective fine-tune | GPTQ/QuIP# fine-tune, distillation | `distill` `_output_loss = ‖X·ΔWᵀ‖² + damp` | ✅ correct objective; +5-9 dB over weight-MSE at fixed bits |
| residual VQ (additive) | RVQ / AQLM | RVQ stages (greedy) | ✅ correct; AQLM's *joint* assignment (beam) is the only gap |
| index entropy coding | ECVQ | zlib | ✅ adequate - within ~1% of entropy at K=256, ~16% over at K=4096 |
| scalar/planar bpw | - | `s8` packs at group_size=1 = 8 bpw | ✅ correct (was an 8x accounting bug, fixed #129, verified vs on-disk bytes) |

## What was actually wrong (the bugs I found by re-validating)
1. **Trellis** - I had no incoherence + a `randn` code instead of 1MAD; "trellis dead" was false. Corrected: competitive with E8.
2. **lattice.py incoherence** - 8-D within-group only; full input-dim Hadamard is measurably better (validated end-to-end). Fixed in this change.

## What was correct (validated, not changed)
GPTQ/LDLQ, AWQ/Hessian weighting, E8 nearest-point, the main-pack Hadamard,
output-objective distill, residual VQ, planar bpw accounting. These match their
papers both structurally (code) and empirically (measured gains in the expected
direction and magnitude).

## Method note
Where WebFetch returned only abstracts, validation was done by reading orka's source
against the known algorithm and confirming empirically (output-SQNR / perplexity
moves in the predicted direction and magnitude). Single-model (smol) probes -
directional, not a multi-model benchmark.
