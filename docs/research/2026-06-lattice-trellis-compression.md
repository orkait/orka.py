# Codebook-free quantization for orka: lattice & trellis (research log)

An autonomous research session probing whether SOTA codebook-free PTQ (QuIP#-style
lattices, QTIP-style trellises) beats orka's VQ for compression. Every number below
is **end-to-end full-model perplexity on SmolLM-135M** (`smol_fullste`, calibration
prompts) unless marked SQNR; all runs on CUDA, weight-only unless noted. fp16
baseline ppl = 59.117.

## TL;DR
- **E8 lattice + incoherence is the keystone win**: PTQ matches orka's 1-hour QAT
  quality at ~4.5 bpw, with **zero codebook and zero training** (1 s/model). Shipped
  as `orka/quant/lattice.py` + `lattice_pack.py`.
- **Trellis: a naive 4-state TCQ fails, but a correct QTIP-style trellis is
  competitive with E8** (NOT dead - that earlier claim was a bug). The 4-state
  primitive (`orka/quant/trellis.py`) is effectively 1-D and explodes on weights.
  A faithful re-implementation (incoherence processing + bitshift shift-register
  trellis + exact 1MAD code, validated vs arXiv:2406.11235) reaches **10.76 dB @
  2.0 bpw vs E8's 11.92 dB** - both near the dim-8 rate-distortion bound, trellis
  beating the memoryless 2-bit optimum (9.3 dB) by +1.5 dB. QTIP only *beats*
  lattices with higher L (>=20, which OOMs at 11 GB), the HYB V>1 code, and
  fine-tuning. See "Trellis re-validation" below.
- **The RD frontier of lattice ≈ orka**, not a breakthrough. Pushing below ~4.5 bpw
  needs LDLQ error feedback + fine-tuning - both of which orka *already has*
  (`compensated_assign`, `distill`); the win is wiring them onto the lattice, not a
  new primitive.
- **7 of 9 "new" ideas were already in orka** (see audit below). The genuine gaps
  were lattice and trellis.

## What translated to real perplexity (the gate)

| method | bpw | ppl ratio | note |
|---|---:|---:|---|
| **E8 lattice e8x1** | ~4.4 | **1.202** | == orka QAT-1500 (1.236), but PTQ, no codebook |
| E8 lattice e8x2 | ~5.5 | 1.021 | near-lossless |
| orka QAT-1500 | 4.5 | 1.236 | 1 hr training |
| orka PTQ (full pack) | 4.5 | 1.689 | |
| VQ rvq-12-12-8 (bare) | 6.0 | 1.512 | weight-only, no scales |
| **trellis R=3/R=4 (weight-only)** | 3-4 | **exploded** | 1-D quantizer, catastrophic |

Single-layer SQNR probes (q_proj/down_proj, output metric `‖X·ΔWᵀ‖`) agreed: E8 has
an ~8.6 dB/bpw RD slope vs orka VQ's 2.6 dB/bpw, at zero codebook.

## What did NOT help (fast-fail log, with the why)

| idea | result | why it failed |
|---|---|---|
| full-width incoherence (rotate all `in`) | entropy identical to 8-D (4.06/4.56/5.05) | 8-D Hadamard already captures the marginal Gaussianization E8 needs |
| per-tensor adaptive scale `c·std` | ≈ fixed scale | rate is set by the bulk distribution, not the scale knob |
| outlier extraction (top 0.5-2%) | range ±300→±37 but **entropy unchanged** | rare tails contribute ~0 entropy; they inflate *range* (fixed-width coding), not *rate* (entropy coding) |
| rANS over zlib for indices | 1% gain @ K=256, 16% @ K=4096 | zlib is already near-entropy for low-K VQ indices |
| trellis on rotated weights (4-state) | still exploded | 4-state TCQ is 1-D; the real QTIP trellis (below) is needed |
| my toy LDLQ on lattice | -34 dB | dropped the `1/Hinv[i,i]` normalization; use orka's correct `compensated_assign` |

## Reasoning: why lattice wins where VQ struggles
- VQ's codebook is **exponential in the vector dimension**, capping orka at group 8;
  at high K the codebook *bytes* overrun the index savings (vq-65536 = 17.9 bpw,
  worse than fp16). The lattice is a **parametric** codebook: infinite effective K,
  zero stored bytes.
- The incoherence rotation makes per-channel scale uniform, so **block scales become
  unnecessary** - row-scaling actually *hurts* (it disrupts the Gaussianization).
- Additive codebooks give `K^M` effective vocabulary from `M·K` storage (add2-256 ≈
  vq-16384 quality at 32× less codebook) - orka's RVQ already does the greedy
  version; the gap to AQLM is the *output objective* + joint assignment.

## Audit: ideas already in orka core (don't re-wire)

| idea | already in orka as |
|---|---|
| output-objective refine (the dominant +9 dB lever) | `distill` (`_output_loss`) |
| Fisher/Hessian weighting | AWQ `sample_weights` = E[x²] |
| additive (residual) | RVQ stages |
| GPTQ/LDLQ error feedback | `compensated_assign` (full Cholesky OBS) |
| incoherence rotation | `transforms/rotate.py` |
| planar/scalar | `rvq-s8` stages |
| index entropy coding | zlib |

## Recommendations (ranked)
1. **Wire the lattice into the pack as a stage type** (`e8`): store rotation seed +
   scales + entropy-coded keys, add a decode kernel. PTQ quality of QAT, no training -
   the best ROI for users.
2. **Lattice + `compensated_assign` (LDLQ)**: the proven-correct error feedback on the
   lattice is the QuIP# recipe for sub-4-bit. Reuse orka's tested OBS, don't reinvent.
3. **Lattice + `distill`** (output-objective fine-tune of per-group scales): the
   dominant lever, already in orka.
4. Skip on this hardware: full-width rotation, outlier extraction, custom rANS -
   all fast-failed above. The QTIP trellis is competitive but not better than E8
   at tractable scale (see below).

## Trellis re-validation (vs arXiv:2406.11235)

An initial 4-state TCQ exploded on weights, which I wrongly generalized to "trellis
is dead." Re-checking against the QTIP paper found two bugs in the bigger prototype:
(1) **no incoherence processing** - QTIP applies a random Hadamard so weights are
i.i.d. Gaussian before the trellis; (2) a pure-`randn` code instead of the exact
**1MAD** `x=(34038481·w+76625530) mod 2^32; x=sum of 4 bytes; (x-510)/147.8`.

Fixed prototype = incoherence + **bitshift shift-register trellis** (2^(L-k) states,
k bits/sample, codeword = code[L-bit window], next_state = window mod 2^(L-k)) +
Viterbi + 1MAD, V=1, L up to 18.

| config (q_proj, 2.0 bpw) | SQNR |
|---|---:|
| 4-state TCQ (toy) | ~3 dB (explodes full-model) |
| QTIP L=16, no incoherence (bug) | 7.8 |
| **QTIP L=16/18 + incoherence + 1MAD** | **10.6 / 10.8** |
| memoryless 2-bit optimum (Lloyd-Max) | 9.3 |
| **E8 lattice @ 1.99 bpw** | **11.92** |

So the correct trellis **beats the memoryless bound by +1.5 dB** (genuine trellis
gain - it works) and is **within ~1 dB of E8** at matched 2.0 bpw. It does not beat
E8 here: at dim 8 both are near-optimal and E8 is the provably-optimal packing.
QTIP's published edge over lattices needs higher L (>=20, which OOMs at 11 GB VRAM),
the HYB V>1 lookup code, and end-to-end fine-tuning. **Verdict: E8 lattice is the
better practical choice on this hardware; the trellis is competitive, not dead.**

## Artifacts
`orka/quant/{lattice,lattice_pack,trellis}.py` + `tests/test_{lattice,trellis}.py`.
Experiment scripts under `/home/kai/ai-models/proto/` (not committed; data-dependent).
