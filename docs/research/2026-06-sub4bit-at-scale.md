# Sub-4-bit compression at scale (Qwen2.5-1.5B)

Validating orka's compression at a real ~1.5B model (vs the 135M smol used until
now, which sits below the quality knee). All runs CUDA, capped at 10 GB VRAM.
fp16 baseline ppl = 10.55 on the bundled calibration set.

## Headline
**orka's existing validated stack - Hessian weighting + LDLQ error-compensation +
distill - delivers ratio 1.16 at 4 bpw / 4.1x compression on Qwen2.5-1.5B.** That
beats the new E8 lattice (1.20 at 6.6 bpw / only 1.9x). The lattice/trellis work
was valuable (and exposed bugs), but the best sub-4-bit path is orka's *existing*
machinery, properly combined.

## Results

| config | bpw | ppl ratio | compression |
|---|---:|---:|---:|
| plain VQ (Hessian-weighted) | 2.5 | 5.31 | 6.5x |
| VQ + distill (200 steps) | 2.5 | 3.57 | 6.5x |
| VQ + **LDLQ** | 2.5 | 2.91 | 6.5x |
| VQ + **LDLQ** | 4.0 | 1.52 | 4.1x |
| VQ + **LDLQ + distill** | 4.0 | **1.16** | 4.1x |
| E8 lattice (adaptive scale) | 6.6 | 1.20 | 1.9x |

## What each lever contributes (at scale)
- **Hessian-diagonal weighting alone is not enough** at 2.5 bpw (5.31) - aggressive
  sub-4-bit needs more than diagonal importance.
- **LDLQ (block-OBS error-compensation) is the big lever**: 5.31 -> 2.91 at 2.5 bpw,
  and gets to 1.52 at 4 bpw. This is the validated `compensated_assign`.
- **distill closes the gap**: 1.52 -> 1.16 at 4 bpw (300 steps).
- **The lattice is RD-mediocre without LDLQ** - it needs 6.6 bpw for 1.20; the VQ
  stack does better at 4 bpw. (The lattice's value is PTQ simplicity / no training,
  not best RD.)

## Lessons (the reason this validation mattered)
1. **Validate at scale.** The lattice's fixed absolute scale worked by luck on smol
   and exploded on Qwen (ratio 5307x) until fixed to adaptive per-tensor scale.
2. **No shortcuts to sub-4-bit.** Plain VQ at 2.5 bpw is unusable (5.31); the
   quality only comes from the full stack (Hessian + LDLQ + distill).
3. **orka already had the winning pieces** - the session's audit found 7/9 SOTA
   tricks already in core; combining them (not the new lattice) is the sub-4-bit win.

## Next levers
- **LDLQ + distill at 3 bpw** (between 2.5 and 4) to map the usable RD frontier.
- **Real QAT** (`orka.qat.train`) instead of light distill - likely pushes 4 bpw
  below 1.1; VRAM-tight on 1.5B at 10 GB, needs `--checkpoint-quantize`.
- **Lattice + LDLQ** - now that the lattice transfers, adding LDLQ (rotated-Hessian)
  could close its RD gap; needs the QuIP#-style incoherent-Hessian build.
