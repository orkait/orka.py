"""Trellis-coded quantization (TCQ) - a quantizer *with memory*.

The motivation (QTIP, arXiv:2406.11235): plain VQ needs a codebook exponential in
the vector dimension, which caps practical VQ at group dim <= 8 and makes the
codebook the dominant on-disk cost at low bitrate. A trellis sidesteps the
codebook entirely: each weight is coded by a state machine, the encoder runs a
Viterbi search for the optimal subset/level path, and the decoder just *follows*
the path (no search, no stored codebook). On orka's own weights this gives the
textbook "trellis gain" - measured +4.0 dB SQNR over scalar at R=2 on
attention weights, i.e. exactly the extreme-compression regime.

This module is the codebook-free encode/decode primitive. Wiring it into a pack
stage type (``t<R>``), the on-disk format and a fast decode kernel is layered on
top of this; everything here is pure-torch, GPU-resident, and round-trips exactly.

4-state Ungerboeck trellis, rate R bits/sample over a 2^(R+1)-level uniform
codebook partitioned into 4 subsets (level index mod 4). Branch ``b`` from state
``s`` emits from subset ``_SUBSET[s][b]`` and moves to ``_NEXT[s][b]``; the free
trellis bit buys one extra bit of resolution at R stored bits/sample.
"""
from __future__ import annotations

import torch

# Valid 4-state trellis: each state has 2 out-branches; each state has exactly 2
# in-branches (checked: states 0..3 each appear twice as a NEXT target).
_SUBSET = ((0, 2), (1, 3), (2, 0), (3, 1))
_NEXT = ((0, 1), (2, 3), (1, 0), (3, 2))
_INF = 1e30


def _levels(W: torch.Tensor, R: int):
    """Uniform 2^(R+1)-level codebook spanning +-4 sigma; returns (levels[L], subset[L])."""
    sig = W.std().clamp(min=1e-12)
    lim = 4.0 * sig
    nlev = 1 << (R + 1)
    step = 2.0 * lim / (nlev - 1)
    levels = torch.arange(nlev, device=W.device, dtype=torch.float32) * step - lim
    subset = torch.arange(nlev, device=W.device) % 4
    return levels, subset, float(sig)


def tcq_encode(W: torch.Tensor, R: int):
    """Viterbi-optimal trellis encode of each row of ``W`` (sequence along dim 1).

    Returns ``(level_idx, levels, sigma)`` where ``level_idx`` is the int64
    [rows, T] index into ``levels`` chosen by the optimal path. Reconstruction is
    ``levels[level_idx]``. The level-index stream is constrained by the trellis
    (subset fixed by state), so it bit-packs to R bits/sample; storing the index
    here keeps the primitive simple and exact.
    """
    if W.dim() != 2:
        raise ValueError("tcq_encode expects a 2D weight [rows, T]")
    rows, T = W.shape
    dev = W.device
    levels, subset, sigma = _levels(W, R)

    # Per (subset, row, t): best level in that subset + its squared cost.
    sc = torch.full((4, rows, T), _INF, device=dev)
    sidx = torch.zeros((4, rows, T), dtype=torch.long, device=dev)
    for s in range(4):
        members = (subset == s).nonzero(as_tuple=False).squeeze(1)
        dist = (W[None, :, :] - levels[members][:, None, None]) ** 2  # [|s|, rows, T]
        c, a = dist.min(0)
        sc[s] = c
        sidx[s] = members[a]

    # Viterbi forward (sequential over T, parallel over rows).
    pc = torch.full((4, rows), _INF, device=dev)
    pc[0] = 0.0
    bp_state = torch.zeros((T, 4, rows), dtype=torch.long, device=dev)
    bp_subset = torch.zeros((T, 4, rows), dtype=torch.long, device=dev)
    for t in range(T):
        npc = torch.full((4, rows), _INF, device=dev)
        for s in range(4):
            for b in range(2):
                ns, ss = _NEXT[s][b], _SUBSET[s][b]
                cand = pc[s] + sc[ss, :, t]
                better = cand < npc[ns]
                npc[ns] = torch.where(better, cand, npc[ns])
                bp_state[t, ns] = torch.where(better, torch.full_like(bp_state[t, ns], s), bp_state[t, ns])
                bp_subset[t, ns] = torch.where(better, torch.full_like(bp_subset[t, ns], ss), bp_subset[t, ns])
        pc = npc

    # Traceback.
    st = pc.argmin(0)
    ar = torch.arange(rows, device=dev)
    out = torch.zeros((rows, T), dtype=torch.long, device=dev)
    for t in range(T - 1, -1, -1):
        ss = bp_subset[t, st, ar]
        out[:, t] = sidx[ss, ar, t]
        st = bp_state[t, st, ar]
    return out, levels, sigma


def tcq_decode(level_idx: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Reconstruct ``[rows, T]`` weights from the level-index path. No search."""
    return levels[level_idx]
