"""GPTQ-style error-compensated VQ assignment (block OBS, GPTVQ-lite).

Column groups are quantized left to right; after committing a group's
quantization error E, the not-yet-quantized columns absorb the optimal
compensation  W[:, rest] -= E @ Hinv_bb^-1 @ Hinv_b,rest  (the block
Optimal Brain Surgeon update), where H = X^T X / n is the calibration input
covariance. Later groups are then quantized against the compensated weights,
so errors cancel in OUTPUT space instead of accumulating.

Codebooks are frozen inputs (learned beforehand, e.g. weighted k-means);
this module only re-derives the index assignment. Requires the vector
layout to be G consecutive columns per row (Orka's row-major grouping) and
unrotated weights (rotation mixes columns out of H's space).
"""

from __future__ import annotations


def _input_covariance(X, damp: float = 0.01):
    """Damped H = X^T X / n with GPTQ dead-column handling."""
    import torch

    X = X.to(dtype=torch.float32)
    H = (X.T @ X) / max(1, int(X.shape[0]))
    diag = torch.diagonal(H)
    dead = diag == 0
    if bool(dead.any()):
        diag[dead] = 1.0
    diag += damp * diag.mean()
    return H


def compensated_assign(
    W,
    codebooks,
    group_size: int,
    X,
    damp: float = 0.01,
):
    """Multi-stage VQ assignment with block-OBS error compensation.

    W: [rows, cols] float32 tensor (cols % group_size == 0).
    codebooks: list of [k_s, group_size] stage codebooks (frozen).
    X: [n, cols] calibration activations.
    Returns (stage_indices, decoded) where stage_indices is a list of
    [rows * groups_per_row] int64 tensors in Orka's row-major vector order
    and decoded is the reconstructed [rows, cols] weight matrix.
    """
    import torch

    from orka.codebook import quantize_vectors_auto

    rows, cols = int(W.shape[0]), int(W.shape[1])
    if cols % group_size != 0:
        raise ValueError("cols must be divisible by group_size")
    gpr = cols // group_size
    device = W.device

    H = _input_covariance(X.to(device), damp=damp)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))

    Wc = W.clone()
    decoded = torch.zeros_like(Wc)
    stage_indices = [
        torch.empty(rows, gpr, dtype=torch.int64, device=device) for _ in codebooks
    ]
    # Codebooks are frozen, so the device transfer hoists out of the per-group loop.
    codebooks_dev = [cb.to(device) for cb in codebooks]

    for g in range(gpr):
        a, b = g * group_size, (g + 1) * group_size
        target = Wc[:, a:b].contiguous()
        dec = torch.zeros_like(target)
        residual = target
        for s, cb_dev in enumerate(codebooks_dev):
            idx, _ = quantize_vectors_auto(residual, cb_dev, "torch", str(device))
            idx = idx.to(device)
            stage_indices[s][:, g] = idx
            dec = dec + cb_dev[idx]
            residual = target - dec
        decoded[:, a:b] = dec
        if b < cols:
            M = torch.linalg.solve(Hinv[a:b, a:b], Hinv[a:b, b:])
            Wc[:, b:] -= residual @ M    # residual == target - dec (the committed error)

    return [si.reshape(-1) for si in stage_indices], decoded
