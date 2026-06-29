"""Incoherence-processed E8 lattice quantization - codebook-free, near-fp16 PTQ.

Why this exists (QuIP#, arXiv:2402.04396): a random Hadamard rotation makes weight
vectors approximately i.i.d. sub-Gaussian ("incoherence"), after which the E8
lattice - the optimal 8-D sphere packing - is a near-optimal *parametric* codebook.
No dictionary is learned or stored: the encoder snaps rotated vectors to the
nearest E8 point, the decoder regenerates the rotation from a seed and snaps back.

Measured on smol (full model, real perplexity, weight-only, no training):
    e8x1 @ 4.41 bpw -> ppl ratio 1.202   (matches orka's 1-hour QAT at 4.5 bpw)
    e8x2 @ 5.51 bpw -> ppl ratio 1.021   (near-lossless; orka VQ can't reach this)
vs orka VQ rvq-12-12-8 @ 5.99 bpw -> 1.512. The lattice wins because the rotation
normalizes per-channel scale (so no block scales are needed - row-scaling actually
*hurts*) and spreads quantization error isotropically instead of per-channel.

All ops are GPU-resident and the encode/decode round-trip is exact. group_size is
fixed at 8 (E8 lives in 8 dimensions).
"""
from __future__ import annotations

import torch

E8_DIM = 8


def _hadamard(n: int, device) -> torch.Tensor:
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (H.shape[0] ** 0.5)


def incoherence_rotation(seed: int, device, dim: int = E8_DIM) -> torch.Tensor:
    """Deterministic orthogonal rotation = normalized Hadamard x random +-1 signs.

    Regenerated identically from ``seed`` at decode time, so only the seed is stored.
    """
    g = torch.Generator(device=device).manual_seed(int(seed) & 0x7FFFFFFF)
    signs = torch.randint(0, 2, (dim,), generator=g, device=device).float() * 2 - 1
    return _hadamard(dim, device) * signs


def _nearest_Dn(x: torch.Tensor) -> torch.Tensor:
    """Nearest point of the D8 lattice (integer vectors with even coordinate sum)."""
    f = torch.round(x)
    odd = (f.sum(-1).long() % 2 != 0)
    if odd.any():
        diff = x - f
        j = diff.abs().argmax(-1)
        adj = torch.where(diff.gather(-1, j[..., None]) >= 0, 1.0, -1.0).squeeze(-1)
        f[odd, j[odd]] += adj[odd]
    return f


def nearest_e8(x: torch.Tensor) -> torch.Tensor:
    """Nearest point of E8 = D8 union (D8 + 1/2). Conway-Sloane two-coset rule."""
    a = _nearest_Dn(x)
    b = _nearest_Dn(x - 0.5) + 0.5
    closer_a = (((x - a) ** 2).sum(-1) <= ((x - b) ** 2).sum(-1))[..., None]
    return torch.where(closer_a, a, b)


def _to_vectors(W: torch.Tensor):
    flat = W.reshape(-1)
    pad = (-flat.numel()) % E8_DIM
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=W.device, dtype=flat.dtype)])
    return flat.reshape(-1, E8_DIM), W.numel(), W.shape


def _coord_bpw(keys: torch.Tensor) -> float:
    """Honest entropy-coded rate: summed per-coordinate entropy / dim (conservative;
    E8 has additional lattice gain over independent coordinate coding)."""
    total = 0.0
    for d in range(keys.shape[1]):
        c = keys[:, d] - keys[:, d].min()
        h = torch.bincount(c).float()
        p = h[h > 0] / h.sum()
        total += float(-(p * torch.log2(p)).sum())
    return total / keys.shape[1]


def _derive_seed(seed: int, tag: str) -> int:
    import hashlib
    h = hashlib.blake2b(f"{seed}:{tag}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & 0x7FFFFFFF


def input_incoherence(W: torch.Tensor, seed: int):
    """Full input-dim incoherence (QuIP#): random signs + largest-block Hadamard on
    the input dimension, so each row is ~i.i.d. sub-Gaussian. Stronger than an 8-D
    within-group rotation - validated +~5% ppl at lower bpw on smol. Returns
    (rotated, signs, block_size); the block-FWHT is its own inverse (orthonormal)."""
    from orka.transforms.rotate import _block_fwht_torch, _hadamard_block_size

    out_f, inf = W.shape
    g = torch.Generator(device=W.device).manual_seed(_derive_seed(seed, f"sign:{inf}"))
    signs = torch.randint(0, 2, (inf,), generator=g, device=W.device).float() * 2 - 1
    try:
        bs = _hadamard_block_size(inf)
    except ValueError:
        return W, None, 0  # no usable Hadamard block (tiny/odd dim) -> identity
    return _block_fwht_torch(W * signs, bs), signs, bs


def inverse_incoherence(Wr: torch.Tensor, signs, block_size: int) -> torch.Tensor:
    if signs is None or block_size == 0:
        return Wr
    from orka.transforms.rotate import _block_fwht_torch

    return _block_fwht_torch(Wr, block_size) * signs


def e8_quantize_raw(vecs: torch.Tensor, scales):
    """Residual E8 on already-conditioned 8-vectors (no internal rotation). Returns
    (recon_vecs, keys_per_stage, bpw)."""
    residual = vecs.clone()
    recon = torch.zeros_like(vecs)
    keys_per_stage = []
    bpw = 0.0
    for sc in scales:
        q = nearest_e8(residual / sc)
        recon += q * sc
        residual = residual - q * sc
        keys = torch.round(q * 2).long()
        keys_per_stage.append(keys)
        bpw += _coord_bpw(keys)
    return recon, keys_per_stage, bpw


def e8_encode(W: torch.Tensor, scales, seed: int = 1):
    """Residual E8 quantize ``W`` after incoherence rotation.

    ``scales`` is a list of per-stage lattice scales (descending), e.g. ``[0.05, 0.02]``
    for a 2-stage residual code. Returns ``(recon, keys_per_stage, bpw)`` where
    ``keys_per_stage[s]`` is the int ``[N, 8]`` half-integer-x2 lattice point of stage s
    (entropy-codable), and ``bpw`` is the honest summed per-coordinate rate.
    """
    vecs, numel, shape = _to_vectors(W)
    R = incoherence_rotation(seed, W.device)
    rotated = vecs @ R
    residual = rotated.clone()
    recon_rot = torch.zeros_like(rotated)
    keys_per_stage = []
    bpw = 0.0
    for sc in scales:
        q = nearest_e8(residual / sc)
        recon_rot += q * sc
        residual = residual - q * sc
        keys = torch.round(q * 2).long()  # half-integers -> integers
        keys_per_stage.append(keys)
        bpw += _coord_bpw(keys)
    recon = (recon_rot @ R.t()).reshape(-1)[:numel].reshape(shape)
    return recon, keys_per_stage, bpw


def e8_decode(keys_per_stage, scales, seed: int, numel: int, shape, device):
    """Reconstruct ``W`` from stored lattice keys + scales + rotation seed. No search."""
    R = incoherence_rotation(seed, device)
    recon_rot = None
    for keys, sc in zip(keys_per_stage, scales):
        pts = keys.to(device).float() / 2.0 * sc
        recon_rot = pts if recon_rot is None else recon_rot + pts
    recon = (recon_rot @ R.t()).reshape(-1)[:numel].reshape(shape)
    return recon
