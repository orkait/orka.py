"""Quantization-aware training for Orka VQ (prototype).

Orka is post-training: it fits codebooks to fixed weights. At low bit budgets
(<=2 effective bpw) PTQ collapses because pretrained weights were never shaped
for a coarse codebook. QAT closes that gap - the weights are fine-tuned WITH
the quantizer in the forward pass so the network learns to be robust to its
own quantization (the AQLM / QuIP# recipe; VQ-VAE supplies the differentiable
machinery for the hard argmin).

This module keeps Orka's identity intact: the bit budget is the per-tensor
MEASURED allocation (water-filling), not a uniform width. ``QATVQLinear`` wraps
one quantized linear; ``build_qat_student`` wires a whole HF model.

Design (VQ-VAE straight-through):
  shadow weight W  (trainable)         codebooks e_s  (trainable, per stage)
  forward:  W_q = W + (decode(W) - W).detach()   -> network sees decode(W),
            gradient to W is identity (commitment path).
  codebook loss: sum_s || sg[residual_s] - e_s[idx_s] ||^2  pulls codebooks to
            the residual they quantize.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Cap the transient [chunk, k] distance matrix at ~this many float32 elements
# (~1 GB at 2.5e8). The row-chunk shrinks as the codebook k grows, so a dynamic /
# larger codebook never blows VRAM.
_ASSIGN_BUDGET = 256 * 1024 * 1024


def _chunked_assign(vectors: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
    """Nearest-codebook index. Delegates to the fused dist+argmin Triton kernel on
    CUDA (~4x the chunked-addmm path, and never materializes the [N,k] matrix so it
    can't OOM at large k); falls back to the addmm path on CPU / no-Triton.

    Distance is ||c||^2 - 2 v.c (||v||^2 is constant across centroids, drops out of
    the argmin) - identical result to torch.cdist, verified 0-mismatch vs addmm."""
    from orka.inference._assign_kernel import assign
    return assign(vectors, cb)


def _kmeans_init(vectors: torch.Tensor, k: int) -> torch.Tensor:
    """Codebook seed = random sample of k vectors (kept for the Lloyd seed)."""
    n = vectors.shape[0]
    k = min(k, n)
    perm = torch.randperm(n, device=vectors.device)[:k]
    return vectors[perm].clone()


def _kmeans_fit(vectors: torch.Tensor, k: int, iters: int = 12) -> torch.Tensor:
    """Lloyd k-means codebook. A real fit (not a random sample) is what gives QAT
    a PTQ-quality warm start; from random seeds the 4096-centroid assignment is
    ~40% off and 150 steps cannot recover it. ORKA_KMEANS_ITERS overrides the
    iteration count (fewer = faster warm-start for quick validation)."""
    import os
    iters = int(os.environ.get("ORKA_KMEANS_ITERS", iters))
    cb = _kmeans_init(vectors, k)
    for _ in range(iters):
        assign = _chunked_assign(vectors, cb)
        new = torch.zeros_like(cb)
        new.index_add_(0, assign, vectors)                       # sum vectors per centroid
        cnt = torch.bincount(assign, minlength=cb.shape[0]).to(new.dtype)  # count, no N-sized ones temp
        mask = cnt > 0
        cb[mask] = new[mask] / cnt[mask].unsqueeze(1)
    return cb


def _pick_block_size(in_features: int, group_size: int, cap: int = 256) -> int:
    """Largest divisor of in_features that is <= cap and a multiple of group_size.
    Per-block scales normalize magnitude so the codebook only encodes shape; a
    block near 256 keeps the scale overhead tiny (16/block bpw)."""
    best = group_size
    b = group_size
    while b <= cap:
        if in_features % b == 0 and b % group_size == 0:
            best = b
        b += group_size
    return best


class QATVQLinear(nn.Module):
    """A linear whose weight is residual-VQ quantized on every forward, with
    straight-through gradients to a trainable shadow weight and learnable
    codebooks. ``stages`` is the per-tensor codebook-size list from the Orka
    allocation map (e.g. [4096, 16] = rvq-12-4 = 2 bpw at group 8)."""

    def __init__(self, weight: torch.Tensor, bias, group_size: int, stages: list[int],
                 commitment: float = 0.25, checkpoint: bool = False,
                 reassign_every: int = 1):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.group_size = group_size
        self.stages = stages
        self.commitment = commitment
        # Cached nearest-codebook assignment. The argmin (cdist) is the entire QAT
        # forward cost (~4s/step across all linears); but the shadow weight only
        # moves on opt.step (in-place add -> .data._version bumps), and the
        # assignment churns ~0.3%/step. So re-search only every `reassign_every`
        # OPTIMIZER steps and reuse the cached idx in between (codebooks are still
        # indexed with the current cb, so cb gradient is unaffected - only the hard
        # argmin is stale). reassign_every=1 still skips the redundant re-search
        # within a grad-accum window (shadow unchanged there): exact, free speedup.
        self.reassign_every = reassign_every
        self._cached_idx: list[torch.Tensor] | None = None
        self._last_version = -1
        self._optsteps_since = 0
        # Gradient-checkpoint quantize() on the forward: the straight-through
        # decode keeps weight-sized fp32 intermediates (sel/decoded) alive for
        # backward, ~8GB across all layers. Checkpointing frees them on forward
        # and recomputes (cheap: cdist assign + gather) in backward. Bit-identical.
        self.checkpoint = checkpoint
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

        self.shadow = nn.Parameter(weight.detach().clone().float())

        # Per-block scales normalize magnitude so the codebook only encodes SHAPE.
        # Without this the single codebook must span the full weight dynamic range
        # -> ~40% recon error even with a perfect fit (this was the QAT-broken bug).
        self.block_size = _pick_block_size(self.in_features, group_size)
        n_blocks = self.in_features // self.block_size
        W = self.shadow.detach()
        sc = W.reshape(self.out_features, n_blocks, self.block_size).abs().amax(-1).clamp_min(1e-8)
        self.scales = nn.Parameter(sc)                                  # [out, n_blocks], trainable

        # Codebooks fit (Lloyd k-means) on the NORMALIZED residual vectors -> a
        # PTQ-quality warm start. Single large codebook is more bit-efficient than
        # multi-stage small ones at equal bpw, so `stages` is usually one big k.
        sc_full = sc.repeat_interleave(self.block_size, dim=1)          # [out, in]
        Wn = (W / sc_full).reshape(-1, group_size)
        residual = Wn.clone()
        cbs = []
        for k in stages:
            cb = _kmeans_fit(residual, int(k))
            assign = _chunked_assign(residual, cb)
            residual = residual - cb[assign]
            cbs.append(nn.Parameter(cb.clone()))
        self.codebooks = nn.ParameterList(cbs)
        self._last_cb_loss = torch.zeros((), device=weight.device)

    def _want_refresh(self) -> bool:
        """Re-run the hard argmin only every `reassign_every` optimizer steps. An
        opt.step mutates shadow in place -> .data._version bumps; a grad-accum
        window repeats the same version across forwards, so count an opt-step once
        per genuine version CHANGE (not per forward). Between refreshes the cached
        idx is reused (codebooks are still indexed live, so cb gradient is exact -
        only the hard argmin is stale, and assignment churns ~0.3%/opt-step)."""
        if self._cached_idx is None:
            return True
        v = self.shadow._version
        if v != self._last_version:        # genuinely new opt-step, not a repeat forward
            self._last_version = v
            self._optsteps_since += 1
        return self._optsteps_since >= self.reassign_every

    def _quantize_impl(self, shadow: torch.Tensor, scales: torch.Tensor, *codebooks: torch.Tensor):
        """Returns (w_q, cb_loss). Pure function of (shadow, scales, codebooks) so
        it can be gradient-checkpointed - cb_loss is a real graph output, not a
        side effect, so the codebook gradient survives the recompute.

        Decode mirrors Orka: normalize the weight by per-block scale, residual-VQ
        the normalized vectors, then re-scale. Scales and codebooks both train."""
        # Per-block scale via broadcast over a [out, n_blocks, block] view - avoids
        # repeat_interleave materializing a full [out, in] scale temp each forward.
        n_blocks = self.in_features // self.block_size
        sc3 = scales[:, :, None]                                          # [out, n_blocks, 1]
        vn = (shadow.reshape(self.out_features, n_blocks, self.block_size) / sc3
              ).reshape(-1, self.group_size)                             # normalized vectors
        refresh = self._want_refresh()
        if refresh:
            self._cached_idx = [None] * len(codebooks)
            self._last_version = self.shadow._version
            self._optsteps_since = 0
        residual = vn
        decoded = torch.zeros_like(vn)
        cb_loss = torch.zeros((), device=vn.device)
        for s, cb in enumerate(codebooks):
            if refresh:
                with torch.no_grad():
                    self._cached_idx[s] = _chunked_assign(residual.detach(), cb.detach())
            assign = self._cached_idx[s]
            sel = cb[assign]                       # differentiable in cb (idx may be cached)
            cb_loss = cb_loss + F.mse_loss(sel, residual.detach())
            decoded = decoded + sel
            residual = vn - decoded
        dec = (decoded.reshape(self.out_features, n_blocks, self.block_size) * sc3
               ).reshape(self.out_features, self.in_features)            # de-normalize
        # straight-through: forward uses dec (quantized weight), grad to shadow is identity
        w_q = shadow + (dec - shadow).detach()
        return w_q, cb_loss

    def quantize(self) -> torch.Tensor:
        w_q, cb_loss = self._quantize_impl(self.shadow, self.scales, *self.codebooks)
        self._last_cb_loss = cb_loss
        return w_q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpoint and self.training:
            import torch.utils.checkpoint as cp
            w_q, cb_loss = cp.checkpoint(
                self._quantize_impl, self.shadow, self.scales, *self.codebooks,
                use_reentrant=False,
            )
            self._last_cb_loss = cb_loss
        else:
            w_q = self.quantize()
        return F.linear(x, w_q.to(x.dtype), self.bias)

    @torch.no_grad()
    def materialized_weight(self) -> torch.Tensor:
        """The actual quantized weight (scaled decode) for export / eval. Mirrors
        _quantize_impl exactly so the saved weight == what the forward optimized."""
        n_blocks = self.in_features // self.block_size
        sc3 = self.scales[:, :, None]
        vn = (self.shadow.reshape(self.out_features, n_blocks, self.block_size) / sc3
              ).reshape(-1, self.group_size)
        residual = vn
        decoded = torch.zeros_like(vn)
        for cb in self.codebooks:
            assign = _chunked_assign(residual, cb)
            decoded = decoded + cb[assign]
            residual = vn - decoded
        return (decoded.reshape(self.out_features, n_blocks, self.block_size) * sc3
                ).reshape(self.out_features, self.in_features)


def build_qat_student(model: nn.Module, allocation: dict, group_size: int = 8,
                      commitment: float = 0.25, checkpoint: bool = False,
                      reassign_every: int = 1) -> dict:
    """Replace each allocated linear in ``model`` with a QATVQLinear using its
    per-tensor stage list. Embeddings / norms / lm_head stay fp16 (sensitive,
    same as Orka's passthrough). Returns {module_name: QATVQLinear}."""
    tensor_stages = {
        name: entry["stages"] for name, entry in allocation.get("tensors", {}).items()
    }
    wrapped = {}
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        wname = full_name + ".weight"
        if wname not in tensor_stages:
            continue
        if module.in_features % group_size != 0:
            continue
        stages = [int(k) for k in tensor_stages[wname] if not (isinstance(k, str) and k.startswith("s"))]
        if not stages:
            continue
        qat = QATVQLinear(module.weight.data, module.bias.data if module.bias is not None else None,
                          group_size, stages, commitment, checkpoint=checkpoint,
                          reassign_every=reassign_every).to(module.weight.device)
        parent = model.get_submodule(full_name.rsplit(".", 1)[0])
        setattr(parent, full_name.rsplit(".", 1)[-1], qat)
        wrapped[full_name] = qat
    return wrapped


def collect_codebook_loss(wrapped: dict) -> torch.Tensor:
    total = None
    for qat in wrapped.values():
        l = qat._last_cb_loss
        total = l if total is None else total + l
    return total if total is not None else torch.zeros(())
