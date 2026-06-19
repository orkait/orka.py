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


_ASSIGN_CHUNK = 16384


def _chunked_assign(vectors: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
    """Nearest-codebook index, memory-bounded. cdist over all vectors x k would
    blow VRAM (millions of vectors x thousands of centroids); chunk the rows."""
    out = torch.empty(vectors.shape[0], dtype=torch.long, device=vectors.device)
    for i in range(0, vectors.shape[0], _ASSIGN_CHUNK):
        chunk = vectors[i : i + _ASSIGN_CHUNK]
        out[i : i + _ASSIGN_CHUNK] = torch.cdist(chunk, cb).argmin(dim=1)
    return out


def _kmeans_init(vectors: torch.Tensor, k: int) -> torch.Tensor:
    """Codebook seed = random sample of k vectors. Codebooks are learnable and
    refined by training, so a heavier k-means init is not worth its cost."""
    n = vectors.shape[0]
    k = min(k, n)
    perm = torch.randperm(n, device=vectors.device)[:k]
    return vectors[perm].clone()


class QATVQLinear(nn.Module):
    """A linear whose weight is residual-VQ quantized on every forward, with
    straight-through gradients to a trainable shadow weight and learnable
    codebooks. ``stages`` is the per-tensor codebook-size list from the Orka
    allocation map (e.g. [4096, 16] = rvq-12-4 = 2 bpw at group 8)."""

    def __init__(self, weight: torch.Tensor, bias, group_size: int, stages: list[int],
                 commitment: float = 0.25, checkpoint: bool = False):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.group_size = group_size
        self.stages = stages
        self.commitment = commitment
        # Gradient-checkpoint quantize() on the forward: the straight-through
        # decode keeps weight-sized fp32 intermediates (sel/decoded) alive for
        # backward, ~8GB across all layers. Checkpointing frees them on forward
        # and recomputes (cheap: cdist assign + gather) in backward. Bit-identical.
        self.checkpoint = checkpoint
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

        self.shadow = nn.Parameter(weight.detach().clone().float())

        vecs = self.shadow.detach().reshape(-1, group_size)
        residual = vecs.clone()
        cbs = []
        for k in stages:
            cb = _kmeans_init(residual, int(k))
            assign = _chunked_assign(residual, cb)
            residual = residual - cb[assign]
            cbs.append(nn.Parameter(cb.clone()))
        self.codebooks = nn.ParameterList(cbs)
        self._last_cb_loss = torch.zeros((), device=weight.device)

    def _quantize_impl(self, shadow: torch.Tensor, *codebooks: torch.Tensor):
        """Returns (w_q, cb_loss). Pure function of (shadow, codebooks) so it can
        be gradient-checkpointed - cb_loss is a real graph output, not a side
        effect, so the codebook gradient survives the recompute."""
        vecs = shadow.reshape(-1, self.group_size)
        residual = vecs
        decoded = torch.zeros_like(vecs)
        cb_loss = torch.zeros((), device=vecs.device)
        for cb in codebooks:
            with torch.no_grad():
                assign = _chunked_assign(residual.detach(), cb.detach())
            sel = cb[assign]                       # differentiable in cb
            cb_loss = cb_loss + F.mse_loss(sel, residual.detach())
            decoded = decoded + sel
            residual = vecs - decoded
        # straight-through: forward uses decoded, gradient to shadow is identity
        w_q = vecs + (decoded - vecs).detach()
        return w_q.reshape(self.out_features, self.in_features), cb_loss

    def quantize(self) -> torch.Tensor:
        w_q, cb_loss = self._quantize_impl(self.shadow, *self.codebooks)
        self._last_cb_loss = cb_loss
        return w_q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpoint and self.training:
            import torch.utils.checkpoint as cp
            w_q, cb_loss = cp.checkpoint(
                self._quantize_impl, self.shadow, *self.codebooks, use_reentrant=False
            )
            self._last_cb_loss = cb_loss
        else:
            w_q = self.quantize()
        return F.linear(x, w_q.to(x.dtype), self.bias)

    @torch.no_grad()
    def materialized_weight(self) -> torch.Tensor:
        """The actual quantized weight (decoded) for export / eval."""
        vecs = self.shadow.reshape(-1, self.group_size)
        residual = vecs
        decoded = torch.zeros_like(vecs)
        for cb in self.codebooks:
            assign = _chunked_assign(residual, cb)
            decoded = decoded + cb[assign]
            residual = vecs - decoded
        return decoded.reshape(self.out_features, self.in_features)


def build_qat_student(model: nn.Module, allocation: dict, group_size: int = 8,
                      commitment: float = 0.25, checkpoint: bool = False) -> dict:
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
                          group_size, stages, commitment, checkpoint=checkpoint).to(module.weight.device)
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
