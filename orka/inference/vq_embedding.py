"""Row-lazy VQ embedding: serve a packed embedding table without ever materializing it.

An embedding is a LOOKUP weight - token 4712 reads row 4712 and the other 32767 rows are
untouched - so unlike a Linear it never needs to exist densely. The VQ payload (indices +
codebooks + block scales) stays resident at its packed size and each forward decodes only
the rows the current tokens ask for.

That is the whole saving: for an UNTIED checkpoint the embedding is often 20-25% of the
parameters, and at 3 bpw its resident cost drops ~5x versus the fp16 dense matrix that
``orka.integrations.hf_quantizer`` currently reconstructs at load time.

Applies only when the tensor is a pure lookup:
  * tied checkpoints are excluded - there the same matrix is also the logit projection, so
    it is a COMPUTE weight and must be dense and precise (see ``keep_head_fp16``).
  * requires ``in_features % group_size == 0`` and ``% block_size == 0`` so a row maps to a
    whole number of groups/scale-blocks; otherwise rows straddle group boundaries.
  * requires row-major indices (the layout the packer writes). ``_to_group_major`` in
    ``_vq_build`` transposes for the GEMM kernels; that layout would make a row's groups
    strided and is not used here.
  * no outlier / salient / low-rank sidecars, which are position-indexed corrections that
    would need their own per-row scatter.

``can_serve`` reports eligibility so a loader can fall back to dense instead of guessing.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn


def can_serve(tensor_meta: dict) -> tuple[bool, str]:
    """(eligible, reason). Reason is empty when eligible."""
    shape = [int(x) for x in tensor_meta["shape"]]
    if len(shape) != 2:
        return False, f"expected a 2-D table, got shape {shape}"
    dim = shape[1]
    group = int(tensor_meta.get("group_size", 8))
    block = int(tensor_meta.get("block_scale_size") or 32)
    if dim % group:
        return False, f"in_features {dim} not divisible by group_size {group}"
    if dim % block:
        return False, f"in_features {dim} not divisible by block_size {block}"
    for side in ("outliers", "salient", "lowrank", "pillars"):
        if tensor_meta.get(side):
            return False, f"{side} sidecar needs a per-row correction scatter"
    for stage in tensor_meta.get("stages", []):
        if int(stage.get("group_size", group)) != group:
            return False, "scalar/mixed-group stage layout"
    return True, ""


class VQEmbedding(nn.Module):
    """nn.Embedding-compatible module backed by a packed VQ payload.

    Buffers mirror the on-disk artifact: one index stream and one codebook per RVQ stage,
    plus the block-scale vector. Nothing here is ``[vocab, dim]``-shaped.
    """

    def __init__(self, vocab: int, dim: int, group_size: int, block_size: int,
                 codebooks: list[torch.Tensor], indices: list[torch.Tensor],
                 scales: torch.Tensor | None,
                 out_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.num_embeddings, self.embedding_dim = vocab, dim
        self.group_size, self.block_size = group_size, block_size
        # Decode accumulates in fp32 regardless (see forward); this is only the dtype handed
        # back. It must match the surrounding model - an fp16 stack will reject fp32 hidden
        # states at the first LayerNorm. Default fp32 mirrors the reference decoder so
        # parity checks compare like with like.
        self.out_dtype = out_dtype
        self.groups_per_row = dim // group_size
        self.blocks_per_row = dim // block_size
        self.n_stages = len(codebooks)
        for s, (cb, idx) in enumerate(zip(codebooks, indices)):
            self.register_buffer(f"codebook_{s}", cb)
            self.register_buffer(f"indices_{s}", idx)
        self.register_buffer("scales", scales)

    # -------------------------------------------------------------- constructor
    @classmethod
    def from_artifact(cls, artifact_dir: Path, tensor_meta: dict,
                      device: str | torch.device = "cpu",
                      out_dtype: torch.dtype = torch.float32) -> VQEmbedding:
        import numpy as np

        from orka.core._format import _read_codebook, _read_float_vector, _read_indices
        from orka.transforms.normalize import stores_block_scales

        ok, why = can_serve(tensor_meta)
        if not ok:
            raise ValueError(f"cannot serve {tensor_meta['name']} row-lazily: {why}")

        artifact_dir = Path(artifact_dir)
        vocab, dim = (int(x) for x in tensor_meta["shape"])
        group = int(tensor_meta.get("group_size", 8))
        block = int(tensor_meta.get("block_scale_size") or 32)
        total = vocab * dim

        stages = tensor_meta.get("stages") or [{
            "codebook": tensor_meta["codebook"],
            "codebook_size": int(tensor_meta["codebook_size"]),
            "index_bits": int(tensor_meta["index_bits"]),
            "indices": tensor_meta["indices"],
        }]

        cbs, idxs = [], []
        for stage in stages:
            n_groups = math.ceil(total / group)
            raw = _read_indices(
                artifact_dir / stage["indices"], int(stage["index_bits"]), n_groups,
                packed=bool(stage.get("packed", False)),
                encoding=stage.get("encoding", "raw"),
            )
            k = int(stage["codebook_size"])
            # Narrowest integer that addresses the codebook: the index stream is the bulk
            # of the resident payload, so int64 here would undo most of the saving.
            dt = torch.uint8 if k <= 256 else (torch.int16 if k <= 32768 else torch.int32)
            # .copy(): _read_indices can hand back a read-only view over the mmapped blob,
            # and torch.from_numpy refuses to own non-writable memory.
            idxs.append(torch.from_numpy(np.ascontiguousarray(raw).copy()).to(dt))
            cb = _read_codebook(artifact_dir / stage["codebook"], group,
                                stage.get("codebook_dtype", "float16"))
            cbs.append(torch.from_numpy(cb).to(torch.float16))

        scales = None
        if stores_block_scales(tensor_meta.get("normalization", "none")):
            arr = _read_float_vector(
                artifact_dir / tensor_meta["scales"], int(tensor_meta["scale_count"]),
                tensor_meta.get("scale_dtype") or "float32",
            )
            scales = torch.from_numpy(arr[: math.ceil(total / block)]).to(torch.float16)

        return cls(vocab, dim, group, block, cbs, idxs, scales,
                   out_dtype=out_dtype).to(device)

    # ------------------------------------------------------------------ forward
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).long()
        n, gpr, g = flat.numel(), self.groups_per_row, self.group_size

        # Row r owns groups [r*gpr, (r+1)*gpr) because the packer flattens row-major.
        ar = torch.arange(gpr, device=flat.device)
        gsel = flat[:, None] * gpr + ar[None, :]                      # [n, gpr]

        # Codebooks are STORED fp16 (that is the point) but accumulated in fp32, matching
        # orka.pipeline.decode._decode_tensor stage-sum order exactly so a row served here
        # is bit-comparable to the reference decoder.
        out = None
        for s in range(self.n_stages):
            idx = getattr(self, f"indices_{s}")[gsel].long()          # [n, gpr]
            part = getattr(self, f"codebook_{s}")[idx].float()        # [n, gpr, g]
            out = part if out is None else out + part
        out = out.reshape(n, self.embedding_dim)

        if self.scales is not None:
            bpr = self.blocks_per_row
            sb = torch.arange(bpr, device=flat.device)
            ssel = flat[:, None] * bpr + sb[None, :]                  # [n, bpr]
            sc = self.scales[ssel].float()
            out = (out.reshape(n, bpr, self.block_size) * sc[:, :, None]).reshape(
                n, self.embedding_dim)

        return out.to(self.out_dtype).reshape(*ids.shape, self.embedding_dim)

    # -------------------------------------------------------------- diagnostics
    def resident_bytes(self) -> dict:
        parts = {}
        for name, buf in self.named_buffers():
            if buf is not None:
                parts[name] = buf.numel() * buf.element_size()
        parts["total"] = sum(parts.values())
        parts["dense_fp16_equivalent"] = self.num_embeddings * self.embedding_dim * 2
        return parts

    def extra_repr(self) -> str:
        r = self.resident_bytes()
        return (f"{self.num_embeddings}x{self.embedding_dim}, stages={self.n_stages}, "
                f"G={self.group_size}, B={self.block_size}, "
                f"resident={r['total'] / 1e6:.1f}MB vs dense fp16 "
                f"{r['dense_fp16_equivalent'] / 1e6:.1f}MB")
