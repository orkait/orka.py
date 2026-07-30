"""Install row-lazy :class:`VQEmbedding` into a live model in place of a dense table.

``VQEmbedding`` shipped without a caller: ``hf_quantizer._needs_dense`` still reconstructs the
full ``[vocab, dim]`` fp16 matrix at load, so the resident saving it was written for was not
reachable from anywhere. This is the loader its docstring refers to.

WHAT DECIDES ELIGIBILITY
An embedding is a lookup weight only while nothing multiplies by it. The gate is therefore
structural and follows the module you hand in, not a config flag:

    model.get_output_embeddings() is not None  ->  the table is ALSO a logit projection
                                                   (a compute weight) -> refuse
    no output embeddings                       ->  pure lookup -> serve row-lazily

That distinction matters for tied checkpoints. LFM2.5-Encoder-230M sets
``tie_word_embeddings=True`` and ties ``lm_head.weight`` to ``lfm2.embed_tokens.weight``, but
its stated use (classification / retrieval / reranking) discards the MLM head. Pass the
``Lfm2BidirectionalForMaskedLM`` and this refuses; pass its ``.lfm2`` encoder and it serves.
Reading ``tie_word_embeddings`` off the config instead would refuse both, including the one
deployment where the saving is real and safe.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from orka.inference.vq_embedding import VQEmbedding, can_serve


def _qualified_name(model: nn.Module, target: nn.Module) -> str | None:
    """Dotted path of ``target`` inside ``model``, matched by identity not by name."""
    for name, mod in model.named_modules():
        if mod is target:
            return name
    return None


def _parent_of(model: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True when two parameters are the same matrix - the tie test.

    ``is`` alone misses a tie re-established after a ``.to()`` copy, so fall back to comparing
    storage identity.
    """
    if a is b:
        return True
    return (a.shape == b.shape
            and a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr())


def is_pure_lookup(model: nn.Module) -> tuple[bool, str]:
    """(eligible, reason) for the input embedding of ``model``. Reason empty when eligible."""
    get_in = getattr(model, "get_input_embeddings", None)
    if get_in is None:
        return False, "model exposes no get_input_embeddings()"
    try:
        emb = get_in()
    except (AttributeError, NotImplementedError) as exc:
        return False, f"get_input_embeddings() failed: {type(exc).__name__}"
    if emb is None:
        return False, "model has no input embedding"

    get_out = getattr(model, "get_output_embeddings", None)
    out = None
    if get_out is not None:
        try:
            out = get_out()
        except (AttributeError, NotImplementedError):
            out = None
    if out is not None and getattr(out, "weight", None) is not None:
        if _shares_storage(out.weight, emb.weight):
            return False, ("tied to the output projection - the table is also a compute "
                           "weight here; pass the base encoder instead")
        return True, ""
    return True, ""


def install_vq_embeddings(
    model: nn.Module,
    artifact_dir: str | Path,
    *,
    manifest: dict | None = None,
    out_dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    allow_tied: bool = False,
) -> dict:
    """Swap the model's dense input embedding for a row-lazy :class:`VQEmbedding`.

    Returns a report: ``installed`` (bool), ``reason`` when it did not, and resident byte
    counts so a caller can assert the saving rather than trust it.
    """
    artifact_dir = Path(artifact_dir)
    if manifest is None:
        manifest = json.loads((artifact_dir / "manifest.json").read_text())

    report: dict = {"installed": False, "reason": "", "tensor": None}

    eligible, why = is_pure_lookup(model)
    if not eligible and not allow_tied:
        report["reason"] = why
        return report
    if not eligible:
        report["forced_tied"] = why

    emb = model.get_input_embeddings()
    dotted = _qualified_name(model, emb)
    if dotted is None:
        report["reason"] = "input embedding module is not reachable from this model"
        return report
    weight_name = f"{dotted}.weight"

    # The artifact names tensors from the CHECKPOINT root ("lfm2.embed_tokens.weight"), but
    # the eligible deployment is often a submodule ("embed_tokens.weight" when the encoder
    # itself is handed in) - that is precisely the lookup-only case this loader exists for.
    # Fall back to a unique suffix match; refuse an ambiguous one rather than guess.
    tensors = manifest.get("tensors", [])
    tm = next((t for t in tensors if t["name"] == weight_name), None)
    if tm is None:
        suffix = [t for t in tensors if t["name"].endswith("." + weight_name)]
        if len(suffix) == 1:
            tm = suffix[0]
        elif len(suffix) > 1:
            report["reason"] = (f"{weight_name} matches {len(suffix)} artifact tensors "
                                f"({', '.join(t['name'] for t in suffix[:3])}); ambiguous")
            return report
    if tm is None:
        report["reason"] = f"{weight_name} is not quantized in this artifact"
        return report

    ok, why = can_serve(tm)
    if not ok:
        report["reason"] = why
        report["tensor"] = weight_name
        return report

    if out_dtype is None:
        out_dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()),
                         torch.float32)
    if device is None:
        device = next((p.device for p in model.parameters()), torch.device("cpu"))

    dense_bytes = emb.weight.numel() * emb.weight.element_size()
    vq = VQEmbedding.from_artifact(artifact_dir, tm, device=device, out_dtype=out_dtype)
    parent, attr = _parent_of(model, dotted)
    setattr(parent, attr, vq)

    resident = vq.resident_bytes()
    report.update({
        "installed": True,
        # the ARTIFACT tensor actually served - with suffix matching this can differ from
        # the module-side path, and reporting the latter would hide which one was picked
        "tensor": tm["name"],
        "module": weight_name,
        "dense_bytes": dense_bytes,
        "resident_bytes": resident["total"],
        "saved_bytes": dense_bytes - resident["total"],
        "ratio": dense_bytes / max(resident["total"], 1),
        "breakdown": resident,
    })
    return report
