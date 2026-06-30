"""End-to-end E8-lattice model compression: HF dir -> compact artifact -> reconstruct.

A standalone codebook-free compressor built on ``orka.quant.lattice``. Each matrix-
heavy Linear is incoherence-rotated and residual-E8 quantized; only the lattice
keys (zlib-packed), the per-stage scales, and a 4-byte rotation seed are stored -
no codebook. Embeddings / norms / 1-D params are kept fp16 (negligible on large
models; the tied head dominates only on tiny ones). Reconstruction regenerates the
rotation from the seed and snaps back - no search.

Kept deliberately standalone (its own ``.lat`` artifact) rather than shoehorned
into orka's VQ-codebook ``.orka`` format + decode kernels, so the deterministic VQ
path is untouched. This is the "make the research usable" layer; format
unification + a fast decode kernel are follow-ups.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import torch

from orka.quant.lattice import E8_DIM, e8_encode, incoherence_rotation


def _output_head_modules(model) -> set:
    """The model's output-head module(s), detected STRUCTURALLY (no name match): HF's
    ``get_output_embeddings()`` by identity, plus any Linear whose ``out_features`` equals
    ``config.vocab_size`` (catches a tied / aliased head). These stay fp16 - quantizing the
    logit projection explodes perplexity. Keying on identity/shape, not the string
    'lm_head', is what makes this hold across architectures."""
    heads = set()
    try:
        oe = model.get_output_embeddings()
        if oe is not None:
            heads.add(oe)
    except Exception:
        pass
    vocab = getattr(getattr(model, "config", None), "vocab_size", None)
    if vocab:
        heads |= {m for m in model.modules()
                  if isinstance(m, torch.nn.Linear) and m.out_features == vocab}
    return heads


def _is_quantizable(name: str, module, head_modules=frozenset()) -> bool:
    """Any 2-D Linear except the output head. The old allow-list ("self_attn"/"mlp") was
    written for a standard transformer and silently skipped everything an architecture
    names differently - on a FalconH1 hybrid it covered only 9% of params (just self_attn),
    leaving feed_forward (43%) and mamba in/out_proj (35%) in fp16 and reporting a fictional
    ``avg_bpw_quantized`` over the 9%. Plain weight quant is valid for feed_forward and SSM
    matrices alike (VQ packs them too); only the output head must stay fp16.

    The head is identified structurally via ``head_modules`` (``_output_head_modules``);
    the ``is_output_head`` name test is only a fallback for when that set is unavailable."""
    from orka.quant import is_output_head

    if not isinstance(module, torch.nn.Linear) or module.weight.dim() != 2:
        return False
    if module in head_modules:
        return False
    return not is_output_head(name)


def _pack_keys(keys_per_stage) -> bytes:
    # GPU rANS entropy-codes the lattice keys to ~entropy (vs zlib's overhead).
    # Keys are small signed ints; shift to non-negative symbols, store the shift.
    import struct
    from orka.quant.ans import ans_compress

    flat = torch.cat([k.reshape(-1) for k in keys_per_stage]).to(torch.int64)
    shift = int(flat.min().item())
    blob = ans_compress((flat - shift).cpu().numpy())
    return struct.pack("<i", shift) + blob


def compress_model(model_dir: str, out_path: str, scales=(0.5, 0.2), seed: int = 1, device: str = "cuda") -> dict:
    """Compress every 2-D Linear (except the output head) of an HF model with residual-E8
    lattice - attn, mlp/feed_forward, and mamba/SSM in_proj/out_proj alike.

    ``scales`` are RELATIVE per-stage coefficients of each tensor's rotated-weight
    std (the absolute scale = coeff * std is computed and stored per tensor). This
    makes the quantizer transfer across models - a fixed absolute scale is brittle
    and exploded on a 1.5B model. Returns a manifest dict; writes ``out_path`` (the
    .lat artifact: meta json + packed payload). fp16 passthrough for everything else.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32).to(device).eval()
    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=True)

    meta = {"scales": list(scales), "seed": seed, "group_size": E8_DIM, "tensors": {}, "passthrough": []}
    payload = bytearray()
    offset = 0
    tot_bits = 0.0
    tot_w = 0
    pass_t = {}
    head_modules = _output_head_modules(model)  # structural: kept fp16 (passthrough)
    from orka.quant.lattice import input_incoherence, e8_quantize_raw, _derive_seed
    for name, mod in model.named_modules():
        if _is_quantizable(name, mod, head_modules):
            W = mod.weight.data.float()
            # seed keyed on the STORED tensor name (name+".weight") so reconstruct,
            # which iterates those keys, regenerates the identical rotation.
            Wr, _, _ = input_incoherence(W, _derive_seed(seed, name + ".weight"))
            # ADAPTIVE per-tensor scale: `scales` are relative coefficients of the
            # rotated-weight std, so the quantizer transfers across models/tensors
            # (a fixed absolute scale is brittle - it exploded on a 1.5B model).
            std = float(Wr.std().clamp(min=1e-8))
            abs_scales = [float(c) * std for c in scales]
            flat = Wr.reshape(-1)
            pad = (-flat.numel()) % E8_DIM
            if pad:
                flat = torch.cat([flat, torch.zeros(pad, device=flat.device)])
            _, keys, bpw = e8_quantize_raw(flat.reshape(-1, E8_DIM), abs_scales)
            blob = _pack_keys(keys)
            meta["tensors"][name + ".weight"] = {
                "shape": list(W.shape), "len": len(blob), "offset": offset,
                "bpw": bpw, "scales": abs_scales,
            }
            payload += blob
            offset += len(blob)
            tot_bits += bpw * W.numel()
            tot_w += W.numel()
            if mod.bias is not None:
                pass_t[name + ".bias"] = mod.bias.data.half().cpu()
    # passthrough: embeddings, norms, biases. Dedup tied weights by storage pointer
    # (e.g. tied lm_head <-> embed_tokens) - storing both doubles the embedding cost.
    seen_ptr = {}
    for n, p in model.state_dict().items():
        if n in meta["tensors"] or n in pass_t:
            continue
        ptr = p.data_ptr()
        if ptr in seen_ptr:
            meta.setdefault("aliases", {})[n] = seen_ptr[ptr]  # n shares seen_ptr[ptr]'s data
            continue
        seen_ptr[ptr] = n
        pass_t[n] = p.half().cpu()
    # write
    (out / "payload.bin").write_bytes(bytes(payload))
    from safetensors.torch import save_file
    save_file(pass_t, str(out / "passthrough.safetensors"))
    meta["avg_bpw_quantized"] = tot_bits / max(tot_w, 1)
    (out / "meta.json").write_text(json.dumps(meta))
    # real sizes
    meta["payload_bytes"] = len(payload)
    meta["passthrough_bytes"] = (out / "passthrough.safetensors").stat().st_size
    return meta


def reconstruct_state_dict(art_path: str, device: str = "cuda") -> dict:
    """Rebuild a full fp16 state dict from a .lat artifact."""
    art = Path(art_path)
    meta = json.loads((art / "meta.json").read_text())
    scales = meta["scales"]
    seed = meta["seed"]
    payload = (art / "payload.bin").read_bytes()
    from safetensors.torch import load_file
    from orka.quant.lattice import input_incoherence, inverse_incoherence, _derive_seed
    sd = {k: v.to(device) for k, v in load_file(str(art / "passthrough.safetensors")).items()}
    n_stages = len(scales)
    for name, info in meta["tensors"].items():
        shape = info["shape"]
        numel = int(np.prod(shape))
        nvec = (numel + E8_DIM - 1) // E8_DIM
        t_scales = info.get("scales", scales)  # per-tensor absolute scales
        import struct
        from orka.quant.ans import ans_decompress
        raw = payload[info["offset"]: info["offset"] + info["len"]]
        shift = struct.unpack("<i", raw[:4])[0]
        sym = ans_decompress(raw[4:], device)
        keys = (sym + shift).reshape(len(t_scales), nvec, E8_DIM)
        recon_rot = None
        for s in range(len(t_scales)):
            pts = keys[s].float() / 2.0 * t_scales[s]
            recon_rot = pts if recon_rot is None else recon_rot + pts
        # regenerate the same input-dim incoherence (signs+block) from seed+name to invert it
        Wr = recon_rot.reshape(-1)[:numel].reshape(shape)
        _, signs, bs = input_incoherence(torch.zeros(shape, device=device), _derive_seed(seed, name))
        sd[name] = inverse_incoherence(Wr, signs, bs).half()
    for alias, source in meta.get("aliases", {}).items():
        if source in sd:
            sd[alias] = sd[source]
    return sd


def reconstruct_to_hf(art_path: str, src_model_dir: str, out_dir: str, device: str = "cuda") -> str:
    """Reconstruct a .lat artifact into a loadable HF model directory (bf16)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sd = reconstruct_state_dict(art_path, device)
    model = AutoModelForCausalLM.from_pretrained(src_model_dir, local_files_only=True, dtype=torch.float16).to(device)
    model.load_state_dict({k: v.half() for k, v in sd.items()}, strict=False)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(str(out))
    try:
        AutoTokenizer.from_pretrained(src_model_dir, local_files_only=True).save_pretrained(str(out))
    except Exception:
        pass
    return str(out)
