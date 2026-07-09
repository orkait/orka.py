"""Convert an .orka artifact into a GGUF carrying the RVQ data: data format +
dequant reference, decoupled from llama.cpp's model graph.

Each quantized linear is stored as separate GGUF tensors - per-stage indices (I16),
per-stage codebooks (F16), and block scales (F16) - plus per-linear metadata in the GGUF
KV store (group_size, block_size, n_stages, in/out features). Embeddings + passthrough are
stored dense F16. This is correctness-first; the bit-plane index packing (#84) is a later
storage optimization.

``dequant_linear`` reconstructs W [out, in] from the stored tensors using the RVQ rule
(W = sum_s codebook_s[idx_s] * block_scale), the reference the C/CUDA kernels must match.
``validate`` checks it equals ``VQLinear.reconstruct_weight`` for a round-trip gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ORKA_META_PREFIX = "orka.linear."


def _quantized_linears(manifest: dict):
    for tm in manifest.get("tensors", []):
        name = tm["name"]
        if name.endswith(".weight") and "embed" not in name and "lm_head" not in name:
            yield tm


def dequant_linear(idx_stages, codebooks, scales, M, K, group_size, block_size, group_major=False):
    """Reconstruct W [M, K] fp32 from stored RVQ tensors. Reference for the GGML kernels.

    idx_stages: list of int arrays, row-major [M*GPR] or group-major [GPR*M] per
    ``group_major``. codebooks: list of [cb, G] f32. scales: [M*BPR] / [BPR*M] f32.
    """
    GPR = K // group_size
    BPR = K // block_size
    W = np.zeros((M, GPR * group_size), dtype=np.float32)
    for idx, cb in zip(idx_stages, codebooks):
        i = idx.reshape(GPR, M).T if group_major else idx.reshape(M, GPR)   # [M, GPR]
        W += cb[i].reshape(M, GPR * group_size)
    sc = scales.reshape(BPR, M).T if group_major else scales.reshape(M, BPR)
    W = (W.reshape(M, BPR, block_size) * sc[:, :, None]).reshape(M, GPR * group_size)
    return W[:, :K]


def export_gguf(artifact_dir, config_dir, out_path) -> dict:
    """Write a GGUF with the orka RVQ data. Returns a summary."""
    from gguf import GGUFWriter

    from orka.inference.vq_linear import build_vq_linear
    from orka.pipeline.decode import _decode_tensor

    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    writer = GGUFWriter(str(out_path), "orka-rvq")

    n_quant = 0
    for tm in manifest.get("tensors", []):
        name = tm["name"]
        shape = [int(x) for x in tm["shape"]]
        is_linear = name.endswith(".weight") and "embed" not in name and "lm_head" not in name and len(shape) == 2
        if not is_linear:
            arr = np.asarray(_decode_tensor(artifact_dir, tm), dtype=np.float32).reshape(shape)
            writer.add_tensor(name, arr.astype(np.float16))
            continue

        layer = build_vq_linear(artifact_dir, tm, bias=None, device="cpu")
        M, K = layer.out_features, layer.in_features
        for s in range(layer.n_stages):
            idx = layer._stage_indices_int(s).cpu().numpy().astype(np.int16)
            writer.add_tensor(f"{name}.idx{s}", idx)
            writer.add_tensor(f"{name}.cb{s}", getattr(layer, f"codebook_{s}").cpu().numpy().astype(np.float16))
        writer.add_tensor(f"{name}.scales", layer.scales.cpu().numpy().astype(np.float16))
        meta = ORKA_META_PREFIX + name + "."
        writer.add_uint32(meta + "out_features", M)
        writer.add_uint32(meta + "in_features", K)
        writer.add_uint32(meta + "group_size", layer.group_size)
        writer.add_uint32(meta + "block_size", layer.block_size)
        writer.add_uint32(meta + "n_stages", layer.n_stages)
        writer.add_uint32(meta + "group_major", int(bool(getattr(layer, "_group_major", False))))
        n_quant += 1

    # Passthrough tensors (layernorms, biases, rotary inv_freq) - needed for a full forward.
    n_pass = 0
    passthrough = artifact_dir / "passthrough.safetensors"
    if passthrough.exists():
        from safetensors import safe_open
        with safe_open(str(passthrough), "pt") as f:
            for k in f.keys():
                if k.endswith(".attention.bias") or k.endswith(".masked_bias"):
                    continue  # causal-mask buffers, rebuilt at runtime
                writer.add_tensor(k, f.get_tensor(k).float().numpy().astype(np.float32))
                n_pass += 1

    # Architecture hyperparameters from config.json (for the runner to build the graph).
    cfg_path = Path(config_dir) / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        def hp(*names, default=None):
            for n in names:
                if n in cfg:
                    return cfg[n]
            return default
        kv = {
            "n_layer": hp("n_layer", "num_hidden_layers"),
            "n_head": hp("n_head", "num_attention_heads"),
            "n_embd": hp("n_embd", "hidden_size"),
            "n_ff": hp("n_ff", "intermediate_size"),
            "n_vocab": hp("n_vocab", "vocab_size"),
            "n_ctx": hp("n_ctx", "max_position_embeddings", default=2048),
        }
        for k, v in kv.items():
            if v is not None:
                writer.add_uint32(ORKA_META_PREFIX + "arch." + k, int(v))
        rp = hp("rotary_pct", "partial_rotary_factor", default=1.0)
        writer.add_float32(ORKA_META_PREFIX + "arch.rotary_pct", float(rp))
        writer.add_float32(ORKA_META_PREFIX + "arch.rotary_base", float(hp("rotary_emb_base", "rope_theta", default=10000.0)))
        writer.add_float32(ORKA_META_PREFIX + "arch.ln_eps", float(hp("layer_norm_eps", "layer_norm_epsilon", default=1e-5)))
        writer.add_uint32(ORKA_META_PREFIX + "arch.parallel_residual", int(bool(hp("use_parallel_residual", default=True))))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {"out": str(out_path), "quantized_linears": n_quant, "passthrough": n_pass}
