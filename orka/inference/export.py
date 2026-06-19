"""export_inference: build an HF model with nn.Linear replaced by VQLinear.

Unlike export_vllm (which reconstructs dense bf16 weights), this keeps weights
in VQ format and runs them via the Triton VQ-GEMM kernel at inference time.
The resulting model is ~2-2.5x smaller in GPU memory than the dense bf16 model.

Usage:
    from orka.inference import export_inference
    model = export_inference("path/to/artifact.orka", "path/to/hf-model-dir")
    # model is a standard HF model, all quantized linears run via Triton
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn


def export_inference(
    artifact_dir: Union[str, Path],
    hf_model_dir: Union[str, Path],
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """Load an HF model and replace quantized nn.Linear layers with VQLinear.

    Args:
        artifact_dir: path to the .orka artifact directory (contains manifest.json)
        hf_model_dir: path to the original HF model directory (for config + tokenizer)
        device: target device ("cuda" or "cpu")
        dtype: dtype for non-quantized weights

    Returns:
        HF model with VQLinear layers, ready for inference.
    """
    from orka._checkpoint import _load_tensors
    from orka.inference.vq_linear import build_vq_linear

    artifact_dir = Path(artifact_dir)
    hf_model_dir = Path(hf_model_dir)

    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    meta_map = {t["name"]: t for t in manifest.get("tensors", [])}

    # Load passthrough tensors (norms, biases, unquantized weights)
    pp_path = artifact_dir / "passthrough.safetensors"
    pp_tensors: dict[str, torch.Tensor] = {}
    if pp_path.exists():
        for k, t in _load_tensors(pp_path):
            if not isinstance(t, torch.Tensor):
                import numpy as np
                t = torch.from_numpy(np.asarray(t, dtype=np.float32))
            pp_tensors[k] = t.to(dtype)

    # Load HF model (CPU first, then we'll move buffers to device selectively)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        local_files_only=True,
        torch_dtype=dtype,
    )

    n_replaced = 0

    def _replace(module: nn.Module, prefix: str = "") -> None:
        nonlocal n_replaced
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            weight_key = f"{full}.weight"

            if isinstance(child, nn.Linear) and weight_key in meta_map:
                bias_t = None
                bias_key = f"{full}.bias"
                if bias_key in pp_tensors:
                    bias_t = pp_tensors[bias_key]
                elif child.bias is not None:
                    bias_t = child.bias.data

                vql = build_vq_linear(
                    artifact_dir=artifact_dir,
                    tensor_meta=meta_map[weight_key],
                    bias=bias_t,
                    device=device,
                )
                setattr(module, name, vql)
                n_replaced += 1
            else:
                _replace(child, full)

    print(f"replacing quantized linears with VQLinear ...", flush=True)
    _replace(model)
    print(f"  replaced {n_replaced} layers", flush=True)

    # Move remaining (non-VQLinear) weights to device
    for name, param in model.named_parameters():
        if param.device.type != device:
            param.data = param.data.to(device)
    for name, buf in model.named_buffers():
        if buf is not None and buf.device.type != device:
            buf.data = buf.data.to(device)

    return model
