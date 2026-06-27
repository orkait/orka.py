"""Orka PyTorch layers and model replacement wrappers for dynamic reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
import torch
import torch.nn as nn

from orka._checkpoint import _load_tensors
from orka.pipeline.decode import _decode_tensor_torch


class OrkaLinear(nn.Module):
    """Custom PyTorch Linear layer that reconstructs Orka-packed weights on-the-fly.

    Avoids storing the unquantized float32 weight parameter on disk,
    decompressing it on-demand to run standard linear transformations.
    """

    def __init__(
        self,
        out_dir: str | Path,
        tensor_meta: dict,
        bias_tensor: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.out_dir = Path(out_dir)
        self.tensor_meta = tensor_meta
        self.in_features = int(tensor_meta["shape"][1])
        self.out_features = int(tensor_meta["shape"][0])

        if bias_tensor is not None:
            self.bias = nn.Parameter(bias_tensor.clone().to(torch.float32))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("_reconstructed_weight", None)

    def reconstruct_weight(self, device: str | torch.device) -> torch.Tensor:
        """Decompress weight parameters to a standard PyTorch float32 tensor on the target device."""
        target_device = torch.device(device)
        if self._reconstructed_weight is not None and self._reconstructed_weight.device == target_device:
            return self._reconstructed_weight

        # Decompress on target device using GPU-accelerated decoder
        w = _decode_tensor_torch(self.out_dir, self.tensor_meta, str(device))
        self._reconstructed_weight = w
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.reconstruct_weight(x.device)
        return torch.nn.functional.linear(x.astype(w.dtype) if hasattr(x, "astype") else x.to(w.dtype), w, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def replace_linear_with_orka(model: nn.Module, out_dir: str | Path) -> None:
    """Recursively replaces nn.Linear modules with OrkaLinear layers loaded from an Orka directory."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Orka manifest: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    meta_map = {t["name"]: t for t in manifest.get("tensors", [])}

    # Load passthrough/bias tensors
    pp_path = out_dir / "passthrough.safetensors"
    pp_tensors = {}
    if pp_path.exists():
        for k, tensor in _load_tensors(pp_path):
            if not isinstance(tensor, torch.Tensor):
                import numpy as np
                tensor = torch.from_numpy(np.asarray(tensor, dtype=np.float32))
            else:
                tensor = tensor.to(torch.float32)
            pp_tensors[k] = tensor

    # Source fallback for any unquantized weights
    source_path = Path(manifest.get("source", ""))
    source_tensors = {}
    if source_path.exists():
        for k, tensor in _load_tensors(source_path):
            if not isinstance(tensor, torch.Tensor):
                import numpy as np
                tensor = torch.from_numpy(np.asarray(tensor, dtype=np.float32))
            else:
                tensor = tensor.to(torch.float32)
            source_tensors[k] = tensor

    def _replace(module: nn.Module, prefix="") -> None:
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                weight_name = f"{full_name}.weight"
                bias_name = f"{full_name}.bias"

                if weight_name in meta_map:
                    bias_t = None
                    if bias_name in pp_tensors:
                        bias_t = pp_tensors[bias_name]
                    elif bias_name in source_tensors:
                        bias_t = source_tensors[bias_name]

                    layer = OrkaLinear(out_dir, meta_map[weight_name], bias_t)
                    setattr(module, name, layer)
            else:
                _replace(child, full_name)

    _replace(model)
