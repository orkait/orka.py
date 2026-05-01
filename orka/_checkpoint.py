"""Source checkpoint loading (.safetensors / .pt / .bin / .json) + inspect."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from orka._tensor import _tensor_numel, _tensor_shape


def _load_tensors(path: Path) -> Iterable[tuple[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open() as f:
            loaded = json.load(f)
        tensors = loaded.get("tensors", loaded) if isinstance(loaded, dict) else loaded
        if not isinstance(tensors, dict):
            raise ValueError("JSON input must be an object or contain a tensors object")
        for name, tensor in tensors.items():
            yield name, tensor
        return

    if suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except Exception as exc:
            raise RuntimeError(
                "safetensors input requires the safetensors package"
            ) from exc
        try:
            import torch  # noqa: F401
        except Exception:
            framework = "np"
        else:
            framework = "pt"
        with safe_open(str(path), framework=framework) as handle:
            for name in handle.keys():
                try:
                    yield name, handle.get_tensor(name)
                except TypeError as exc:
                    if framework == "np":
                        raise RuntimeError(
                            f"safetensors tensor {name} uses a dtype that requires torch loading"
                        ) from exc
                    raise
        return

    if suffix in {".pt", ".pth", ".bin"}:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch checkpoint input requires torch") from exc
        loaded = torch.load(path, map_location="cpu")
        state = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else loaded
        if not isinstance(state, dict):
            raise ValueError(
                "checkpoint must load to a tensor dictionary or contain state_dict"
            )
        for name, tensor in state.items():
            yield name, tensor
        return

    raise ValueError(f"unsupported input format: {path.suffix}")

def inspect_checkpoint(path: Path) -> dict:
    tensors = []
    total_params = 0
    for name, tensor in _load_tensors(path):
        numel = _tensor_numel(tensor)
        shape = _tensor_shape(tensor)
        if numel <= 0:
            continue
        total_params += numel
        tensors.append(
            {
                "name": name,
                "shape": shape,
                "numel": numel,
                "candidate": len(shape) >= 2,
            }
        )
    return {
        "source": str(path),
        "tensor_count": len(tensors),
        "total_params": total_params,
        "tensors": tensors,
    }
