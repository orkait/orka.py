"""Source checkpoint loading (.safetensors / .pt / .bin / .json) + inspect."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from orka.core._tensor import _tensor_numel, _tensor_shape


def _tensor_shapes(path: Path) -> dict:
    """Tensor name -> shape tuple, read CHEAPLY (safetensors header only - no tensor data
    materialized). Used to resolve structural head/recurrent detection up front without a
    second full checkpoint load. Returns {} for formats whose shapes need a real load
    (.pt/.bin/.json) - callers then fall back to name-based detection."""
    import struct

    shapes: dict = {}
    if path.is_dir():
        for shard in sorted(path.glob("*.safetensors")):
            shapes.update(_tensor_shapes(shard))
        return shapes
    if path.suffix.lower() != ".safetensors":
        return shapes
    try:
        with path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for name, info in header.items():
            if name != "__metadata__" and isinstance(info, dict) and "shape" in info:
                shapes[name] = tuple(info["shape"])
    except Exception:
        return {}
    return shapes


def _read_vocab_size(source: Path) -> int | None:
    """Best-effort vocab_size from a config.json beside the source (dir or file parent).
    Only a refinement: output_head_names falls back to the dominant output dim without it."""
    base = source if source.is_dir() else source.parent
    try:
        cfg = json.loads((base / "config.json").read_text())
        v = cfg.get("vocab_size")
        return int(v) if v else None
    except Exception:
        return None


def _load_tensors(path: Path) -> Iterable[tuple[str, object]]:
    if path.is_dir():
        # Sharded Checkpoint Support
        print(f"INFO: Loading sharded checkpoint from directory: {path}", flush=True)
        # Priority: Safetensors -> Torch -> Bin
        patterns = ["*.safetensors", "*.pt", "*.pth", "*.bin"]
        found_any = False
        for pattern in patterns:
            for shard in sorted(path.glob(pattern)):
                yield from _load_tensors(shard)
                found_any = True
            if found_any: break
        return

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

        # Candidate logic: Dense weights only.
        # Exclude biases, norms, and architectural sidecars.
        is_candidate = len(shape) >= 2
        name_lower = name.lower()
        if any(
            x in name_lower
            for x in (".bias", ".norm", ".layernorm", "rotary_emb", "attention.bias")
        ):
            is_candidate = False

        tensors.append(
            {
                "name": name,
                "shape": shape,
                "numel": numel,
                "candidate": is_candidate,
            }
        )
    return {
        "source": str(path),
        "tensor_count": len(tensors),
        "total_params": total_params,
        "tensors": tensors,
    }
