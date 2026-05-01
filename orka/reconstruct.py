"""reconstruct_artifact: decode all tensors to JSON or safetensors."""

from __future__ import annotations

import json
from pathlib import Path

from orka._checkpoint import _load_tensors
from orka._format import ORKA_VERSION
from orka._tensor import _flatten_float_values, _tensor_shape
from orka._util import _reshape_flat
from orka.pipeline.decode import _decode_tensor, _decode_tensor_torch


def _decoded_tensor_map(out_dir: Path, manifest: dict) -> dict:
    tensors = {}
    for tensor_meta in manifest.get("tensors", []):
        decoded = _decode_tensor(out_dir, tensor_meta)
        shape = [int(x) for x in tensor_meta.get("shape", [])]
        tensors[tensor_meta["name"]] = {
            "shape": shape,
            "flat": decoded,
            "values": _reshape_flat(decoded, shape),
        }
    return tensors


def _complete_decoded_tensor_map(out_dir: Path, manifest: dict) -> dict:
    tensors = {}
    packed_names = {t["name"] for t in manifest.get("tensors", [])}

    # Load passthrough tensors from artifact (self-contained, no source needed).
    passthrough_path = out_dir / "passthrough.safetensors"
    if passthrough_path.exists():
        for name, tensor in _load_tensors(passthrough_path):
            shape = _tensor_shape(tensor)
            flat = _flatten_float_values(tensor)
            tensors[name] = {"shape": shape, "flat": flat, "values": _reshape_flat(flat, shape)}

    # Fall back to source for anything still missing (backward compat, sensitivity-map skips).
    source = Path(manifest["source"])
    if source.exists():
        for name, tensor in _load_tensors(source):
            if name in packed_names or name in tensors:
                continue
            shape = _tensor_shape(tensor)
            flat = _flatten_float_values(tensor)
            tensors[name] = {"shape": shape, "flat": flat, "values": _reshape_flat(flat, shape)}

    tensors.update(_decoded_tensor_map(out_dir, manifest))
    return tensors

def _write_json_reconstruction(
    out_dir: Path, output_path: Path, manifest: dict, tensors: dict
) -> None:
    output = {
        "format": "orka-reconstruction",
        "version": ORKA_VERSION,
        "source_artifact": str(out_dir),
        "source_checkpoint": manifest.get("source"),
        "tensor_count": len(tensors),
        "tensors": {
            name: {
                "shape": tensor["shape"],
                "values": tensor["values"],
            }
            for name, tensor in tensors.items()
        },
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n")


def _write_safetensors_reconstruction(output_path: Path, tensors: dict) -> None:
    try:
        import numpy as np
        from safetensors.numpy import save_file
    except Exception as exc:
        raise RuntimeError(
            "safetensors reconstruction requires numpy and safetensors"
        ) from exc

    arrays = {}
    for name, tensor in tensors.items():
        arrays[name] = np.asarray(tensor["flat"], dtype=np.float32).reshape(
            tensor["shape"]
        )
    save_file(arrays, str(output_path))

def _write_complete_safetensors_reconstruction(
    out_dir: Path, output_path: Path, manifest: dict, device: str | None = None
) -> dict:
    """Reconstruct full model. Uses GPU streaming path when device='cuda' to avoid Python list bloat."""
    if device is not None and "cuda" in str(device).lower():
        try:
            import torch
            if torch.cuda.is_available():
                from safetensors.torch import save_file as save_torch
                from safetensors import safe_open
                arrays: dict = {}
                packed_names = {t["name"] for t in manifest.get("tensors", [])}
                # Passthrough first
                pp = out_dir / "passthrough.safetensors"
                if pp.exists():
                    with safe_open(str(pp), framework="pt") as f:
                        for name in f.keys():
                            arrays[name] = f.get_tensor(name).contiguous()
                # Source fallback for anything missing
                source = Path(manifest["source"])
                if source.exists():
                    with safe_open(str(source), framework="pt") as f:
                        for name in f.keys():
                            if name in packed_names or name in arrays:
                                continue
                            arrays[name] = f.get_tensor(name).contiguous()
                # GPU decode quantized tensors, move to CPU immediately to free GPU memory
                for tm in manifest.get("tensors", []):
                    dec_gpu = _decode_tensor_torch(out_dir, tm, device)
                    arrays[tm["name"]] = dec_gpu.cpu().contiguous()
                    del dec_gpu
                    torch.cuda.empty_cache()
                save_torch(arrays, str(output_path))
                return {"out": str(output_path), "tensor_count": len(arrays), "format": "safetensors"}
        except Exception as exc:
            print(f"GPU reconstruction failed ({exc}); falling back to numpy path", flush=True)
    # CPU/numpy fallback (the slow path)
    tensors = _complete_decoded_tensor_map(out_dir, manifest)
    _write_safetensors_reconstruction(output_path, tensors)
    return {
        "out": str(output_path),
        "tensor_count": len(tensors),
        "format": "safetensors",
    }

def reconstruct_artifact(
    out_dir: Path, output_path: Path, output_format: str = "json"
) -> dict:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Orka manifest: {manifest_path}")
    if output_format not in {"json", "safetensors"}:
        raise ValueError("output_format must be 'json' or 'safetensors'")

    manifest = json.loads(manifest_path.read_text())
    tensors = _decoded_tensor_map(out_dir, manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        _write_json_reconstruction(out_dir, output_path, manifest, tensors)
    else:
        tensors = _complete_decoded_tensor_map(out_dir, manifest)
        _write_safetensors_reconstruction(output_path, tensors)

    return {
        "out": str(output_path),
        "tensor_count": len(tensors),
        "format": output_format,
    }
