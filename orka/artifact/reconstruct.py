"""reconstruct_artifact: decode all tensors to JSON or safetensors."""

from __future__ import annotations

import json
from pathlib import Path

from orka.core._checkpoint import _load_tensors
from orka.core._format import ORKA_VERSION
from orka.core._tensor import _flatten_float_values, _tensor_shape
from orka.core._util import _product, _reshape_flat
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


def _write_complete_safetensors_reconstruction_binary(
    out_dir: Path, output_path: Path, manifest: dict, device: str | None = None
) -> dict:
    """Reconstruct full model using a custom binary writer to avoid RAM OOMs."""
    import struct
    import numpy as np
    from safetensors import safe_open

    # 1. IDENTIFY ALL TENSORS AND THEIR SHAPES (WITHOUT LOADING DATA)
    # We need name -> {shape, source_type, meta_ptr}
    registry = {}
    packed_names = {t["name"] for t in manifest.get("tensors", [])}

    # A. Passthrough tensors (from Orka artifact)
    pp = out_dir / "passthrough.safetensors"
    if pp.exists():
        with safe_open(str(pp), framework="np") as f:
            for name in f.keys():
                registry[name] = {"shape": f.get_slice(name).get_shape(), "source": "passthrough"}

    # B. Source fallback (anything missing from packed/passthrough)
    source = Path(manifest["source"])
    source_map = {}
    if source.exists():
        print(f"INFO: Indexing source checkpoint for reconstruction fallback...", flush=True)
        for name, tensor in _load_tensors(source):
            if name not in packed_names and name not in registry:
                registry[name] = {"shape": _tensor_shape(tensor), "source": "source_fallback"}
                source_map[name] = tensor

    # 2. CALCULATE OFFSETS AND BUILD HEADER
    for tm in manifest.get("tensors", []):
        registry[tm["name"]] = {"shape": tm["shape"], "source": "quantized", "meta": tm}

    # 2. CALCULATE OFFSETS AND BUILD HEADER
    header = {"__metadata__": {"format": "pt" if device and "cuda" in str(device) else "np"}}
    current_offset = 0
    
    # Sort names for deterministic layout
    sorted_names = sorted(registry.keys())
    
    for name in sorted_names:
        reg = registry[name]
        shape = [int(x) for x in reg["shape"]]
        numel = 1
        for dim in shape:
            numel *= dim
        
        # All Orka reconstructions are float32 (4 bytes)
        byte_size = numel * 4
        
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [current_offset, current_offset + byte_size]
        }
        current_offset += byte_size

    # 3. SERIALIZE HEADER
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_json)
    
    # Safetensors requires the data block to start at an 8-byte aligned offset.
    # Data start = 8 (prefix) + N (header_len). 
    # To make (8 + N) % 8 == 0, N must be a multiple of 8.
    padding = (8 - (header_len % 8)) % 8
    header_json_padded = header_json + (b" " * padding)
    header_full_len = len(header_json_padded)
    
    # 4. STREAM TO DISK
    print(f"Streaming reconstruction to {output_path.name} ({len(sorted_names)} tensors)...", flush=True)
    with open(output_path, "wb") as f:
        # 8-byte little-endian header size (N)
        f.write(struct.pack("<Q", header_full_len))
        # JSON Header + Space Padding
        f.write(header_json_padded)
        
        # Data block: decode and write ONE-BY-ONE
        for i, name in enumerate(sorted_names):
            reg = registry[name]
            
            arr = None
            if reg["source"] == "quantized":
                if device and "cuda" in str(device).lower():
                    # GPU decode
                    import torch
                    dec = _decode_tensor_torch(out_dir, reg["meta"], device)
                    arr = dec.detach().cpu().numpy().astype(np.float32)
                    del dec
                    torch.cuda.empty_cache()
                else:
                    # CPU decode
                    dec = _decode_tensor(out_dir, reg["meta"])
                    arr = np.asarray(dec, dtype=np.float32)
            else:
                # Passthrough or source fallback: handle BF16 via torch if possible
                if reg["source"] == "passthrough":
                    try:
                        import torch
                        with safe_open(str(pp), framework="pt") as s:
                            arr = s.get_tensor(name).to(torch.float32).cpu().numpy()
                    except (ImportError, RuntimeError):
                        with safe_open(str(pp), framework="np") as s:
                            arr = s.get_tensor(name).astype(np.float32)
                else: # source_fallback
                    t = source_map.get(name)
                    if t is not None:
                        # Convert to numpy float32
                        if hasattr(t, "detach"): # Torch
                            import torch
                            arr = t.detach().cpu().to(torch.float32).numpy()
                        else: # Numpy
                            arr = np.asarray(t, dtype=np.float32)
                    else:
                        raise ValueError(f"CRITICAL: Tensor {name} missing from both artifact and source.")
            
            f.write(arr.tobytes())
            del arr # Mandatory cleanup


    return {
        "out": str(output_path),
        "tensor_count": len(sorted_names),
        "format": "safetensors",
    }


def reconstruct_artifact(
    out_dir: Path, output_path: Path, output_format: str = "json", device: str | None = None
) -> dict:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Orka manifest: {manifest_path}")
    if output_format not in {"json", "safetensors"}:
        raise ValueError("output_format must be 'json' or 'safetensors'")

    manifest = json.loads(manifest_path.read_text())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "safetensors":
        # An artifact packed with --max-values-per-tensor stores fewer values
        # than the shape implies; a safetensors header built from the full shape
        # would be byte-misaligned. Refuse with a clear error instead.
        truncated = [
            t["name"]
            for t in manifest.get("tensors", [])
            if int(t.get("packed_values", 0)) != _product([int(x) for x in t.get("shape", [])])
        ]
        if truncated:
            raise ValueError(
                f"cannot reconstruct to safetensors: {len(truncated)} tensor(s) were packed "
                f"with --max-values-per-tensor (packed_values < shape), e.g. {truncated[0]}. "
                "Re-pack without the sample limit, or use --format json."
            )

    if output_format == "json":
        tensors = _decoded_tensor_map(out_dir, manifest)
        _write_json_reconstruction(out_dir, output_path, manifest, tensors)
        return {
            "out": str(output_path),
            "tensor_count": len(tensors),
            "format": output_format,
        }
    else:
        return _write_complete_safetensors_reconstruction_binary(out_dir, output_path, manifest, device)
