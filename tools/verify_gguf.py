#!/usr/bin/env python3
"""Verify Orka -> GGUF decompression correctness.

Decodes every tensor twice - once through the reference Python decoder
(orka.pipeline.decode._decode_tensor, reading the .orka sidecars) and once from
the GGUF file - and reports the difference. A small residual is expected: the
GGUF codebooks/scales are stored Q8_0, so the difference is Q8_0 quantization
noise, not a structural decode error.
"""
import sys
import json
import math
import argparse
from pathlib import Path
import numpy as np

# Add local path and llama.cpp/gguf-py to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType

from orka.transforms.rotate import _unrotate_flat
from orka.pipeline.decode import _decode_tensor


def _dequant_or_fp16(tensor) -> np.ndarray:
    """Read a GGUF float payload: dequantize Q8_0, else interpret as fp16."""
    if tensor.tensor_type == GGMLQuantizationType.Q8_0:
        return dequantize(tensor.data, GGMLQuantizationType.Q8_0)
    return tensor.data.view(np.float16).astype(np.float32)


def decompress_gguf_tensor(tmeta, gguf_tensors, reader):
    name = tmeta["name"]
    group_size = int(tmeta["group_size"])
    padded_values = int(tmeta["padded_values"])
    packed_values = int(tmeta["packed_values"])
    shape = [int(x) for x in tmeta["shape"]]
    index_count = math.ceil(padded_values / group_size)

    decoded = np.zeros(index_count * group_size, dtype=np.float32)
    stages = tmeta.get("stages", [])
    if not stages:
        stages = [{
            "stage": 0,
            "codebook_size": int(tmeta["codebook_size"]),
            "index_bits": int(tmeta["index_bits"]),
        }]

    for stage in stages:
        sid = stage.get("stage", 0)
        idx_bits = int(stage.get("index_bits", tmeta["index_bits"]))
        s_group_size = int(stage.get("group_size", group_size))

        # 1. Load codebook (via the cb dedup map)
        original_cb_name = f"{name}.orka.s{sid}.codebook"
        cb_map_key = f"orka.cb_map.{original_cb_name}"
        field = reader.fields.get(cb_map_key)
        if field is None:
            raise ValueError(f"Missing cb_map metadata for {original_cb_name}")
        shared_cb_name = field.contents()
        cb = _dequant_or_fp16(gguf_tensors[shared_cb_name]).reshape(-1, s_group_size)

        # 2. Load indices. The writer stored the unsigned index bit pattern in a
        #    signed GGML int tensor; reinterpret as unsigned to recover values
        #    (correct for indices >= 2^15, where a signed read would go negative).
        idx_tensor = gguf_tensors[f"{name}.orka.s{sid}.indices"]
        if idx_bits > 8:
            indices = idx_tensor.data.view(np.uint16).astype(np.int64)
        else:
            indices = idx_tensor.data.view(np.uint8).astype(np.int64)

        decoded += cb[indices].reshape(-1)

    decoded = decoded[:packed_values]

    # Outlier / pillar escape: absolute positions overwritten pre-rotation / pre-scale.
    outl = tmeta.get("outliers")
    if outl:
        pos = gguf_tensors[f"{name}.orka.outlier.idx"].data.astype(np.int64)
        val = _dequant_or_fp16(gguf_tensors[f"{name}.orka.outlier.val"])
        mask = pos < decoded.size
        decoded[pos[mask]] = val[mask]

    # 3. Apply rotation
    rotation = tmeta.get("rotation", "none")
    if rotation in {"orthogonal", "hadamard"}:
        seed = int(tmeta.get("rotation_seed") or 0)
        decoded = np.array(
            _unrotate_flat(decoded.tolist(), tmeta["shape"], rotation, seed),
            dtype=np.float32,
        )

    # 4. Apply scales
    norm = tmeta.get("normalization", "none")
    if norm in ("block-max", "channel-block-max", "awq-block-max", "slrq-block"):
        scales = _dequant_or_fp16(gguf_tensors[f"{name}.orka.scales"])
        block_size = int(tmeta.get("block_scale_size") or 32)
        n = decoded.size
        pad = (-n) % block_size
        if pad:
            decoded = np.concatenate([decoded, np.zeros(pad, dtype=np.float32)])
        decoded = (decoded.reshape(-1, block_size) * scales[:decoded.size // block_size, None]).reshape(-1)
        if pad:
            decoded = decoded[:n]

    # 5. Apply salient outliers
    salient = tmeta.get("salient")
    if salient:
        sal_idx = gguf_tensors[f"{name}.orka.salient.idx"].data.astype(np.int64)
        sal_val = _dequant_or_fp16(gguf_tensors[f"{name}.orka.salient.val"])
        block_size = int(tmeta.get("block_scale_size") or 32)
        for b_idx, (local_idx, weight) in enumerate(zip(sal_idx, sal_val)):
            flat_idx = b_idx * block_size + int(local_idx)
            if flat_idx < decoded.size:
                decoded[flat_idx] = float(weight)

    # Low-rank correction: decoded += (A @ B^T), applied last.
    lr = tmeta.get("lowrank")
    if lr:
        r = int(lr["rank"])
        a = _dequant_or_fp16(gguf_tensors[f"{name}.orka.lowrank.a"]).reshape(-1, r)
        b = _dequant_or_fp16(gguf_tensors[f"{name}.orka.lowrank.b"]).reshape(-1, r)
        rows = a.shape[0]
        cols = b.shape[0]
        decoded = (decoded[:rows * cols].reshape(rows, cols) + a @ b.T).reshape(-1)

    return decoded.reshape(shape)


def main():
    parser = argparse.ArgumentParser(description="Verify Orka GGUF decompression correctness.")
    parser.add_argument("orka_dir", type=Path, help="Path to reference .orka directory")
    parser.add_argument("gguf_path", type=Path, help="Path to GGUF file")
    args = parser.parse_args()

    orka_dir = args.orka_dir
    gguf_path = args.gguf_path

    print("=" * 60)
    print("  ORKA -> GGUF Decompression & Correctness Verification")
    print("=" * 60)
    print(f"  Orka Dir:   {orka_dir}")
    print(f"  GGUF File:  {gguf_path}")
    print("-" * 60)

    reader = GGUFReader(gguf_path)
    gguf_tensors = {t.name: t for t in reader.tensors}

    with open(orka_dir / "manifest.json") as f:
        manifest = json.load(f)

    overall_max_diff = 0.0
    overall_sum_sq_diff = 0.0
    overall_elements = 0

    for tmeta in manifest["tensors"]:
        name = tmeta["name"]
        print(f"Verifying {name}...")

        # 1. Decompress using standard Orka library (from files)
        w_orka = np.array(_decode_tensor(orka_dir, tmeta), dtype=np.float32).reshape(tmeta["shape"])

        # 2. Decompress using GGUF
        w_gguf = decompress_gguf_tensor(tmeta, gguf_tensors, reader)

        # 3. Compare
        diff = np.abs(w_orka - w_gguf)
        max_diff = float(np.max(diff)) if diff.size else 0.0
        mse = float(np.mean(diff ** 2)) if diff.size else 0.0

        print(f"  Shape:     {w_orka.shape}")
        print(f"  Max Diff:  {max_diff:.6f}")
        print(f"  MSE:       {mse:.6e}")

        overall_max_diff = max(overall_max_diff, max_diff)
        overall_sum_sq_diff += float(np.sum(diff ** 2))
        overall_elements += w_orka.size

    overall_mse = overall_sum_sq_diff / overall_elements if overall_elements else 0.0
    print("=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Overall Max Difference:     {overall_max_diff:.6f}")
    print(f"  Overall Mean Squared Error: {overall_mse:.6e}")
    print(f"  Overall RMSE:               {math.sqrt(overall_mse):.6e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
