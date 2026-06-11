#!/usr/bin/env python3
import sys
import json
import math
import argparse
import base64
import hashlib
from pathlib import Path
import numpy as np

# Add local path and llama.cpp/gguf-py to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType

from orka.transforms.rotate import _unrotate_flat
from orka.transforms.normalize import _apply_block_max_scales
from orka.pipeline.decode import _decode_tensor

XOR_KEY = b"ORKA_PRIVATE_KEY_2026_DO_NOT_SHARE"

def _xor_decrypt_array(arr: np.ndarray) -> np.ndarray:
    """XOR decrypts a numpy array by applying the key."""
    arr_bytes = np.ascontiguousarray(arr).view(np.uint8)
    key_arr = np.frombuffer(XOR_KEY, dtype=np.uint8)
    tiled_key = np.resize(key_arr, arr_bytes.shape)
    decrypted = np.bitwise_xor(arr_bytes, tiled_key)
    return decrypted

def decompress_gguf_tensor(tmeta, gguf_tensors, reader, cb_map, obfuscate=False):
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

    base_name = hashlib.md5(name.encode()).hexdigest()[:8] if obfuscate else name

    for stage in stages:
        sid = stage.get("stage", 0)
        cb_size = stage.get("codebook_size", int(tmeta["codebook_size"]))
        idx_bits = stage.get("index_bits", int(tmeta["index_bits"]))
        s_group_size = int(stage.get("group_size", group_size))
        s_index_count = math.ceil(padded_values / s_group_size)

        # 1. Load codebook
        original_cb_name = f"{name}.orka.s{sid}.codebook"
        if obfuscate:
            shared_cb_name = cb_map.get(original_cb_name)
            if shared_cb_name is None:
                raise ValueError(f"Missing cb_map metadata for {original_cb_name}")
        else:
            cb_map_key = f"orka.cb_map.{original_cb_name}"
            field = reader.fields.get(cb_map_key)
            if field is None:
                raise ValueError(f"Missing cb_map metadata for {original_cb_name}")
            shared_cb_name = field.contents()

        cb_tensor = gguf_tensors[shared_cb_name]
        if obfuscate:
            raw_bytes = cb_tensor.data.view(np.uint8)
            decrypted_bytes = _xor_decrypt_array(raw_bytes)
            cb_data = decrypted_bytes.view(np.float16).astype(np.float32)
        else:
            if cb_tensor.tensor_type == GGMLQuantizationType.Q8_0:
                cb_data = dequantize(cb_tensor.data, GGMLQuantizationType.Q8_0)
            else:
                cb_data = cb_tensor.data.view(np.float16).astype(np.float32)

        cb = cb_data.reshape(-1, s_group_size)

        # 2. Load indices
        idx_tensor_name = f"t.{base_name}.i{sid}" if obfuscate else f"{name}.orka.s{sid}.indices"
        idx_tensor = gguf_tensors[idx_tensor_name]
        if idx_bits > 8:
            indices = idx_tensor.data.view(np.int16).astype(np.int64)
        else:
            indices = idx_tensor.data.view(np.int8).astype(np.int64)

        decoded += cb[indices].reshape(-1)

    decoded = decoded[:packed_values]

    # 3. Apply rotation
    rotation = tmeta.get("rotation", "none")
    if rotation in {"orthogonal", "hadamard"}:
        seed = int(tmeta.get("rotation_seed") or 0)
        decoded_list = decoded.tolist()
        decoded_unrotated = _unrotate_flat(decoded_list, tmeta["shape"], rotation, seed)
        decoded = np.array(decoded_unrotated, dtype=np.float32)

    # 4. Apply scales
    norm = tmeta.get("normalization", "none")
    if norm in ("block-max", "channel-block-max", "awq-block-max", "slrq-block"):
        scales_tensor_name = f"t.{base_name}.s" if obfuscate else f"{name}.orka.scales"
        scales_tensor = gguf_tensors[scales_tensor_name]
        if obfuscate:
            raw_bytes = scales_tensor.data.view(np.uint8)
            decrypted_bytes = _xor_decrypt_array(raw_bytes)
            scales = decrypted_bytes.view(np.float16).astype(np.float32)
        else:
            if scales_tensor.tensor_type == GGMLQuantizationType.Q8_0:
                scales = dequantize(scales_tensor.data, GGMLQuantizationType.Q8_0)
            else:
                scales = scales_tensor.data.view(np.float16).astype(np.float32)

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
        sal_idx_name = f"t.{base_name}.x" if obfuscate else f"{name}.orka.salient.idx"
        sal_idx_tensor = gguf_tensors[sal_idx_name]

        # Read index with correct signed type mapping
        raw_data = sal_idx_tensor.data
        if raw_data.dtype == np.uint8:
            sal_idx = raw_data.view(np.int8).astype(np.int64)
        elif raw_data.dtype == np.uint16:
            sal_idx = raw_data.view(np.int16).astype(np.int64)
        else:
            sal_idx = raw_data.astype(np.int64)

        sal_val_name = f"t.{base_name}.y" if obfuscate else f"{name}.orka.salient.val"
        sal_val_tensor = gguf_tensors[sal_val_name]
        if obfuscate:
            raw_bytes = sal_val_tensor.data.view(np.uint8)
            decrypted_bytes = _xor_decrypt_array(raw_bytes)
            sal_val = decrypted_bytes.view(np.float16).astype(np.float32)
        else:
            if sal_val_tensor.tensor_type == GGMLQuantizationType.Q8_0:
                sal_val = dequantize(sal_val_tensor.data, GGMLQuantizationType.Q8_0)
            else:
                sal_val = sal_val_tensor.data.view(np.float16).astype(np.float32)

        block_size = int(tmeta.get("block_scale_size") or 32)
        for b_idx, (local_idx, weight) in enumerate(zip(sal_idx, sal_val)):
            flat_idx = b_idx * block_size + int(local_idx)
            if flat_idx < decoded.size:
                decoded[flat_idx] = float(weight)

    return decoded.reshape(shape)

def main():
    parser = argparse.ArgumentParser(description="Verify Orka GGUF decompression correctness.")
    parser.add_argument("orka_dir", type=Path, help="Path to reference .orka directory")
    parser.add_argument("gguf_path", type=Path, help="Path to GGUF file")
    parser.add_argument("--obfuscate", action="store_true", help="Expect obfuscated GGUF and manifest")
    args = parser.parse_args()

    orka_dir = args.orka_dir
    gguf_path = args.gguf_path

    print("=" * 60)
    print("  ORKA → GGUF Decompression & Correctness Verification")
    print("=" * 60)
    print(f"  Orka Dir:   {orka_dir}")
    print(f"  GGUF File:  {gguf_path}")
    print(f"  Obfuscated: {args.obfuscate}")
    print("-" * 60)

    reader = GGUFReader(gguf_path)
    gguf_tensors = {t.name: t for t in reader.tensors}

    cb_map = {}
    if args.obfuscate:
        # Load obfuscated manifest from sys.cfg
        sys_cfg_field = reader.fields.get("sys.cfg")
        if sys_cfg_field is None:
            print("Error: Obfuscated GGUF is missing 'sys.cfg' metadata key.", file=sys.stderr)
            sys.exit(1)
        b64_data = sys_cfg_field.parts[-1].tobytes()
        enc_manifest = base64.b64decode(b64_data)
        manifest_arr = np.frombuffer(enc_manifest, dtype=np.uint8)
        dec_manifest_arr = _xor_decrypt_array(manifest_arr)
        manifest = json.loads(dec_manifest_arr.tobytes().decode('utf-8'))
        cb_map = manifest.get("cb_map", {})
    else:
        with open(orka_dir / "manifest.json") as f:
            manifest = json.load(f)

    overall_max_diff = 0.0
    overall_sum_sq_diff = 0.0
    overall_elements = 0

    for tmeta in manifest["tensors"]:
        name = tmeta["name"]
        print(f"Verifying {name}...")

        # 1. Decompress using standard Orka library (from files)
        w_orka_list = _decode_tensor(orka_dir, tmeta)
        w_orka = np.array(w_orka_list, dtype=np.float32).reshape(tmeta["shape"])

        # 2. Decompress using GGUF
        w_gguf = decompress_gguf_tensor(tmeta, gguf_tensors, reader, cb_map, args.obfuscate)

        # 3. Compare
        diff = np.abs(w_orka - w_gguf)
        max_diff = np.max(diff)
        mse = np.mean(diff ** 2)

        print(f"  Shape:     {w_orka.shape}")
        print(f"  Max Diff:  {max_diff:.6f}")
        print(f"  MSE:       {mse:.6e}")

        if max_diff > overall_max_diff:
            overall_max_diff = max_diff
        overall_sum_sq_diff += np.sum(diff ** 2)
        overall_elements += w_orka.size

    overall_mse = overall_sum_sq_diff / overall_elements
    print("=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Overall Max Difference:  {overall_max_diff:.6f}")
    print(f"  Overall Mean Squared Error: {overall_mse:.6e}")
    print(f"  Overall RMSE:               {math.sqrt(overall_mse):.6e}")
    print("=" * 60)

if __name__ == "__main__":
    main()
