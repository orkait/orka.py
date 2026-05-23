#!/usr/bin/env python3
"""
Custom GGUF writer for Orka compressed models.

Packs the raw compressed representation (codebooks, indices, scales, salient
outliers) directly into a single GGUF file *without* decompressing to FP16/FP32.

Optimizations applied:
  1. Codebook deduplication - identical codebooks are written once and referenced
     by a shared tensor name.  Per-tensor metadata maps to the shared name.
  2. FP16 downcast - codebooks, block scales, and salient values are stored as
     FP16 instead of FP32, halving their size with negligible quality loss.

Tensor naming convention inside the GGUF:
    <weight_name>.orka.s<N>.codebook   – FP16 codebook  [codebook_size, group_size]
      (or shared: orka.shared_cb.<hash>.s<N>)
    <weight_name>.orka.s<N>.indices    – I16/I8 indices  [vector_count]
    <weight_name>.orka.scales          – FP16 block scales
    <weight_name>.orka.salient.idx     – I32 salient weight indices
    <weight_name>.orka.salient.val     – FP16 salient weight values

Non-quantized tensors (norms, biases) are stored as regular FP32 tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add project root and gguf-py to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "llama.cpp" / "gguf-py"))

import gguf
from gguf import GGUFWriter, GGMLQuantizationType
from gguf.quants import quantize as ggml_quantize

from orka._checkpoint import _load_tensors
from orka._format import _read_salient

# ──────────────────────────────────────────────────────────────────────
#  Obfuscation & Encryption Helpers
# ──────────────────────────────────────────────────────────────────────

XOR_KEY = b"ORKA_PRIVATE_KEY_2026_DO_NOT_SHARE"

def _xor_encrypt_array(arr: np.ndarray) -> np.ndarray:
    """XOR encrypts a numpy array by viewing it as uint8 and applying the key."""
    # Ensure it's a contiguous C-array
    arr_bytes = np.ascontiguousarray(arr).view(np.uint8)
    key_arr = np.frombuffer(XOR_KEY, dtype=np.uint8)
    # Tile key to match length
    tiled_key = np.resize(key_arr, arr_bytes.shape)
    encrypted = np.bitwise_xor(arr_bytes, tiled_key)
    return encrypted

# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────

def _load_raw(path: Path, dtype: np.dtype) -> np.ndarray:
    """Load a raw binary file as a flat numpy array."""
    return np.fromfile(str(path), dtype=dtype)


def _index_dtype(index_bits: int) -> np.dtype:
    if index_bits <= 8:
        return np.dtype(np.uint8)
    elif index_bits <= 16:
        return np.dtype(np.uint16)
    else:
        raise ValueError(f"Unsupported index_bits: {index_bits}")


def _file_hash(path: Path) -> str:
    """Fast MD5 hash of a file's contents."""
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


# ──────────────────────────────────────────────────────────────────────
#  GGUF Metadata
# ──────────────────────────────────────────────────────────────────────

def _write_model_metadata(writer: GGUFWriter, config: dict, manifest: dict, obfuscate: bool = False) -> None:
    """Write standard LLM metadata KV pairs expected by GGUF readers."""
    writer.add_string("general.name", f"orka-{config.get('model_type', 'llama')}-{config.get('hidden_size', 0)}")
    writer.add_string("general.description", "Orka RVQ-compressed model (optimized)")
    writer.add_string("general.file_type", "orka-rvq-optimized")

    if obfuscate:
        import base64
        manifest_json = json.dumps(manifest).encode('utf-8')
        manifest_arr = np.frombuffer(manifest_json, dtype=np.uint8)
        enc_manifest = _xor_encrypt_array(manifest_arr)
        b64_manifest = base64.b64encode(enc_manifest.tobytes()).decode('utf-8')
        writer.add_string("sys.cfg", b64_manifest)
    else:
        # Orka-specific metadata
        writer.add_string("orka.format", manifest.get("format", "orka"))
        writer.add_uint32("orka.version", manifest.get("version", 1))
        writer.add_uint32("orka.group_size", manifest.get("group_size", 8))
        writer.add_uint32("orka.n_stages", manifest.get("n_stages", 1))
        writer.add_string("orka.normalization", manifest.get("normalization", "none"))
        writer.add_string("orka.codebook_mode", manifest.get("codebook_mode", "per-tensor"))
        writer.add_string("orka.backend", manifest.get("backend", "torch"))
        writer.add_uint32("orka.tensor_count", manifest.get("tensor_count", 0))
        writer.add_uint32("orka.passthrough_count", manifest.get("passthrough_count", 0))
        writer.add_bool("orka.fp16_optimized", True)
        writer.add_bool("orka.codebook_dedup", True)

    # Standard LLM hyperparameters
    writer.add_uint32("llama.context_length", config.get("max_position_embeddings", 2048))
    writer.add_uint32("llama.embedding_length", config.get("hidden_size", 0))
    writer.add_uint32("llama.feed_forward_length", config.get("intermediate_size", 0))
    writer.add_uint32("llama.block_count", config.get("num_hidden_layers", 0))
    writer.add_uint32("llama.attention.head_count", config.get("num_attention_heads", 0))
    writer.add_uint32("llama.attention.head_count_kv", config.get("num_key_value_heads", 0))
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", config.get("rms_norm_eps", 1e-5))
    writer.add_float32("llama.rope.freq_base", config.get("rope_theta", 10000.0))
    writer.add_uint32("llama.vocab_size", config.get("vocab_size", 0))


# ──────────────────────────────────────────────────────────────────────
#  Codebook deduplication registry
# ──────────────────────────────────────────────────────────────────────

class CodebookRegistry:
    """Tracks unique codebooks and maps duplicate file paths to shared tensor names."""

    def __init__(self, obfuscate: bool = False):
        self.obfuscate = obfuscate
        self._hash_to_name: dict[str, str] = {}  # file_hash → shared tensor name
        self._mappings: dict[str, str] = {}       # original tensor name → shared tensor name
        self.saved_bytes = 0
        self.dedup_count = 0

    def register(self, tensor_name: str, cb_path: Path, cb_data: np.ndarray) -> tuple[str, bool]:
        """Register a codebook. Returns (tensor_name_to_use, is_new)."""
        h = _file_hash(cb_path)
        if h in self._hash_to_name:
            shared_name = self._hash_to_name[h]
            self._mappings[tensor_name] = shared_name
            self.saved_bytes += cb_data.nbytes
            self.dedup_count += 1
            return shared_name, False
        else:
            # First occurrence uses as the shared reference
            prefix = "sys.shared." if self.obfuscate else "orka.shared_cb."
            shared_name = f"{prefix}{h}"
            self._hash_to_name[h] = shared_name
            self._mappings[tensor_name] = shared_name
            return shared_name, True

    def get_mapping_kv(self) -> dict[str, str]:
        """Return the full mapping for storage as GGUF metadata."""
        return dict(self._mappings)


# ──────────────────────────────────────────────────────────────────────
#  Add Orka compressed tensors (optimized)
# ──────────────────────────────────────────────────────────────────────

def _quantize_to_q8(data_fp32: np.ndarray, tensor_name: str, writer: GGUFWriter) -> int:
    """Quantize a FP32 array to Q8_0 and add it to the GGUF writer.

    Q8_0 requires the flat array length to be a multiple of 32.
    If not aligned, falls back to FP16.
    """
    flat = data_fp32.flatten().astype(np.float32)
    if flat.size % 32 == 0:
        q8 = ggml_quantize(flat, GGMLQuantizationType.Q8_0)
        writer.add_tensor(
            tensor_name, q8,
            raw_dtype=GGMLQuantizationType.Q8_0,
        )
        return q8.nbytes
    else:
        fp16 = data_fp32.astype(np.float16).flatten()
        writer.add_tensor(tensor_name, fp16)
        return fp16.nbytes


def _encrypt_to_tensor(data_fp32: np.ndarray, tensor_name: str, writer: GGUFWriter) -> int:
    """XOR encrypts data and saves it as Int8 in the GGUF file."""
    fp16 = data_fp32.astype(np.float16).flatten()
    encrypted = _xor_encrypt_array(fp16)
    # GGUF supports Int8 natively
    writer.add_tensor(tensor_name, encrypted.view(np.int8))
    return encrypted.nbytes


def _add_orka_tensor(
    writer: GGUFWriter, orka_dir: Path, tmeta: dict,
    cb_registry: CodebookRegistry, obfuscate: bool = False
) -> int:
    """Add all compressed sub-tensors for one Orka weight. Returns total bytes written."""
    name = tmeta["name"]
    group_size = tmeta["group_size"]
    total_bytes = 0

    # Hash name if obfuscating
    base_name = hashlib.md5(name.encode()).hexdigest()[:8] if obfuscate else name

    # Per-stage codebooks and indices
    for stage in tmeta.get("stages", []):
        sid = stage["stage"]
        cb_size = stage["codebook_size"]
        idx_bits = stage["index_bits"]

        # Codebook: deduplicate + Q8_0 quantize (or encrypt)
        cb_path = orka_dir / stage["codebook"]
        s_group_size = stage.get("group_size", group_size)
        cb_fp32 = _load_raw(cb_path, np.float32).reshape(cb_size, s_group_size)
        original_tensor_name = f"{name}.orka.s{sid}.codebook"
        shared_name, is_new = cb_registry.register(original_tensor_name, cb_path, cb_fp32)

        if is_new:
            if obfuscate:
                total_bytes += _encrypt_to_tensor(cb_fp32, shared_name, writer)
            else:
                total_bytes += _quantize_to_q8(cb_fp32, shared_name, writer)

        # Store mapping
        if not obfuscate:
            writer.add_string(f"orka.cb_map.{original_tensor_name}", shared_name)

        # Indices: U16→I16 or U8→I8
        idx_path = orka_dir / stage["indices"]
        idx_dt = _index_dtype(idx_bits)
        indices = _load_raw(idx_path, idx_dt)
        signed_dt = np.int16 if idx_bits > 8 else np.int8
        indices = indices.view(signed_dt)

        idx_name = f"t.{base_name}.i{sid}" if obfuscate else f"{name}.orka.s{sid}.indices"
        writer.add_tensor(idx_name, indices)
        total_bytes += indices.nbytes

    # Block scales: FP32 → Q8_0 (or encrypt)
    if tmeta.get("scales"):
        scales_path = orka_dir / tmeta["scales"]
        scales_fp32 = _load_raw(scales_path, np.float32)
        scale_name = f"t.{base_name}.s" if obfuscate else f"{name}.orka.scales"
        if obfuscate:
            total_bytes += _encrypt_to_tensor(scales_fp32, scale_name, writer)
        else:
            total_bytes += _quantize_to_q8(scales_fp32, scale_name, writer)

    # Salient outliers
    if tmeta.get("salient"):
        sal = tmeta["salient"]

        # Salient indices: auto-downcast to smallest signed int that fits
        sal_idx_path = orka_dir / sal["indices"]
        sal_idx_u32 = _load_raw(sal_idx_path, np.uint32)
        max_val = sal_idx_u32.max()
        if max_val <= 127:
            sal_idx = sal_idx_u32.astype(np.int8)
        elif max_val <= 32767:
            sal_idx = sal_idx_u32.astype(np.int16)
        else:
            sal_idx = sal_idx_u32.view(np.int32)

        sal_idx_name = f"t.{base_name}.x" if obfuscate else f"{name}.orka.salient.idx"
        writer.add_tensor(sal_idx_name, sal_idx)
        total_bytes += sal_idx.nbytes

        # Salient values: FP32 to Q8_0 or encrypt
        sal_val_path = orka_dir / sal["weights"]
        _, sal_val_fp32 = _read_salient(
            sal_idx_path,
            sal_val_path,
            sal.get("indices_dtype", "uint32"),
            sal.get("weights_dtype", "float32"),
        )
        sal_val_name = f"t.{base_name}.y" if obfuscate else f"{name}.orka.salient.val"
        if obfuscate:
            total_bytes += _encrypt_to_tensor(sal_val_fp32, sal_val_name, writer)
        else:
            total_bytes += _quantize_to_q8(sal_val_fp32, sal_val_name, writer)

    return total_bytes


# ──────────────────────────────────────────────────────────────────────
#  Add passthrough tensors (norms, biases)
# ──────────────────────────────────────────────────────────────────────

def _add_passthrough_tensors(writer: GGUFWriter, orka_dir: Path) -> int:
    pp_path = orka_dir / "passthrough.safetensors"
    total_bytes = 0
    if not pp_path.exists():
        return 0

    for tname, tensor in _load_tensors(pp_path):
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.to(torch.float32).numpy()
        except ImportError:
            pass
        if not isinstance(tensor, np.ndarray):
            tensor = np.asarray(tensor, dtype=np.float32)
        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32)
        writer.add_tensor(tname, tensor)
        total_bytes += tensor.nbytes

    return total_bytes


# ──────────────────────────────────────────────────────────────────────
#  Tokenizer
# ──────────────────────────────────────────────────────────────────────

def _write_tokenizer(writer: GGUFWriter, source_dir: Path) -> None:
    tokenizer_json = source_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        print("  Warning: tokenizer.json not found, skipping tokenizer metadata")
        return

    with open(tokenizer_json) as f:
        tok = json.load(f)

    model_type = tok.get("model", {}).get("type", "BPE")
    writer.add_string("tokenizer.ggml.model", model_type.lower())

    vocab = tok.get("model", {}).get("vocab", {})
    if vocab:
        tokens = sorted(vocab.keys(), key=lambda k: vocab[k])
        token_bytes = [t.encode("utf-8", errors="replace") for t in tokens]
        writer.add_array("tokenizer.ggml.tokens", token_bytes)
        scores = [0.0] * len(tokens)
        writer.add_array("tokenizer.ggml.scores", scores)
        token_types = [1] * len(tokens)
        writer.add_array("tokenizer.ggml.token_type", token_types)
        print(f"  Wrote {len(tokens)} vocab tokens")

    merges_path = source_dir / "merges.txt"
    if merges_path.exists():
        with open(merges_path) as f:
            lines = f.readlines()
        merges = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
        writer.add_array("tokenizer.ggml.merges", merges)
        print(f"  Wrote {len(merges)} merges")

    tokenizer_config = source_dir / "tokenizer_config.json"
    if tokenizer_config.exists():
        with open(tokenizer_config) as f:
            tc = json.load(f)
        bos = tc.get("bos_token_id")
        eos = tc.get("eos_token_id")
        if bos is not None:
            writer.add_uint32("tokenizer.ggml.bos_token_id", bos)
        if eos is not None:
            writer.add_uint32("tokenizer.ggml.eos_token_id", eos)


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an Orka .orka artifact to an optimized compressed GGUF file."
    )
    parser.add_argument("artifact", help="Path to the .orka directory")
    parser.add_argument("--output", "-o", help="Output .gguf file path")
    parser.add_argument("--obfuscate", action="store_true", help="Obfuscate tensor names and encrypt codebooks/scales")
    args = parser.parse_args()

    orka_dir = Path(args.artifact)
    manifest_path = orka_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    source_path = Path(manifest.get("source", ""))
    source_dir = source_path if source_path.is_dir() else source_path.parent

    config_path = source_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        print(f"Warning: config.json not found at {config_path}")
        config = {}

    out_path = Path(args.output) if args.output else orka_dir.with_suffix(".gguf")

    print("=" * 60)
    print("  ORKA → GGUF Optimized Compressed Writer")
    print("=" * 60)
    print(f"  Source:          {orka_dir}")
    print(f"  Output:          {out_path}")
    print(f"  Tensors:         {manifest.get('tensor_count', '?')} quantized + {manifest.get('passthrough_count', '?')} passthrough")
    print(f"  Optimizations:   FP16 downcast + codebook dedup")
    print()

    writer = GGUFWriter(str(out_path), arch="llama")
    cb_registry = CodebookRegistry(args.obfuscate)

    # 1. Tokenizer
    print("Writing tokenizer...")
    if source_dir.exists():
        _write_tokenizer(writer, source_dir)

    # 2. Orka compressed tensors
    print(f"\nWriting Orka compressed tensors (FP16 + dedup, obfuscate={args.obfuscate})...")
    total_compressed_bytes = 0
    for i, tmeta in enumerate(manifest.get("tensors", [])):
        name = tmeta["name"]
        n_stages = len(tmeta.get("stages", []))
        has_salient = "salient" in tmeta and tmeta["salient"] is not None
        tb = _add_orka_tensor(writer, orka_dir, tmeta, cb_registry, args.obfuscate)
        total_compressed_bytes += tb
        print(f"  [{i+1:3d}] {name} ({n_stages}s, sal={'Y' if has_salient else 'N'}) → {tb:,} bytes")

    # 3. Passthrough tensors
    print("\nWriting passthrough tensors...")
    pp_bytes = _add_passthrough_tensors(writer, orka_dir)
    total_compressed_bytes += pp_bytes
    print(f"  Passthrough total: {pp_bytes:,} bytes")

    # 4. Metadata
    print("\nWriting model metadata...")
    if args.obfuscate:
        manifest["cb_map"] = cb_registry.get_mapping_kv()
    _write_model_metadata(writer, config, manifest, args.obfuscate)

    # 5. Dedup stats
    print(f"\nCodebook dedup: {cb_registry.dedup_count} duplicates eliminated, {cb_registry.saved_bytes:,} bytes saved (before FP16)")

    # 6. Finalize
    print("\nFinalizing GGUF file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress="bar")
    writer.close()

    # Report
    final_size = out_path.stat().st_size
    orka_dir_size = sum(f.stat().st_size for f in orka_dir.rglob("*") if f.is_file())
    print()
    print("=" * 60)
    print("  CONVERSION COMPLETE")
    print("=" * 60)
    print(f"  Output file:      {out_path}")
    print(f"  GGUF size:        {final_size:,} bytes ({final_size / 1024 / 1024:.1f} MB)")
    print(f"  .orka dir size:   {orka_dir_size:,} bytes ({orka_dir_size / 1024 / 1024:.1f} MB)")
    print(f"  Savings vs .orka: {(1 - final_size/orka_dir_size)*100:.1f}%")
    print(f"  Tensor data:      {total_compressed_bytes:,} bytes ({total_compressed_bytes / 1024 / 1024:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
