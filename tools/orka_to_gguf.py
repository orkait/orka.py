#!/usr/bin/env python3
"""
Custom GGUF writer for Orka compressed models.

Packs the raw compressed representation (codebooks, indices, scales, salient
outliers) directly into a single GGUF file *without* decompressing to FP16/FP32.

All on-disk sidecars are read through the canonical orka._format readers, so
the writer handles every v2 format detail correctly: bit-packed (non-byte-
aligned) and zlib-encoded index streams, fp16/int8 codebooks, and fp16 scales.
Reading them with a plain np.fromfile (the previous approach) silently produced
garbage for any of those - the common case on modern artifacts.

Optimizations applied:
  1. Codebook deduplication - identical codebooks are written once and referenced
     by a shared tensor name.  Per-tensor metadata maps to the shared name.
  2. Q8_0 / FP16 downcast - codebooks, block scales, and salient values are
     stored compactly, with negligible quality loss.

Tensor naming convention inside the GGUF:
    <weight_name>.orka.s<N>.codebook   - codebook  [codebook_size, group_size]
      (or shared: orka.shared_cb.<hash>)
    <weight_name>.orka.s<N>.indices    - I16/I8 indices  [vector_count]
    <weight_name>.orka.scales          - block scales
    <weight_name>.orka.salient.idx     - salient weight indices
    <weight_name>.orka.salient.val     - salient weight values

GGML has no native unsigned-integer tensor type, so VQ indices are stored as
signed I8/I16 carrying the unsigned bit pattern; readers reinterpret them as
unsigned (uint8/uint16). The bit pattern round-trips exactly for all index
values, including those >= 2^15.

Non-quantized tensors (norms, biases) are stored as regular FP32 tensors.
"""

from __future__ import annotations

import argparse
import json
import math
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
from orka._format import (
    _read_codebook,
    _read_float_vector,
    _read_indices,
    _read_salient,
)

# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    """Fast MD5 hash of a file's contents."""
    import hashlib

    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


# ──────────────────────────────────────────────────────────────────────
#  GGUF Metadata
# ──────────────────────────────────────────────────────────────────────

def _write_model_metadata(writer: GGUFWriter, config: dict, manifest: dict) -> None:
    """Write standard LLM metadata KV pairs expected by GGUF readers."""
    writer.add_string("general.name", f"orka-{config.get('model_type', 'llama')}-{config.get('hidden_size', 0)}")
    writer.add_string("general.description", "Orka RVQ-compressed model (optimized)")
    writer.add_string("general.file_type", "orka-rvq-optimized")

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

    def __init__(self):
        self._hash_to_name: dict[str, str] = {}  # file_hash -> shared tensor name
        self._mappings: dict[str, str] = {}        # original tensor name -> shared tensor name
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
        shared_name = f"orka.shared_cb.{h}"
        self._hash_to_name[h] = shared_name
        self._mappings[tensor_name] = shared_name
        return shared_name, True


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
    fp16 = data_fp32.astype(np.float16).flatten()
    writer.add_tensor(tensor_name, fp16)
    return fp16.nbytes


def _add_orka_tensor(
    writer: GGUFWriter, orka_dir: Path, tmeta: dict, cb_registry: CodebookRegistry
) -> int:
    """Add all compressed sub-tensors for one Orka weight. Returns total bytes written."""
    name = tmeta["name"]
    group_size = int(tmeta["group_size"])
    padded_values = int(tmeta["padded_values"])
    total_bytes = 0

    # Per-stage codebooks and indices
    for stage in tmeta.get("stages", []):
        sid = stage["stage"]
        idx_bits = int(stage["index_bits"])
        s_group_size = int(stage.get("group_size", group_size))
        s_index_count = math.ceil(padded_values / s_group_size)

        # Codebook: canonical reader handles fp16/int8/f32. Dedup + Q8_0.
        cb_path = orka_dir / stage["codebook"]
        cb_fp32 = _read_codebook(cb_path, s_group_size, stage.get("codebook_dtype", "float32"))
        original_tensor_name = f"{name}.orka.s{sid}.codebook"
        shared_name, is_new = cb_registry.register(original_tensor_name, cb_path, cb_fp32)
        if is_new:
            total_bytes += _quantize_to_q8(cb_fp32, shared_name, writer)
        writer.add_string(f"orka.cb_map.{original_tensor_name}", shared_name)

        # Indices: decode bit-packing + entropy coding via the canonical reader,
        # then store the integer values as GGML signed ints (bit-identical to
        # unsigned; readers reinterpret as unsigned - GGML has no uint tensor).
        idx_path = orka_dir / stage["indices"]
        unpacked = _read_indices(
            idx_path,
            idx_bits,
            s_index_count,
            packed=bool(stage.get("packed", idx_bits % 8 != 0)),
            encoding=stage.get("encoding", "raw"),
        )
        unsigned_dt = np.uint16 if idx_bits > 8 else np.uint8
        signed_dt = np.int16 if idx_bits > 8 else np.int8
        indices = np.asarray(unpacked, dtype=unsigned_dt).view(signed_dt)
        writer.add_tensor(f"{name}.orka.s{sid}.indices", indices)
        total_bytes += indices.nbytes

    # Block scales (stored fp16 on disk) -> Q8_0
    if tmeta.get("scales"):
        scales_fp32 = _read_float_vector(
            orka_dir / tmeta["scales"],
            int(tmeta["scale_count"]),
            tmeta.get("scale_dtype") or "float32",
        )
        total_bytes += _quantize_to_q8(scales_fp32, f"{name}.orka.scales", writer)

    # Salient outliers (slrq)
    if tmeta.get("salient"):
        sal = tmeta["salient"]
        sal_idx, sal_val = _read_salient(
            orka_dir / sal["indices"],
            orka_dir / sal["weights"],
            sal.get("indices_dtype", "uint32"),
            sal.get("weights_dtype", "float32"),
        )
        # Salient indices are local block offsets (small); a signed int that fits
        # round-trips exactly.
        max_val = int(sal_idx.max()) if sal_idx.size else 0
        if max_val <= 127:
            sal_idx_store = sal_idx.astype(np.int8)
        elif max_val <= 32767:
            sal_idx_store = sal_idx.astype(np.int16)
        else:
            sal_idx_store = sal_idx.astype(np.int32)
        writer.add_tensor(f"{name}.orka.salient.idx", sal_idx_store)
        total_bytes += sal_idx_store.nbytes
        total_bytes += _quantize_to_q8(np.asarray(sal_val, dtype=np.float32), f"{name}.orka.salient.val", writer)

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
        except Exception:  # torch absent, or broken native install (OSError)
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
    print("  ORKA -> GGUF Optimized Compressed Writer")
    print("=" * 60)
    print(f"  Source:          {orka_dir}")
    print(f"  Output:          {out_path}")
    print(f"  Tensors:         {manifest.get('tensor_count', '?')} quantized + {manifest.get('passthrough_count', '?')} passthrough")
    print(f"  Optimizations:   Q8_0/FP16 downcast + codebook dedup")
    print()

    writer = GGUFWriter(str(out_path), arch="llama")
    cb_registry = CodebookRegistry()

    # 1. Tokenizer
    print("Writing tokenizer...")
    if source_dir.exists():
        _write_tokenizer(writer, source_dir)

    # 2. Orka compressed tensors
    print("\nWriting Orka compressed tensors (Q8_0/FP16 + dedup)...")
    total_compressed_bytes = 0
    for i, tmeta in enumerate(manifest.get("tensors", [])):
        name = tmeta["name"]
        n_stages = len(tmeta.get("stages", []))
        has_salient = "salient" in tmeta and tmeta["salient"] is not None
        tb = _add_orka_tensor(writer, orka_dir, tmeta, cb_registry)
        total_compressed_bytes += tb
        print(f"  [{i+1:3d}] {name} ({n_stages}s, sal={'Y' if has_salient else 'N'}) -> {tb:,} bytes")

    # 3. Passthrough tensors
    print("\nWriting passthrough tensors...")
    pp_bytes = _add_passthrough_tensors(writer, orka_dir)
    total_compressed_bytes += pp_bytes
    print(f"  Passthrough total: {pp_bytes:,} bytes")

    # 4. Metadata
    print("\nWriting model metadata...")
    _write_model_metadata(writer, config, manifest)

    # 5. Dedup stats
    print(f"\nCodebook dedup: {cb_registry.dedup_count} duplicates eliminated, {cb_registry.saved_bytes:,} bytes saved (before Q8_0)")

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
    if orka_dir_size:
        print(f"  Savings vs .orka: {(1 - final_size/orka_dir_size)*100:.1f}%")
    print(f"  Tensor data:      {total_compressed_bytes:,} bytes ({total_compressed_bytes / 1024 / 1024:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
