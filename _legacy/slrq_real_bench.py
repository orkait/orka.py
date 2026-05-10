"""Pure SLRQ on real Qwen3-0.6B weights (no VQ/RVQ/EM-AQ).

Tests SLRQ in isolation as the only quantization step:
  Linear / BFP / Pure SLRQ / Block-Salient SLRQ
across bit widths {2, 3, 4, 5, 6, 8} on real LLM weight distributions.
Reports effective bpw including all sidecar overhead.
"""

import os, sys, gc, math, time
from pathlib import Path

_CPU_CAP = 6
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(_CPU_CAP)

try:
    import psutil
except ImportError:
    sys.exit("ERROR: psutil missing")

import numpy as np
import orka._runtime as _rt
from orka._runtime import _apply_cpu_cap, _apply_system_ram_cap, _check_ram_cap, _stop_ram_monitor

RAM_CAP = 10.0
_apply_system_ram_cap(RAM_CAP)
_apply_cpu_cap(_CPU_CAP)
if not (_rt._MONITOR_THREAD and _rt._MONITOR_THREAD.is_alive()):
    sys.exit("ERROR: RAM monitor failed to start")

vm = psutil.virtual_memory()
print(f"[caps] RAM={RAM_CAP}GB CPU={_CPU_CAP}  used={vm.used/1024**3:.1f}GB\n")

# ──────────────────────────────────────────────────────────────────────────
# Quantizers (pure, no VQ)
# ──────────────────────────────────────────────────────────────────────────

def q_linear_global(w, bits):
    qmin, qmax = float(w.min()), float(w.max())
    levels = (1 << bits) - 1
    if qmax == qmin:
        return np.full_like(w, qmin), 0.0
    scale = (qmax - qmin) / levels
    idx = np.round((w - qmin) / scale)
    rec = idx * scale + qmin
    overhead_bits_per_w = 0.0  # global scale negligible
    return rec, overhead_bits_per_w


def q_bfp(w, bits, block_size=16):
    flat = w.reshape(-1)
    n = flat.size
    pad = (-n) % block_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    blocks = flat.reshape(-1, block_size)
    scales = np.abs(blocks).max(axis=1)
    scales_safe = np.where(scales == 0, 1e-9, scales)
    norm = blocks / scales_safe[:, None]
    levels = (1 << (bits - 1)) - 1
    if levels < 1:
        levels = 1
    q = np.round(norm * levels)
    rec_blocks = (q / levels) * scales_safe[:, None]
    rec = rec_blocks.reshape(-1)[:n].reshape(w.shape)
    # 16-bit fp16 scale per block
    overhead = 16.0 / block_size
    return rec, overhead


def q_pure_slrq(w, bits, block_size=16):
    flat = w.reshape(-1)
    n = flat.size
    pad = (-n) % block_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    blocks = flat.reshape(-1, block_size)
    max_mags = np.abs(blocks).max(axis=1)
    safe_mags = np.where(max_mags == 0, 1e-9, max_mags)
    anchors = np.exp2(np.ceil(np.log2(safe_mags))).astype(np.float32)
    norm = blocks / anchors[:, None]
    levels = (1 << (bits - 1)) - 1
    if levels < 1:
        levels = 1
    q = np.round(norm * levels)
    rec_blocks = (q / levels) * anchors[:, None]
    rec = rec_blocks.reshape(-1)[:n].reshape(w.shape)
    # power-of-2 anchor encoded in 5 bits per block (covers 2^-15..2^15)
    overhead = 5.0 / block_size
    return rec, overhead


def q_block_salient_slrq(w, bits, block_size=16):
    flat = w.reshape(-1).astype(np.float32, copy=True)
    n = flat.size
    pad = (-n) % block_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    blocks = flat.reshape(-1, block_size).copy()
    abs_blocks = np.abs(blocks)
    salient_idx = np.argmax(abs_blocks, axis=1)
    rows = np.arange(blocks.shape[0])
    salient_val = blocks[rows, salient_idx].copy()
    blocks[rows, salient_idx] = 0.0

    max_rem = np.abs(blocks).max(axis=1)
    safe = np.where(max_rem == 0, 1e-9, max_rem)
    anchors = np.exp2(np.ceil(np.log2(safe))).astype(np.float32)

    norm = blocks / anchors[:, None]
    levels = (1 << (bits - 1)) - 1
    if levels < 1:
        levels = 1
    q = np.round(norm * levels)
    rec_blocks = (q / levels) * anchors[:, None]
    rec_blocks[rows, salient_idx] = salient_val

    rec = rec_blocks.reshape(-1)[:n].reshape(w.shape)
    # overhead: 16-bit fp16 salient + ceil(log2(block_size))-bit index + 5-bit anchor, per block
    idx_bits = math.ceil(math.log2(block_size))
    overhead = (16 + idx_bits + 5) / block_size
    # body bits: bits per weight, but salient slot doesn't carry index in dense layout
    # Conservative: assume body still uses N bits per weight (dense storage).
    return rec, overhead


# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────

def cosine(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def sqnr(a, b):
    sig = float(np.sum(a * a))
    noise = float(np.sum((a - b) ** 2))
    if noise == 0:
        return float("inf")
    if sig == 0:
        return 0.0
    return 10.0 * math.log10(sig / noise)


# ──────────────────────────────────────────────────────────────────────────
# Run on real Qwen3 layer weights
# ──────────────────────────────────────────────────────────────────────────

import safetensors.torch as st_t

MODEL = Path("/mnt/storage/codespace/code/orkait/graphstore/graphstore/models/Qwen3-0.6B/model.safetensors")
if not MODEL.exists():
    sys.exit(f"ERROR: model not found: {MODEL}")

print("Loading Qwen3-0.6B layer tensors...")
all_t = st_t.load_file(str(MODEL))
layer = {k: v.float().numpy() for k, v in all_t.items()
         if "layers." in k and v.ndim == 2 and min(v.shape) >= 64}
del all_t; gc.collect()
print(f"Loaded {len(layer)} tensors  total params: {sum(v.size for v in layer.values()):,}\n")

BLOCK = 16
BIT_WIDTHS = [2, 3, 4, 5, 6, 8]
METHODS = [
    ("Linear (global)",       q_linear_global,       lambda b: float(b)),
    ("BFP (block-fp)",        q_bfp,                 lambda b: b + 16/BLOCK),
    ("Pure SLRQ (pow2)",      q_pure_slrq,           lambda b: b + 5/BLOCK),
    ("Block-Salient SLRQ",    q_block_salient_slrq,  lambda b: b + (16+math.ceil(math.log2(BLOCK))+5)/BLOCK),
]

# Aggregate over all tensors via SSE / signal-energy accumulation (proper SQNR)
print(f"{'Method':<22} {'bits':>4} {'eff_bpw':>8} {'cosine':>10} {'SQNR(dB)':>10} {'time':>7}")
print("-" * 80)

for bits in BIT_WIDTHS:
    for label, fn, bpw_fn in METHODS:
        _check_ram_cap()
        t0 = time.time()
        sse_total = 0.0
        sig_total = 0.0
        dot_total = 0.0
        a_norm_sq = 0.0
        b_norm_sq = 0.0
        for name, w in layer.items():
            if label == "Linear (global)":
                rec, _ = fn(w, bits)
            else:
                rec, _ = fn(w, bits, BLOCK)
            diff = w - rec
            sse_total += float(np.sum(diff * diff))
            sig_total += float(np.sum(w * w))
            dot_total += float(np.sum(w * rec))
            a_norm_sq += float(np.sum(w * w))
            b_norm_sq += float(np.sum(rec * rec))
            del rec, diff
        cos_agg = dot_total / (math.sqrt(a_norm_sq) * math.sqrt(b_norm_sq) + 1e-12)
        sqnr_agg = 10.0 * math.log10(sig_total / sse_total) if sse_total > 0 else float("inf")
        eff_bpw = bpw_fn(bits)
        elapsed = time.time() - t0
        print(f"{label:<22} {bits:>4} {eff_bpw:>8.2f} {cos_agg:>10.6f} {sqnr_agg:>10.2f} {elapsed:>6.1f}s",
              flush=True)
        gc.collect()
    print()

_stop_ram_monitor()
print("Done.")
