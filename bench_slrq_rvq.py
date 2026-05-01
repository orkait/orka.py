"""Benchmark: SLRQ + rvq-16-8-8 vs lower-bpw configs.

Resource limits enforced at top — script hard-aborts if psutil missing or
RAM headroom insufficient. CPU env vars set before any numpy/scipy import.
"""

import os
import sys

# CPU cap pre-import so BLAS thread pools obey it at load time.
_CPU_CAP = 8
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = str(_CPU_CAP)

import gc
import json
import math
import tempfile
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    print("ERROR: psutil required for RAM enforcement. Install it first.", file=sys.stderr)
    sys.exit(1)

import numpy as np

from orka import pack_checkpoint, verify_artifact
from orka._runtime import (
    _apply_cpu_cap,
    _apply_system_ram_cap,
    _check_ram_cap,
    _stop_ram_monitor,
)

# ── Enforce limits — hard abort if they cannot be applied ────────────────────

RAM_CAP_GB = 14.0
MIN_FREE_GB = 6.0
CPU_CAP     = 8

def _enforce_and_verify():
    vm = psutil.virtual_memory()
    avail_gb = vm.available / 1024 ** 3
    used_gb  = vm.used      / 1024 ** 3

    print(f"[pre-check] RAM used={used_gb:.1f}GB  avail={avail_gb:.1f}GB  cap={RAM_CAP_GB}GB")

    if avail_gb < MIN_FREE_GB:
        print(f"ERROR: only {avail_gb:.1f} GB free, need {MIN_FREE_GB} GB minimum. Aborting.", file=sys.stderr)
        sys.exit(1)

    if RAM_CAP_GB >= (vm.total / 1024 ** 3):
        print("ERROR: RAM cap must be below total RAM. Aborting.", file=sys.stderr)
        sys.exit(1)

    _apply_system_ram_cap(RAM_CAP_GB)
    _apply_cpu_cap(CPU_CAP)

    # Verify monitor thread started (psutil path only).
    from orka._runtime import _MONITOR_THREAD
    if _MONITOR_THREAD is None or not _MONITOR_THREAD.is_alive():
        print("ERROR: RAM monitor thread did not start. Aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"[limits OK] RAM cap={RAM_CAP_GB}GB  CPU cap={CPU_CAP} threads  monitor=alive")
    print()

_enforce_and_verify()

# ── Test data — use safetensors (binary), not JSON ───────────────────────────
# JSON lists of floats are ~28x larger in memory than numpy arrays.

try:
    import safetensors.numpy as st_np
except ImportError:
    print("ERROR: safetensors required. pip install safetensors", file=sys.stderr)
    _stop_ram_monitor()
    sys.exit(1)

rng = np.random.default_rng(42)

def make_weight(rows: int = 512, cols: int = 512) -> np.ndarray:
    w = (rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)
    n_out = int(rows * cols * 0.005)
    w[rng.integers(rows, size=n_out), rng.integers(cols, size=n_out)] = (
        rng.standard_normal(n_out) * 0.5
    ).astype(np.float32)
    return w

# 512×512 → 32768 vectors. Sufficient for k≤256 (128x ratio).
# For k=65536 we need 131072+ vectors → use 1024×512 = 524288 elements → 65536 vectors exactly.
# Use 1024×512 so k=65536 ratio = 1x (borderline) — note this to user.
ROWS, COLS = 1024, 512
n_vecs = ROWS * COLS // 8
print(f"Tensor shape:   [{ROWS}, {COLS}]  params={ROWS*COLS:,}")
print(f"Vectors/tensor: {n_vecs:,}  (group_size=8)")
print(f"k=256   ratio:  {n_vecs//256}x  ✓")
print(f"k=65536 ratio:  {n_vecs//65536}x  (1x = borderline; stage-1 may partially memorize)")
print()

_check_ram_cap()

# ── Base config ───────────────────────────────────────────────────────────────

BASE = dict(
    iterations    = 20,
    backend       = "numpy",
    sample_vectors= 32768,
    block_scale_size = 32,
    normalization = "slrq-block",
    codebook_mode = "per-tensor",
    group_size    = 8,
    outlier_frac  = 0.002,
)

CONFIGS = [
    ("rvq-8-8    slrq  2bpw", dict(codebook_sizes=[256, 256])),
    ("rvq-8-8-8  slrq  3bpw", dict(codebook_sizes=[256, 256, 256])),
    ("rvq-16-8   slrq  3bpw", dict(codebook_sizes=[65536, 256])),
    ("rvq-16-8-8 slrq  4bpw", dict(codebook_sizes=[65536, 256, 256])),
]

# ── Runner ────────────────────────────────────────────────────────────────────

print(f"{'Config':<28}  {'bpw':>4}  {'cosine':>8}  {'rel_rmse':>10}  {'time':>7}")
print("-" * 66)

try:
    for label, extra in CONFIGS:
        _check_ram_cap()

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "model.safetensors"
            out = Path(tmp) / "model.orka"

            # Write binary safetensors — no Python list overhead.
            tensors = {
                "model.layers.0.self_attn.q_proj.weight": make_weight(ROWS, COLS),
                "model.layers.0.mlp.down_proj.weight":    make_weight(ROWS, COLS),
            }
            st_np.save_file(tensors, str(src))
            del tensors
            gc.collect()

            _check_ram_cap()

            t0 = time.time()
            m  = pack_checkpoint(src, out, **BASE, **extra)
            elapsed = time.time() - t0

            v   = verify_artifact(out)
            cb  = extra["codebook_sizes"]
            bpw = sum(math.ceil(math.log2(k)) for k in cb) / BASE["group_size"]

            vm = psutil.virtual_memory()
            print(
                f"{label:<28}  {bpw:>4.1f}  "
                f"{v['cosine_similarity']:>8.6f}  "
                f"{v['relative_rmse']:>10.2e}  "
                f"{elapsed:>6.1f}s  "
                f"[RAM {vm.used/1024**3:.1f}GB]",
                flush=True,
            )

        gc.collect()

finally:
    _stop_ram_monitor()
    print("\nDone. RAM monitor stopped.")
