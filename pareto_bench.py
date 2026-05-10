"""Pareto frontier: best quality at each bpw. Limits enforced hard."""

import os, sys, gc, math, tempfile, time
from pathlib import Path

_CPU_CAP = 8
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(_CPU_CAP)

try:
    import psutil
except ImportError:
    sys.exit("ERROR: psutil missing")

import numpy as np
import safetensors.numpy as st_np
from orka import pack_checkpoint, verify_artifact
from orka._runtime import (_apply_cpu_cap, _apply_system_ram_cap,
                            _check_ram_cap, _stop_ram_monitor)
import orka._runtime as _rt

# ── limits ───────────────────────────────────────────────────────────────────
RAM_CAP = 14.0
_apply_system_ram_cap(RAM_CAP)
_apply_cpu_cap(_CPU_CAP)
if not (_rt._MONITOR_THREAD and _rt._MONITOR_THREAD.is_alive()):
    sys.exit("ERROR: RAM monitor failed to start")
vm = psutil.virtual_memory()
print(f"[limits] RAM cap={RAM_CAP}GB  CPU={_CPU_CAP}  "
      f"used={vm.used/1024**3:.1f}GB  avail={vm.available/1024**3:.1f}GB\n")

# ── data: 1024×512, 65536 vectors/tensor ─────────────────────────────────────
rng = np.random.default_rng(42)
def make_w(rows=1024, cols=512):
    w = (rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)
    n = int(rows * cols * 0.005)
    w[rng.integers(rows,size=n), rng.integers(cols,size=n)] = (
        rng.standard_normal(n) * 0.5).astype(np.float32)
    return w

# ── configs ───────────────────────────────────────────────────────────────────
# bpw = sum(ceil(log2(k)) for k in stages) / group_size
configs = [
    # label,                            stages,          gs,  blk, samp,  itr,  out,   norm
    ("1.0  vq-8    gs8  blk32",        [256],            8,   32,  32768, 20,  0.002, "slrq-block"),
    ("1.0  vq-8    gs8  blk16",        [256],            8,   16,  32768, 20,  0.002, "slrq-block"),
    ("2.0  rvq-8-8 gs8  blk32",        [256,256],        8,   32,  32768, 20,  0.002, "slrq-block"),
    ("2.0  rvq-8-8 gs8  blk16",        [256,256],        8,   16,  32768, 20,  0.002, "slrq-block"),
    ("2.0  rvq-8-8 gs8  blk16 i30",    [256,256],        8,   16,  32768, 30,  0.002, "slrq-block"),
    ("2.0  rvq-8-8 gs4  blk16 (4bpw)", [256,256],        4,   16,  32768, 20,  0.002, "slrq-block"),
    ("3.0  rvq-8-8-8 gs8  blk32",      [256,256,256],    8,   32,  32768, 20,  0.002, "slrq-block"),
    ("3.0  rvq-8-8-8 gs8  blk16",      [256,256,256],    8,   16,  32768, 20,  0.002, "slrq-block"),
    ("3.0  rvq-8-8-8 gs8  blk16 i30",  [256,256,256],    8,   16,  65536, 30,  0.002, "slrq-block"),
    ("3.0  rvq-8-8-8 gs8  blk16+had",  [256,256,256],    8,   16,  32768, 20,  0.002, "slrq-block"),  # hadamard
    ("4.0  rvq-8-8-8-8 gs8 blk16",     [256,256,256,256],8,   16,  32768, 20,  0.002, "slrq-block"),
    ("4.0  rvq-8-8-8-8 gs8 blk16 i30", [256,256,256,256],8,   16,  65536, 30,  0.002, "slrq-block"),
]

use_hadamard = {"3.0  rvq-8-8-8 gs8  blk16+had"}

print(f"{'Config':<42}  {'bpw':>4}  {'cosine':>8}  {'rel_rmse':>10}  {'time':>7}  RAM")
print("-" * 90)

try:
    for label, stages, gs, blk, samp, itr, out_frac, norm in configs:
        _check_ram_cap()
        bpw = sum(math.ceil(math.log2(k)) for k in stages) / gs
        rotation = "hadamard" if label in use_hadamard else "none"

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "m.safetensors"
            out = Path(tmp) / "m.orka"
            st_np.save_file({
                "model.layers.0.self_attn.q_proj.weight": make_w(),
                "model.layers.0.mlp.down_proj.weight":    make_w(),
            }, str(src))
            gc.collect()
            _check_ram_cap()

            t0 = time.time()
            m = pack_checkpoint(src, out,
                codebook_sizes=stages, group_size=gs, block_scale_size=blk,
                normalization=norm, outlier_frac=out_frac, iterations=itr,
                sample_vectors=samp, backend="numpy", codebook_mode="per-tensor",
                rotation=rotation)
            elapsed = time.time() - t0
            v = verify_artifact(out)

        ram = psutil.virtual_memory().used / 1024**3
        print(f"{label:<42}  {bpw:>4.1f}  "
              f"{v['cosine_similarity']:>8.6f}  {v['relative_rmse']:>10.2e}  "
              f"{elapsed:>6.1f}s  {ram:.1f}GB", flush=True)
        gc.collect()
finally:
    _stop_ram_monitor()
    print("\nDone.")
