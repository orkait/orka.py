"""Benchmark Pareto-optimal configs on Qwen3-0.6B (real LLM weights).

Tests best config per bpw level. Uses CUDA when available, falls back to numpy.
RAM/CPU caps enforced. Skips embed_tokens/lm_head (151K-row vocab projections).
"""

import os, sys, gc, math, tempfile, time
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
from orka import pack_checkpoint, verify_artifact

try:
    import torch as _torch
    _CUDA = _torch.cuda.is_available()
except ImportError:
    _CUDA = False

GPU_FRAC = 0.70
if _CUDA:
    _torch.cuda.set_per_process_memory_fraction(GPU_FRAC)

BACKEND = "torch" if _CUDA else "numpy"
DEVICE  = "cuda"  if _CUDA else "cpu"
print(f"[backend] {BACKEND}  device={DEVICE}  {'GPU: ' + _torch.cuda.get_device_name(0) if _CUDA else 'CPU only'}")

RAM_CAP = 10.0
_apply_system_ram_cap(RAM_CAP)
_apply_cpu_cap(_CPU_CAP)
if not (_rt._MONITOR_THREAD and _rt._MONITOR_THREAD.is_alive()):
    sys.exit("ERROR: RAM monitor failed to start")

vm = psutil.virtual_memory()
print(f"[limits] RAM cap={RAM_CAP}GB  CPU={_CPU_CAP}  used={vm.used/1024**3:.1f}GB  avail={vm.available/1024**3:.1f}GB\n")

MODEL = Path("/mnt/storage/codespace/code/orkait/graphstore/graphstore/models/Qwen3-0.6B/model.safetensors")
if not MODEL.exists():
    sys.exit(f"ERROR: model not found at {MODEL}")

import tempfile as _tf
import safetensors.torch as _st

print("Loading model weights...")
_all = _st.load_file(str(MODEL))
_layer_tensors = {k: v.float() for k, v in _all.items()
                  if "layers." in k and v.ndim == 2 and min(v.shape) >= 64}
del _all; gc.collect()

n_layer = len(_layer_tensors)
n_params = sum(v.numel() for v in _layer_tensors.values())
_filtered_dir = _tf.mkdtemp()
_filtered_path = Path(_filtered_dir) / "qwen3_layers.safetensors"
_st.save_file(_layer_tensors, str(_filtered_path))
del _layer_tensors; gc.collect()

print(f"Layer tensors: {n_layer}  params: {n_params:,}  (skipped embed_tokens + lm_head)")
vm2 = psutil.virtual_memory()
print(f"RAM after prep: {vm2.used/1024**3:.1f}GB used\n")
MODEL = _filtered_path

CONFIGS = [
    # label,                         stages,              gs, blk, samp,  itr,  norm,           rotation
    ("1bpw  vq-8     gs8 blk16",    [256],               8,  16,  32768, 20,   "slrq-block",   "none"),
    ("2bpw  rvq-8-8  gs8 blk16",    [256,256],           8,  16,  32768, 20,   "slrq-block",   "none"),
    ("3bpw  rvq-8-8-8 gs8 blk16",   [256,256,256],       8,  16,  32768, 20,   "slrq-block",   "none"),
    ("4bpw  rvq-8-8-8-8 gs8 blk16", [256,256,256,256],   8,  16,  32768, 20,   "slrq-block",   "none"),
]

print(f"{'Config':<36}  {'bpw':>4}  {'cosine':>8}  {'rel_rmse':>10}  {'time':>8}  RAM")
print("-" * 84)

try:
    for label, stages, gs, blk, samp, itr, norm, rot in CONFIGS:
        _check_ram_cap()
        bpw = sum(math.ceil(math.log2(k)) for k in stages) / gs

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "model.orka"

            t0 = time.time()
            pack_checkpoint(
                MODEL, out,
                codebook_sizes=stages,
                group_size=gs,
                block_scale_size=blk,
                normalization=norm,
                outlier_frac=0.002,
                iterations=itr,
                sample_vectors=samp,
                backend=BACKEND,
                device=DEVICE,
                codebook_mode="per-tensor",
                rotation=rot,
            )
            elapsed = time.time() - t0
            v = verify_artifact(out)

        ram = psutil.virtual_memory().used / 1024**3
        print(
            f"{label:<36}  {bpw:>4.1f}  "
            f"{v['cosine_similarity']:>8.6f}  {v['relative_rmse']:>10.2e}  "
            f"{elapsed:>7.1f}s  {ram:.1f}GB",
            flush=True,
        )
        gc.collect()

finally:
    _stop_ram_monitor()
    print("\nDone.")
