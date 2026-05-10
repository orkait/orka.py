"""Hadamard hypothesis test on SmolLM2-135M.

Tests rotation effect WITHOUT slrq's salient extraction interference:
  A. norm=none           rotation=none      (baseline pure VQ)
  B. norm=none           rotation=hadamard  (does Hadamard help bare VQ?)
  C. norm=block-max      rotation=none
  D. norm=block-max      rotation=hadamard  (Hadamard + plain block scales)
  E. norm=slrq-block     rotation=none      (current orka default)
  F. norm=slrq-block     rotation=hadamard  (current default + Hadamard)
"""

import os, sys, gc, json, tempfile, time
from pathlib import Path

_CPU_CAP = 6
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(_CPU_CAP)

import psutil
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

RAM_CAP = 22.0  # baseline ~10GB + ~12GB process budget for Qwen3 (under 28GB hard ceiling)
_apply_system_ram_cap(RAM_CAP)
_apply_cpu_cap(_CPU_CAP)
print(f"[caps] RAM={RAM_CAP}GB CPU={_CPU_CAP} GPU={int(GPU_FRAC*100)}% backend={BACKEND}\n")

import safetensors.torch as st_t
import tempfile as _tf

# Switchable model path - default Qwen3 for bigger Hadamard blocks (3072 -> H_1024)
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/storage/codespace/code/orkait/graphstore/graphstore/models/Qwen3-0.6B"
MODEL_DIR = Path(MODEL_PATH)
SRC_FULL = MODEL_DIR / "model.safetensors"
print(f"[model] {MODEL_DIR.name}")

print(f"Filtering {MODEL_DIR.name} layer weights...")
all_t = st_t.load_file(str(SRC_FULL))
layer = {k: v.float() for k, v in all_t.items()
         if "layers." in k and v.ndim == 2 and min(v.shape) >= 64}
del all_t; gc.collect()

filt_dir = _tf.mkdtemp()
SRC = Path(filt_dir) / "smollm2_layers.safetensors"
st_t.save_file(layer, str(SRC))
del layer; gc.collect()
print(f"Source: {SRC}\n")


def _pack(label, normalization, rotation):
    _check_ram_cap()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "model.orka"
        t0 = time.time()
        kwargs = dict(
            codebook_sizes=[256, 256],
            group_size=8,
            block_scale_size=16,
            normalization=normalization,
            iterations=15,
            sample_vectors=16384,
            backend=BACKEND, device=DEVICE,
            codebook_mode="per-tensor",
            rotation=rotation,
        )
        if normalization == "slrq-block":
            kwargs["outlier_frac"] = 0.002
        pack_checkpoint(SRC, out, **kwargs)
        elapsed = time.time() - t0
        v = verify_artifact(out)
    ram = psutil.virtual_memory().used / 1024**3
    print(f"  {label:<32}  cosine={v['cosine_similarity']:.6f}  "
          f"sqnr={v['sqnr']:6.2f}dB  time={elapsed:5.1f}s  RAM={ram:.1f}GB", flush=True)
    gc.collect()
    return v


print("Running 6 configs (2bpw RVQ, gs=8, blk=16, samp=16384, itr=15)\n")
print("-" * 95)

results = []
configs = [
    ("A. norm=none      rot=none",      "none",       "none"),
    ("B. norm=none      rot=hadamard",  "none",       "hadamard"),
    ("C. norm=block-max rot=none",      "block-max",  "none"),
    ("D. norm=block-max rot=hadamard",  "block-max",  "hadamard"),
    ("E. norm=slrq      rot=none",      "slrq-block", "none"),
    ("F. norm=slrq      rot=hadamard",  "slrq-block", "hadamard"),
]
for label, n, r in configs:
    v = _pack(label, n, r)
    results.append((label, v["cosine_similarity"], v["sqnr"]))

print("\n── Summary ──")
print(f"  {'Config':<32}  {'cosine':>10}  {'sqnr':>8}")
for label, cos, sq in results:
    print(f"  {label:<32}  {cos:>10.6f}  {sq:>7.2f}dB")

# pairwise deltas (Hadamard vs no-Hadamard at each normalization)
print("\n── Hadamard delta (B-A, D-C, F-E) ──")
for i in range(0, 6, 2):
    none_lbl, none_cos, none_sq = results[i]
    hadm_lbl, hadm_cos, hadm_sq = results[i+1]
    norm = none_lbl.split("=")[1].split()[0]
    print(f"  norm={norm:<12}  cos delta={hadm_cos - none_cos:+.6f}  sqnr delta={hadm_sq - none_sq:+.2f}dB")

_stop_ram_monitor()
print("\nDone.")
