import os, sys, gc, tempfile, time
from pathlib import Path

for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "8"

import orka._runtime as _rt
_rt._apply_system_ram_cap(22.0)
_rt._apply_cpu_cap(8)

import torch, safetensors.torch as st, gc
from orka import pack_checkpoint, verify_artifact

MODEL = Path("/mnt/storage/codespace/code/orkait/graphstore/graphstore/models/Qwen3-0.6B/model.safetensors")
OUT   = Path("results/qwen3-3bpw-slrq.orka")

# filter to layer tensors only (skip embed/lm_head)
print("Loading weights...")
all_t = st.load_file(str(MODEL))
layer_t = {k: v.float() for k,v in all_t.items()
           if "layers." in k and v.ndim == 2 and min(v.shape) >= 64}
del all_t; gc.collect()
print(f"  {len(layer_t)} tensors  {sum(v.numel() for v in layer_t.values()):,} params")

tmp_dir = tempfile.mkdtemp()
src = Path(tmp_dir) / "qwen3_layers.safetensors"
st.save_file(layer_t, str(src))
del layer_t; gc.collect()

print("Packing 3bpw (cuda)...")
t0 = time.time()
pack_checkpoint(
    src, OUT,
    codebook_sizes=[256, 256, 256],
    group_size=8,
    block_scale_size=16,
    normalization="slrq-block",
    outlier_frac=0.002,
    iterations=20,
    sample_vectors=32768,
    backend="torch",
    device="cuda",
    codebook_mode="per-tensor",
)
elapsed = time.time() - t0
print(f"Pack done in {elapsed:.1f}s")

print("Verifying...")
v = verify_artifact(OUT)
print(f"  cosine : {v['cosine_similarity']:.6f}")
print(f"  sqnr   : {v['sqnr']:.2f} dB")
print(f"  rel_rmse: {v['relative_rmse']:.4f}")

_rt._stop_ram_monitor()
