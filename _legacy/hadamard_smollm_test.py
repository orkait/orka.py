"""Isolated Hadamard test on SmolLM2-135M.

Pack same weights with and without block-diagonal Hadamard rotation.
Measures: per-tensor SQNR delta + aggregate cosine + total time.
No EM-AQ, no family_stages_map - isolate Hadamard effect only.
"""

import os, sys, gc, json, tempfile, time
from pathlib import Path

_CPU_CAP = 6
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(_CPU_CAP)

try:
    import psutil
except ImportError:
    sys.exit("ERROR: psutil missing")

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
print(f"[backend] {BACKEND} device={DEVICE}")

RAM_CAP = 16.0
_apply_system_ram_cap(RAM_CAP)
_apply_cpu_cap(_CPU_CAP)
if not (_rt._MONITOR_THREAD and _rt._MONITOR_THREAD.is_alive()):
    sys.exit("ERROR: RAM monitor failed to start")

vm = psutil.virtual_memory()
print(f"[caps] RAM={RAM_CAP}GB CPU={_CPU_CAP} GPU={int(GPU_FRAC*100)}%  used={vm.used/1024**3:.1f}GB\n")

# ── Filter SmolLM2 layer weights ─────────────────────────────────────────
import safetensors.torch as st_t
import tempfile as _tf

MODEL_DIR = Path("/mnt/storage/codespace/code/orkait/graphstore/graphstore/models/orka-smollm2-135m")
SRC_FULL = MODEL_DIR / "model.safetensors"

print("Loading + filtering SmolLM2-135M layer weights...")
all_t = st_t.load_file(str(SRC_FULL))
layer = {k: v.float() for k, v in all_t.items()
         if "layers." in k and v.ndim == 2 and min(v.shape) >= 64}
del all_t; gc.collect()

n_t = len(layer)
n_p = sum(v.numel() for v in layer.values())
filt_dir = _tf.mkdtemp()
SRC = Path(filt_dir) / "smollm2_layers.safetensors"
st_t.save_file(layer, str(SRC))
del layer; gc.collect()
print(f"Filtered: {n_t} layer tensors, {n_p:,} params -> {SRC}\n")

# ── Pack runs ────────────────────────────────────────────────────────────
def _pack(label, rotation):
    _check_ram_cap()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "model.orka"
        t0 = time.time()
        pack_checkpoint(
            SRC, out,
            codebook_sizes=[256, 256],   # 2bpw uniform RVQ
            group_size=8,
            block_scale_size=16,
            normalization="slrq-block",
            outlier_frac=0.002,
            iterations=20,
            sample_vectors=32768,
            backend=BACKEND, device=DEVICE,
            codebook_mode="per-tensor",
            rotation=rotation,
        )
        elapsed = time.time() - t0
        v = verify_artifact(out)
        manifest = json.loads((out / "manifest.json").read_text())
        per_tensor_sqnr = {t["name"]: t.get("sqnr", 0.0) for t in manifest["tensors"]}
    ram = psutil.virtual_memory().used / 1024**3
    print(f"  {label:<20}  cosine={v['cosine_similarity']:.6f}  "
          f"sqnr={v['sqnr']:6.2f}dB  rel_rmse={v['relative_rmse']:.2e}  "
          f"time={elapsed:5.1f}s  RAM={ram:.1f}GB")
    gc.collect()
    return v, per_tensor_sqnr

print(f"Packing 2bpw RVQ on {n_t} tensors with/without Hadamard...\n")
print("Config: rvq-8-8, gs=8, blk=16, slrq-block, samp=32768, itr=20")
print("-" * 88)

v_none, sqnr_none = _pack("rotation=none",     "none")
v_hadm, sqnr_hadm = _pack("rotation=hadamard", "hadamard")

# ── Per-tensor delta analysis ────────────────────────────────────────────
print("\n── Aggregate delta (Hadamard vs none) ──")
d_cos  = v_hadm["cosine_similarity"] - v_none["cosine_similarity"]
d_sqnr = v_hadm["sqnr"] - v_none["sqnr"]
d_rmse_pct = (v_hadm["relative_rmse"] - v_none["relative_rmse"]) / v_none["relative_rmse"] * 100
print(f"  cosine       delta = {d_cos:+.6f}")
print(f"  SQNR         delta = {d_sqnr:+.2f} dB")
print(f"  rel_rmse     delta = {d_rmse_pct:+.1f}%")

print("\n── Per-tensor SQNR delta (top 10 winners + losers) ──")
deltas = []
for name, s_none in sqnr_none.items():
    s_hadm = sqnr_hadm.get(name, 0.0)
    deltas.append((name, s_none, s_hadm, s_hadm - s_none))
deltas.sort(key=lambda x: x[3], reverse=True)

print(f"  {'Top winners':<60}  {'none':>8}  {'hadm':>8}  {'delta':>7}")
for name, sn, sh, d in deltas[:10]:
    short = name.replace("model.", "").replace(".weight", "")[:58]
    print(f"  {short:<60}  {sn:>7.2f}  {sh:>7.2f}  {d:>+7.2f}")

print(f"\n  {'Top losers (regressions)':<60}  {'none':>8}  {'hadm':>8}  {'delta':>7}")
for name, sn, sh, d in deltas[-5:]:
    short = name.replace("model.", "").replace(".weight", "")[:58]
    print(f"  {short:<60}  {sn:>7.2f}  {sh:>7.2f}  {d:>+7.2f}")

# Family breakdown
print("\n── By family ──")
families = {"q_proj": [], "k_proj": [], "v_proj": [], "o_proj": [],
            "gate_proj": [], "up_proj": [], "down_proj": []}
for name, _, _, d in deltas:
    for fam in families:
        if fam in name:
            families[fam].append(d)
            break

print(f"  {'family':<14} {'count':>6} {'avg delta dB':>14} {'min':>8} {'max':>8}")
for fam, vs in families.items():
    if not vs:
        continue
    import statistics
    avg = statistics.mean(vs)
    print(f"  {fam:<14} {len(vs):>6} {avg:>14.2f} {min(vs):>8.2f} {max(vs):>8.2f}")

_stop_ram_monitor()
print("\nDone.")
