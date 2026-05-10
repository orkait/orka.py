"""Two-pass quality improvement benchmark.

Pass 1: Quick 2bpw scan -> per-tensor SQNR
Pass 2: Mixed precision (attn=3bpw, mlp=2bpw) + passthrough for worst tensors
Compares against uniform 2bpw and 3bpw baselines.
"""

import os, sys, gc, math, json, tempfile, time
from pathlib import Path

_CPU_CAP = 6
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
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
print(f"[backend] {BACKEND}  device={DEVICE}")

RAM_CAP = 10.0
_apply_system_ram_cap(RAM_CAP)
_apply_cpu_cap(_CPU_CAP)
if not (_rt._MONITOR_THREAD and _rt._MONITOR_THREAD.is_alive()):
    sys.exit("ERROR: RAM monitor failed to start")

vm = psutil.virtual_memory()
print(f"[limits] RAM cap={RAM_CAP}GB  CPU={_CPU_CAP}  "
      f"used={vm.used/1024**3:.1f}GB  avail={vm.available/1024**3:.1f}GB\n")

MODEL_DIR = Path("/mnt/storage/codespace/code/orkait/graphstore/graphstore/models/Qwen3-0.6B")
MODEL = MODEL_DIR / "model.safetensors"
if not MODEL.exists():
    sys.exit(f"ERROR: model not found at {MODEL}")

import tempfile as _tf
import safetensors.torch as _st

print("Loading + filtering model weights...")
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
FILTERED = _filtered_path

print(f"Layer tensors: {n_layer}  params: {n_params:,}")
vm2 = psutil.virtual_memory()
print(f"RAM after prep: {vm2.used/1024**3:.1f}GB\n")


def _pack_and_verify(src, label, bpw_approx, **kwargs):
    _check_ram_cap()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "model.orka"
        t0 = time.time()
        pack_checkpoint(src, out, backend=BACKEND, device=DEVICE,
                        outlier_frac=0.002, normalization="slrq-block",
                        codebook_mode="per-tensor", **kwargs)
        elapsed = time.time() - t0
        v = verify_artifact(out)
        manifest = json.loads((out / "manifest.json").read_text())
    ram = psutil.virtual_memory().used / 1024**3
    print(f"{label:<46}  bpw~{bpw_approx:.1f}  "
          f"cosine={v['cosine_similarity']:.6f}  rel_rmse={v['relative_rmse']:.2e}  "
          f"sqnr={v['sqnr']:.1f}dB  {elapsed:.0f}s  {ram:.1f}GB", flush=True)
    gc.collect()
    return v, manifest


try:
    print(f"{'Config':<46}  {'bpw':>6}  {'cosine':>10}  {'rel_rmse':>10}  {'sqnr':>8}  {'time':>6}  RAM")
    print("-" * 110)

    # ── Baseline: uniform 2bpw ────────────────────────────────────────────────
    v_base2, manifest_base2 = _pack_and_verify(
        FILTERED, "baseline  2bpw uniform (rvq-8-8 gs8)", 2.0,
        codebook_sizes=[256, 256], group_size=8, block_scale_size=16,
        sample_vectors=32768, iterations=20,
    )

    # ── Build SQNR sensitivity map from Pass 1 ───────────────────────────────
    SQNR_SKIP_THRESHOLD = 6.0  # tensors below this -> keep FP16
    sensitive_layers = []
    sqnr_by_tensor = {}
    for t in manifest_base2.get("tensors", []):
        sqnr_val = t.get("sqnr", float("inf"))
        name_no_weight = t["name"].replace(".weight", "")
        sqnr_by_tensor[name_no_weight] = sqnr_val
        if sqnr_val < SQNR_SKIP_THRESHOLD:
            sensitive_layers.append({"layer": name_no_weight, "loss_delta": 2.0})

    sensitivity_map = {"layers": sensitive_layers} if sensitive_layers else None
    n_sensitive = len(sensitive_layers)
    print(f"\n[sensitivity] {n_sensitive} tensors SQNR < {SQNR_SKIP_THRESHOLD}dB -> passthrough FP16")
    if sensitive_layers:
        worst = sorted(sensitive_layers, key=lambda x: sqnr_by_tensor.get(x["layer"], 0))[:5]
        for w in worst:
            print(f"  {w['layer']}: {sqnr_by_tensor[w['layer']]:.1f}dB")
    print()

    # ── Baseline: uniform 3bpw ────────────────────────────────────────────────
    v_base3, _ = _pack_and_verify(
        FILTERED, "baseline  3bpw uniform (rvq-8-8-8 gs8)", 3.0,
        codebook_sizes=[256, 256, 256], group_size=8, block_scale_size=16,
        sample_vectors=32768, iterations=20,
    )

    # ── Mixed precision: attn=3bpw, mlp=2bpw ─────────────────────────────────
    # Effective bpw = ceil(log2(256)) * stages / group_size
    # attn(3 stages): 8*3/8 = 3bpw  mlp(2 stages): 8*2/8 = 2bpw
    # avg ~ 2.4bpw for this model (196 tensors, attn:mlp ratio ~4:3 per layer)
    v_mix, _ = _pack_and_verify(
        FILTERED, "mixed  attn=3bpw mlp=2bpw other=1bpw", 2.4,
        family_stages_map={
            "attention": [256, 256, 256],
            "mlp":       [256, 256],
            "other":     [256],
        },
        group_size=8, block_scale_size=16,
        sample_vectors=32768, iterations=20,
    )

    # ── Mixed + sensitivity passthrough ──────────────────────────────────────
    if sensitivity_map is not None:
        v_mix_sens, _ = _pack_and_verify(
            FILTERED, f"mixed+sensitivity ({n_sensitive} FP16 tensors)", 2.4,
            family_stages_map={
                "attention": [256, 256, 256],
                "mlp":       [256, 256],
                "other":     [256],
            },
            group_size=8, block_scale_size=16,
            sample_vectors=32768, iterations=20,
            sensitivity_map=sensitivity_map,
        )
    else:
        print("(no tensors below SQNR threshold - sensitivity passthrough skipped)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Delta vs uniform 2bpw baseline ──")
    for label, v in [
        ("uniform 3bpw",   v_base3),
        ("mixed 2.4bpw",   v_mix),
    ]:
        d_cos  = v["cosine_similarity"] - v_base2["cosine_similarity"]
        d_sqnr = v["sqnr"] - v_base2["sqnr"]
        print(f"  {label:<28}  cosine delta={d_cos:+.6f}  sqnr delta={d_sqnr:+.1f}dB")

finally:
    _stop_ram_monitor()
    print("\nDone.")
