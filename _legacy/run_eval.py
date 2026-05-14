"""Pack full Qwen3-0.6B at 3bpw (mixed: attn=3, mlp=2) then run perplexity eval
against wiki_prompts.txt. Embed/lm_head pass through as FP16.
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

import orka._runtime as _rt
from orka._runtime import _apply_cpu_cap, _apply_system_ram_cap, _check_ram_cap, _stop_ram_monitor
from orka import pack_checkpoint, verify_artifact
from orka.eval import eval_artifact

try:
    import torch as _torch
    _CUDA = _torch.cuda.is_available()
except ImportError:
    _CUDA = False

GPU_FRAC = 0.70  # cap at 70% of VRAM, leave headroom for OS
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

FULL_MODEL = Path("/mnt/storage/codespace/code/orkait/graphstore/graphstore/models/Qwen3-0.6B/model.safetensors")
MODEL_DIR  = FULL_MODEL.parent
PROMPTS    = Path("/mnt/storage/codespace/code/orkait/bonsai-models/wiki_prompts.txt")
ARTIFACT   = Path("/tmp/qwen3_3bpw_mixed_artifact")

if not FULL_MODEL.exists():
    sys.exit(f"ERROR: model not found: {FULL_MODEL}")
if not PROMPTS.exists():
    sys.exit(f"ERROR: prompts not found: {PROMPTS}")

# embed_tokens and lm_head -> force passthrough (loss_delta > 1.5 threshold)
SENSITIVITY_MAP = {
    "layers": [
        {"layer": "model.embed_tokens", "loss_delta": 2.0},
        {"layer": "lm_head",            "loss_delta": 2.0},
    ]
}

try:
    # ── Pack ──────────────────────────────────────────────────────────────────
    if ARTIFACT.exists() and (ARTIFACT / "manifest.json").exists():
        print(f"[pack] Reusing existing artifact at {ARTIFACT}")
    else:
        ARTIFACT.mkdir(parents=True, exist_ok=True)
        print(f"[pack] Packing 3bpw mixed (attn=3, mlp=2) from full model...")
        _check_ram_cap()
        t0 = time.time()
        pack_checkpoint(
            FULL_MODEL, ARTIFACT,
            family_stages_map={
                "attention":  [256, 256, 256],
                "mlp":        [256, 256],
                "other":      [256],
            },
            group_size=8,
            block_scale_size=16,
            normalization="slrq-block",
            outlier_frac=0.002,
            iterations=20,
            sample_vectors=32768,
            backend=BACKEND,
            device=DEVICE,
            codebook_mode="per-tensor",
            sensitivity_map=SENSITIVITY_MAP,
        )
        elapsed = time.time() - t0
        print(f"[pack] Done in {elapsed:.0f}s")
        gc.collect()

    # ── Verify (tier-1 metrics) ───────────────────────────────────────────────
    print("\n[verify] Computing tier-1 metrics...")
    v = verify_artifact(ARTIFACT)
    print(f"  tensors verified : {v['verified_tensors']}")
    print(f"  passthrough      : {v['verified_passthrough_tensors']}")
    print(f"  cosine           : {v['cosine_similarity']:.6f}")
    print(f"  rel_rmse         : {v['relative_rmse']:.2e}")
    print(f"  sqnr             : {v['sqnr']:.1f} dB")
    gc.collect()

    # ── Eval (tier-2b: perplexity) ────────────────────────────────────────────
    OUT_JSON = Path("/tmp/qwen3_3bpw_eval.json")
    print(f"\n[eval] Running perplexity eval (100 prompts, max_length=512, device={DEVICE})...")
    _check_ram_cap()
    t1 = time.time()
    result = eval_artifact(
        artifact_dir=ARTIFACT,
        prompts_path=PROMPTS,
        out_path=OUT_JSON,
        model_dir=MODEL_DIR,
        max_prompts=100,
        max_length=512,
        device=DEVICE,
        local_files_only=True,
    )
    elapsed2 = time.time() - t1
    print(f"[eval] Done in {elapsed2:.0f}s")

    print("\n── Perplexity Results ──")
    print(f"  prompts scored       : {result['prompt_count']}")
    print(f"  tokens scored        : {result['token_count']}")
    print(f"  original perplexity  : {result['original_perplexity']:.4f}")
    print(f"  orka perplexity      : {result['orka_perplexity']:.4f}")
    print(f"  perplexity ratio     : {result['perplexity_ratio']:.4f}x")
    print(f"  loss delta           : {result['loss_delta']:+.4f}")
    print(f"\n  Full results -> {OUT_JSON}")

finally:
    _stop_ram_monitor()
    print("\nDone.")
