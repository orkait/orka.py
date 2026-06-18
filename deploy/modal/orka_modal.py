"""Reusable Modal app for the Orka compiler.

Runs the full Orka recipe (allocate -> pack -> distill -> correct -> report ->
export) or any raw `orka` subcommand on a Modal GPU, persisting artifacts to a
Volume. The orka package is baked into the image, so jobs need no local state.

Examples (from repo root, with the venv's modal on PATH):

    # full recipe on a public HF model, 4bpw + int8 codebooks
    modal run deploy/modal/orka_modal.py::compress \
        --repo SupraLabs/Supra-1.5-50M-Instruct-exp --target-bpw 4 --codebook-dtype int8

    # raw passthrough: any orka subcommand
    modal run deploy/modal/orka_modal.py::raw --args "report /data/artifacts/<name>.orka"

    # list / download artifacts from the volume
    modal run deploy/modal/orka_modal.py::ls
    modal volume get orka-data artifacts/<name>.orka ./local_dir

GPU default A10G (24GB, sm_86 - supports Orka's fp32 cdist; cheaper than A100).
Override with `--gpu A100` on the function call where exposed.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "safetensors", "numpy", "scipy", "tqdm",
        "huggingface_hub", "datasets", "accelerate",
    )
    # bake the orka package into the image so `python -m orka` works
    .add_local_dir(str(REPO_ROOT / "orka"), "/root/orka", copy=True)
)

app = modal.App("orka")
data_vol = modal.Volume.from_name("orka-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("orka-hf-cache", create_if_missing=True)

VOLUMES = {"/data": data_vol, "/hf": hf_vol}
ENV = {"HF_HOME": "/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def _orka(*args: str) -> str:
    """Run `python -m orka ...`, stream output, return stdout tail."""
    cmd = [sys.executable, "-m", "orka", *[str(a) for a in args]]
    print("+ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, cwd="/root")
    if p.stdout:
        print(p.stdout, flush=True)
    if p.returncode != 0:
        print(p.stderr, flush=True)
        raise RuntimeError(f"orka {args[0]} failed (exit {p.returncode})")
    return p.stdout


def _last_json(text: str) -> dict:
    i = text.rfind("{")
    return json.loads(text[i:]) if i >= 0 else {}


@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=3 * 3600,
              env=ENV, retries=0)
def compress(
    repo: str,
    target_bpw: float = 4.0,
    codebook_dtype: str = "int8",
    distill_steps: int = 120,
    correct_rank: int = 8,
    outlier_frac: float = 0.005,
    calib_lines: int = 64,
    sample_vectors: int = 0,          # 0 -> auto-scale to GPU memory
    iterations: int = 8,
    calib_samples: int = 8192,
    hf_token: str = "",
) -> dict:
    """Full recipe: download -> allocate -> pack -> distill -> correct -> report
    -> export. Returns the report; artifact + HF export land in the volume.

    Quality knobs (sample_vectors / iterations / calib_samples) are sized up
    from the 3060/T4 defaults to exploit the A10's headroom: more codebook
    training vectors and Lloyd iterations -> tighter codebooks at the same
    bits. They cost memory + time we have free here, not quality."""
    import torch
    from huggingface_hub import login, snapshot_download

    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"=== GPU: {name} ({total_gb:.0f} GB) ===", flush=True)
    if sample_vectors <= 0:
        # ~64k per 8GB of VRAM; the k-means distance matrix is the binding cost.
        sample_vectors = int(min(524288, max(65536, total_gb * 8192)))
    print(f"=== quality knobs: sample_vectors={sample_vectors} iterations={iterations} "
          f"calib_samples={calib_samples} distill_steps={distill_steps} ===", flush=True)
    if hf_token:
        login(token=hf_token)

    model_dir = snapshot_download(
        repo, allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*", "merges*", "vocab*"]
    )
    src = next(Path(model_dir).glob("*.safetensors"))
    slug = repo.split("/")[-1]
    run = Path("/data") / slug
    run.mkdir(parents=True, exist_ok=True)
    art = run / f"{slug}-{target_bpw}bpw.orka"

    # calibration / eval corpus from wikitext
    calib = run / "calib.txt"
    eval_p = run / "eval.txt"
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        lines = [r["text"].strip() for r in ds if len(r["text"].strip()) > 200]
        calib.write_text("\n".join(lines[:calib_lines]))
        eval_p.write_text("\n".join(lines[60:72]))
    except Exception as exc:
        print(f"wikitext fetch failed ({exc}); fallback prompts", flush=True)
        fb = ["The history of science is long and storied." * 4] * 32
        calib.write_text("\n".join(fb))
        eval_p.write_text("\n".join(fb[:11]))

    alloc = run / "alloc.json"
    cands = ["vq-12", "rvq-12-8", "rvq-12-12", "rvq-12-12-8", "rvq-12-12-12"] \
        if target_bpw >= 3 else ["vq-8", "vq-12", "rvq-12-4", "rvq-8-8", "rvq-12-8"]
    _orka("allocate", str(src), "--out", str(alloc), "--target-bpw", target_bpw,
          "--candidates", *cands, "--group-size", 8, "--sample-vectors", 8192,
          "--iterations", 4, "--backend", "torch", "--device", "cuda")

    if art.exists():
        import shutil
        shutil.rmtree(art)
    _orka("pack", str(src), "--out", str(art), "--allocation-map", str(alloc),
          "--codebook-mode", "per-tensor", "--normalization", "slrq-block",
          "--outlier-frac", outlier_frac, "--awq-calibration", str(calib),
          "--awq-model-dir", model_dir, "--calibration-max-prompts", 32,
          "--calibration-max-length", 256, "--backend", "torch",
          "--device", "cuda", "--sample-vectors", sample_vectors,
          "--iterations", iterations, "--em-aq-passes", 1,
          "--calibration-max-samples", calib_samples,
          "--group-size", 8, "--codebook-dtype", codebook_dtype)
    data_vol.commit()  # the expensive pack survives even if a later step fails

    # Post-pack steps are cheap and best-effort: a failure here must never
    # discard the pack or (with retries=0) trigger a re-run of the pipeline.
    def _best_effort(label, fn):
        try:
            return fn()
        except Exception as exc:
            import traceback
            print(f"=== {label} FAILED (non-fatal) ===\n{traceback.format_exc()}", flush=True)
            return {}

    _best_effort("distill", lambda: _orka(
        "distill", str(art), "--steps", distill_steps, "--lr", 1e-3, "--device", "cuda",
        "--model-dir", model_dir, "--prompts", str(calib),
        "--calibration-max-prompts", 32, "--calibration-max-length", 256))
    _best_effort("correct", lambda: _orka("correct", str(art), "--rank", correct_rank, "--device", "cuda"))

    report = _best_effort("report", lambda: _last_json(_orka("report", str(art))))
    pulse = _best_effort("pulse-check", lambda: _last_json(_orka(
        "pulse-check", str(art), "--prompts", str(eval_p), "--out", str(run / "pulse.json"),
        "--model-dir", model_dir, "--max-prompts", 11, "--max-length", 192, "--device", "cuda")))
    hf_out = run / "hf"

    def _export():
        if hf_out.exists():
            import shutil
            shutil.rmtree(hf_out)
        _orka("export-vllm", str(art), "--out", str(hf_out), "--model-dir", model_dir, "--dtype", "bfloat16")
    _best_effort("export-vllm", _export)

    data_vol.commit()
    result = {
        "repo": repo, "artifact": str(art), "hf_export": str(hf_out),
        "size": report.get("artifact_size"),
        "ratio": report.get("compression_ratio_fp16_to_artifact"),
        "codebook_mb": round(report.get("total_codebook_bytes", 0) / 1e6, 1),
        "cosine": report.get("cosine_similarity"),
        "kl": pulse.get("kl_divergence"), "top1": pulse.get("top1_agreement"),
    }
    print("=== RESULT ===\n" + json.dumps(result, indent=2), flush=True)
    return result


@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=3 * 3600, env=ENV, retries=0)
def raw(args: str) -> str:
    """Run any orka subcommand. `args` is the full arg string after `orka`."""
    out = _orka(*args.split())
    data_vol.commit()
    return out


@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=3 * 3600,
              env=ENV, retries=0)
def pack_ptq(
    repo: str,
    target_bpw: float = 4.0,
    codebook_dtype: str = "int8",
    outlier_frac: float = 0.005,
    calib_lines: int = 64,
    sample_vectors: int = 0,
    iterations: int = 8,
    calib_samples: int = 8192,
    hf_token: str = "",
) -> dict:
    """Pure PTQ: download -> allocate -> pack -> report -> pulse-check.

    No distill, no correct (those are gradient steps = QAT-flavored). This is the
    post-training-only path, matching the Kaggle recipe but on the A10G."""
    import torch
    from huggingface_hub import login, snapshot_download

    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"=== GPU: {name} ({total_gb:.0f} GB) ===", flush=True)
    if sample_vectors <= 0:
        sample_vectors = int(min(524288, max(65536, total_gb * 8192)))
    print(f"=== PTQ knobs: sample_vectors={sample_vectors} iterations={iterations} "
          f"calib_samples={calib_samples} target_bpw={target_bpw} ===", flush=True)
    if hf_token:
        login(token=hf_token)

    model_dir = snapshot_download(
        repo, allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*", "merges*", "vocab*"]
    )
    src = next(Path(model_dir).glob("*.safetensors"))
    slug = repo.split("/")[-1]
    run = Path("/data") / slug
    run.mkdir(parents=True, exist_ok=True)
    art = run / f"{slug}-{target_bpw}bpw-ptq.orka"

    calib = run / "calib.txt"
    eval_p = run / "eval.txt"
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        lines = [r["text"].strip() for r in ds if len(r["text"].strip()) > 200]
        calib.write_text("\n".join(lines[:calib_lines]))
        eval_p.write_text("\n".join(lines[60:72]))
    except Exception as exc:
        print(f"wikitext fetch failed ({exc}); fallback prompts", flush=True)
        fb = ["The history of science is long and storied." * 4] * 32
        calib.write_text("\n".join(fb))
        eval_p.write_text("\n".join(fb[:11]))

    alloc = run / "alloc.json"
    cands = ["vq-12", "rvq-12-8", "rvq-12-12", "rvq-12-12-8", "rvq-12-12-12"] \
        if target_bpw >= 3 else ["vq-8", "vq-12", "rvq-12-4", "rvq-8-8", "rvq-12-8"]
    _orka("allocate", str(src), "--out", str(alloc), "--target-bpw", target_bpw,
          "--candidates", *cands, "--group-size", 8, "--sample-vectors", 8192,
          "--iterations", 4, "--backend", "torch", "--device", "cuda")

    if art.exists():
        import shutil
        shutil.rmtree(art)
    _orka("pack", str(src), "--out", str(art), "--allocation-map", str(alloc),
          "--codebook-mode", "per-tensor", "--normalization", "slrq-block",
          "--outlier-frac", outlier_frac, "--awq-calibration", str(calib),
          "--awq-model-dir", model_dir, "--calibration-max-prompts", 32,
          "--calibration-max-length", 256, "--backend", "torch",
          "--device", "cuda", "--sample-vectors", sample_vectors,
          "--iterations", iterations, "--em-aq-passes", 1,
          "--calibration-max-samples", calib_samples,
          "--group-size", 8, "--codebook-dtype", codebook_dtype)
    data_vol.commit()

    def _best_effort(label, fn):
        try:
            return fn()
        except Exception as exc:
            import traceback
            print(f"=== {label} FAILED (non-fatal) ===\n{traceback.format_exc()}", flush=True)
            return {}

    report = _best_effort("report", lambda: _last_json(_orka("report", str(art))))
    pulse = _best_effort("pulse-check", lambda: _last_json(_orka(
        "pulse-check", str(art), "--prompts", str(eval_p), "--out", str(run / "pulse.json"),
        "--model-dir", model_dir, "--max-prompts", 11, "--max-length", 192, "--device", "cuda")))
    data_vol.commit()
    result = {
        "repo": repo, "artifact": str(art),
        "size": report.get("artifact_size"),
        "ratio": report.get("compression_ratio_fp16_to_artifact"),
        "codebook_mb": round(report.get("total_codebook_bytes", 0) / 1e6, 1),
        "cosine": report.get("cosine_similarity"),
        "weighted_mse": report.get("weighted_mse"),
        "kl": pulse.get("kl_divergence"), "top1": pulse.get("top1_agreement"),
    }
    print("=== PTQ RESULT ===\n" + json.dumps(result, indent=2), flush=True)
    return result


@app.function(image=image, volumes=VOLUMES)
def ls() -> list[str]:
    """List artifacts in the data volume."""
    files = [str(p.relative_to("/data")) for p in Path("/data").rglob("*.orka")]
    print("\n".join(files) if files else "(no artifacts)")
    return files


@app.local_entrypoint()
def main(repo: str = "", target_bpw: float = 4.0, codebook_dtype: str = "int8"):
    if not repo:
        print("usage: modal run orka_modal.py::compress --repo <hf/model> [--target-bpw 4] [--codebook-dtype int8]")
        print("       modal run orka_modal.py::raw --args 'report /data/...'")
        print("       modal run orka_modal.py::ls")
        return
    print(json.dumps(compress.remote(repo, target_bpw, codebook_dtype), indent=2))
