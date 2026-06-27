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


@app.function(image=image, gpu="H100", volumes=VOLUMES, timeout=6 * 3600,
              env=ENV, retries=0)
def qat(
    repo: str,
    target_bpw: float = 4.0,
    steps: int = 600,
    seq_len: int = 0,        # 0 -> auto-scale to VRAM
    batch: int = 0,          # 0 -> auto-scale to VRAM
    lr: float = 1e-4,
    commit: float = 0.5,
    cb_weight: float = 0.5,
    ppl_ctx: int = 512,
    ppl_maxtok: int = 25600,
    hf_token: str = "",
) -> dict:
    """4bpw VQ-QAT: allocate -> PTQ pack -> qat_train -> wikitext PPL + KL/top1.

    QAT is gradient training (compute-bound). batch/seq_len default to 0 = auto:
    sized from the GPU's VRAM and the model's param count so the same call
    naturally fills a T4, A10G, or H100. lr 1e-4 / commit 0.5 (the 3e-4 Supra
    default diverges on Falcon H1's SSM layers).

    Re-run safe: ptq.orka, ptq-hf, alloc.json, corpus.txt are reused if they
    already exist in the volume. Only qat-hf is always rebuilt from scratch."""
    import json as _json
    import math
    import sys as _sys
    import torch
    from huggingface_hub import login, snapshot_download

    print(f"=== GPU: {torch.cuda.get_device_name(0)} ===", flush=True)
    if hf_token:
        login(token=hf_token)
    model_dir = snapshot_download(
        repo, allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*", "merges*", "vocab*"]
    )
    src = next(Path(model_dir).glob("*.safetensors"))
    slug = repo.split("/")[-1]
    run = Path("/data") / f"{slug}-qat"
    run.mkdir(parents=True, exist_ok=True)
    import shutil as _sh
    # Only clear qat-hf (always rebuilt); keep ptq.orka/ptq-hf/alloc.json/corpus.txt
    # from prior runs so PTQ is not redone unnecessarily on timeout-retries.
    _sh.rmtree(run / "qat-hf", ignore_errors=True)

    from datasets import load_dataset
    corpus = run / "corpus.txt"
    if not corpus.exists():
        train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        tlines = [r["text"].strip() for r in train if len(r["text"].strip()) > 200]
        corpus.write_text("\n".join(tlines[:900]))
        print("  corpus built", flush=True)
    else:
        print("  corpus reused from volume", flush=True)
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ppl_text = "\n\n".join(t for t in (r["text"] for r in test) if t.strip())

    alloc_path = run / "alloc.json"
    if not alloc_path.exists():
        from orka.allocate import build_allocation
        alloc = build_allocation(
            src, target_bpw,
            candidate_specs=("vq-12", "rvq-12-8", "rvq-12-12", "rvq-12-12-8", "rvq-12-12-12"),
            group_size=8, sample_vectors=8192, iterations=4, backend="torch", device="cuda",
        )
        alloc_path.write_text(_json.dumps(alloc, indent=2))
        print(f"  alloc built: {alloc['achieved_bpw']:.3f} bpw", flush=True)
    else:
        alloc = _json.loads(alloc_path.read_text())
        print(f"  alloc reused from volume: {alloc['achieved_bpw']:.3f} bpw", flush=True)

    from orka.allocate import allocation_tensor_stages
    from orka.pipeline.pack import pack_checkpoint
    from orka.artifact.export import export_vllm
    ptq_art = run / "ptq.orka"
    ptq_hf = run / "ptq-hf"
    if not ptq_art.exists():
        pack_checkpoint(
            source=src, out_dir=ptq_art, group_size=8, codebook_size=4096,
            codebook_mode="per-tensor", backend="torch", device="cuda",
            normalization="slrq-block", outlier_frac=0.005, sample_vectors=131072,
            iterations=8, em_aq_passes=1, tensor_stages_map=allocation_tensor_stages(alloc),
        )
        data_vol.commit()
        print("  ptq.orka packed and committed", flush=True)
    else:
        print("  ptq.orka reused from volume", flush=True)
    if not ptq_hf.exists():
        export_vllm(ptq_art, ptq_hf, model_dir=Path(model_dir), dtype="bfloat16")
        data_vol.commit()
        print("  ptq-hf exported and committed", flush=True)
    else:
        print("  ptq-hf reused from volume", flush=True)

    # --- dynamic VRAM-aware sizing (aggressive estimate + OOM backoff) ---
    # QAT memory = teacher(bf16,2B) + student(fp32,4B) + grads(4B) + adam(8B)
    # ~= 18 B/param fixed; the rest of VRAM funds activations (batch x seq).
    # The estimate targets ~85% VRAM; if it overshoots, the backoff loop below
    # halves batch on CUDA OOM and retries, so the GPU is filled without guessing.
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    nparams = alloc["total_params"]
    auto_seq = seq_len if seq_len > 0 else (512 if total_gb >= 24 else 256)
    if batch > 0:
        auto_batch = batch
    else:
        fixed_gb = nparams * 18 / 1e9
        usable = max(1.0, total_gb * 0.85 - fixed_gb)
        per_seq_gb = (nparams / 1e9) * (auto_seq / 512) * 1.6
        auto_batch = int(max(2, min(128, usable / max(0.1, per_seq_gb))))
    auto_max_seqs = max(700, auto_batch * 100)
    print(f"=== autoscale: VRAM {total_gb:.0f}GB, {nparams/1e6:.0f}M params -> "
          f"start batch={auto_batch} seq_len={auto_seq} max_seqs={auto_max_seqs} ===", flush=True)

    qat_hf = run / "qat-hf"
    from orka.qat_train import main as qat_main
    cur_batch = auto_batch
    while True:
        if qat_hf.exists():
            _sh.rmtree(qat_hf, ignore_errors=True)
        _sys.argv = [
            "qat_train", str(model_dir), str(alloc_path), str(corpus), str(qat_hf),
            "--steps", str(steps), "--seq-len", str(auto_seq), "--batch", str(cur_batch),
            "--lr", str(lr), "--commit", str(commit), "--cb-weight", str(cb_weight),
            "--device", "cuda", "--max-seqs", str(max(700, cur_batch * 100)),
        ]
        try:
            print(f"=== QAT attempt: batch={cur_batch} ===", flush=True)
            qat_main()
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower() or cur_batch <= 2:
                raise
            torch.cuda.empty_cache()
            cur_batch = max(2, cur_batch // 2)
            print(f"=== CUDA OOM at batch={cur_batch * 2}; backing off to batch={cur_batch} ===", flush=True)
    data_vol.commit()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch.nn.functional as F

    def wikitext_ppl(d):
        tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
        ids = tok(ppl_text, return_tensors="pt").input_ids[0][:ppl_maxtok]
        m = AutoModelForCausalLM.from_pretrained(d, local_files_only=True, dtype=torch.float32).cuda().eval()
        nll = ntok = 0
        with torch.no_grad():
            for i in range(0, len(ids) - 1, ppl_ctx):
                ch = ids[i:i + ppl_ctx].unsqueeze(0).cuda()
                if ch.shape[1] < 2: break
                nll += m(ch, labels=ch).loss.item() * (ch.shape[1] - 1); ntok += ch.shape[1] - 1
        del m; torch.cuda.empty_cache()
        return math.exp(nll / ntok), ntok

    ppl = {}
    for tag, d in (("fp16", model_dir), ("4bpw-PTQ", ptq_hf), ("4bpw-QAT", qat_hf)):
        pp, n = wikitext_ppl(d); ppl[tag] = {"ppl": pp, "tokens": n}
        print(f"  PPL {tag}: {pp:.4f} ({n} tok)", flush=True)

    gp = ["The capital of France is", "import numpy as np\ndef softmax(x):", "Q: What is 2+2?\nA:"]
    tk = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    gens = {}
    for tag, d in (("fp16", model_dir), ("4bpw-PTQ", ptq_hf), ("4bpw-QAT", qat_hf)):
        m = AutoModelForCausalLM.from_pretrained(d, local_files_only=True, dtype=torch.bfloat16).cuda().eval()
        outs = []
        with torch.no_grad():
            for p in gp:
                ids = {k: v.cuda() for k, v in tk(p, return_tensors="pt").items()}
                o = m.generate(**ids, max_new_tokens=24, do_sample=False, pad_token_id=tk.eos_token_id)
                outs.append(tk.decode(o[0], skip_special_tokens=True))
        gens[tag] = outs; del m; torch.cuda.empty_cache()

    report = {"repo": repo, "achieved_bpw": alloc["achieved_bpw"], "steps": steps,
              "perplexity": ppl, "generations": gens, "prompts": gp}
    (run / "qat_report.json").write_text(_json.dumps(report, indent=2)); data_vol.commit()
    print("=== QAT RESULT ===\n" + _json.dumps({"perplexity": ppl}, indent=2), flush=True)
    return report


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
