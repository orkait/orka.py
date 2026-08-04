"""Run `orka correct` on a packed artifact, then re-evaluate.

correct adds a low-rank residual patch to each quantized tensor - no training, no calibration
data. It is a step of orka's own documented recipe (allocate -> pack -> distill -> correct ->
report) that every pack this session skipped.

Copies the artifact first so the corrected version lands beside the original rather than
mutating the only good result.

    modal run deploy/modal/correct_artifact.py::correct --artifact <name> --rank 32
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "safetensors", "numpy", "scipy", "tqdm",
                 "huggingface_hub", "datasets", "accelerate")
    .add_local_dir(str(REPO_ROOT / "orka"), "/root/orka", copy=True)
)
app = modal.App("orka-correct")
data_vol = modal.Volume.from_name("orka-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("orka-hf-cache", create_if_missing=True)
VOLUMES = {"/data": data_vol, "/hf": hf_vol}
ENV = {"HF_HOME": "/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=4 * 3600, env=ENV)
def correct(artifact: str, rank: int = 32, repo: str = "LiquidAI/LFM2.5-2.6B-Base",
            n_calib: int = 48, n_eval: int = 96, max_len: int = 256):
    src_art = Path("/data/artifacts") / artifact
    if not src_art.exists():
        raise FileNotFoundError(f"missing artifact: {src_art}")
    out_art = Path("/data/artifacts") / artifact.replace(".orka", f"-corr{rank}.orka")
    if out_art.exists():
        shutil.rmtree(out_art)
    shutil.copytree(src_art, out_art)
    before = sum(p.stat().st_size for p in out_art.rglob("*") if p.is_file())
    print(f"copied -> {out_art.name}  ({before/1e6:.1f} MB)", flush=True)

    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "orka", "correct", str(out_art),
                        "--rank", str(rank), "--device", "cuda"],
                       cwd="/root", capture_output=True, text=True)
    print(r.stdout[-3000:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], flush=True)
        raise RuntimeError(f"orka correct failed (exit {r.returncode})")
    after = sum(p.stat().st_size for p in out_art.rglob("*") if p.is_file())
    print(f"corrected in {time.perf_counter()-t0:.0f}s   "
          f"{before/1e6:.1f} -> {after/1e6:.1f} MB "
          f"(+{100*(after-before)/before:.1f}%)", flush=True)
    data_vol.commit()

    # re-evaluate on the same held-out prompts
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(repo))
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    longs = [t.strip() for t in ds["text"] if len(t.strip()) > 400]
    prompts = Path("/data/eval_prompts.txt")
    prompts.write_text("\n".join(t.replace("\n", " ")
                                 for t in longs[n_calib:n_calib + n_eval]))
    ev = Path(f"/data/eval-{out_art.name}.json")
    subprocess.run([sys.executable, "-m", "orka", "eval", str(out_art),
                    "--prompts", str(prompts), "--out", str(ev),
                    "--model-dir", str(src), "--device", "cuda",
                    "--max-length", str(max_len)], cwd="/root", check=False)
    q = json.loads(ev.read_text()) if ev.exists() else {}
    res = {"artifact": out_art.name, "rank": rank, "bytes_before": before,
           "bytes_after": after, "eval": q}
    print("\nRESULT " + json.dumps(res)[:900], flush=True)
    data_vol.commit()
    return res
