"""Perplexity for packed artifacts against the bf16 reference.

The data-free and AWQ packs came out byte-identical in size (1,401,087,478 vs 1,401,106,513)
at the same 2.767 bpw, so size says nothing about which is better. Calibration changes which
codewords are chosen, not the format. Only a quality number separates them, and neither run
produced one.

Eval prompts are wikitext passages held out from the calibration slice used to pack.

    modal run deploy/modal/eval_artifacts.py::evaluate --repo LiquidAI/LFM2.5-2.6B-Base
"""

import json
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
app = modal.App("orka-eval")
data_vol = modal.Volume.from_name("orka-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("orka-hf-cache", create_if_missing=True)
VOLUMES = {"/data": data_vol, "/hf": hf_vol}
ENV = {"HF_HOME": "/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=3 * 3600, env=ENV)
def evaluate(repo: str, artifacts: str = "", n_calib: int = 48, n_eval: int = 96,
             max_len: int = 256):
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    src = Path(snapshot_download(repo))
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    longs = [t.strip() for t in ds["text"] if len(t.strip()) > 400]
    evals = longs[n_calib:n_calib + n_eval]          # disjoint from the packing calibration
    prompts = Path("/data/eval_prompts.txt")
    prompts.write_text("\n".join(t.replace("\n", " ") for t in evals))
    print(f"[{time.perf_counter()-t0:.0f}s] {len(evals)} held-out prompts", flush=True)

    tok = AutoTokenizer.from_pretrained(repo)
    ref = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    tot_nll = tot_tok = 0
    with torch.no_grad():
        for t in evals:
            b = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to("cuda")
            n = b["input_ids"].shape[1]
            loss = ref(input_ids=b["input_ids"], labels=b["input_ids"]).loss
            tot_nll += float(loss) * n
            tot_tok += n
    import math
    ref_ppl = math.exp(tot_nll / tot_tok)
    print(f"[{time.perf_counter()-t0:.0f}s] bf16 reference ppl {ref_ppl:.4f}", flush=True)
    del ref
    torch.cuda.empty_cache()

    names = [a for a in artifacts.split(",") if a] or [
        f"{repo.split('/')[-1]}-fisher2.6.orka",
        f"{repo.split('/')[-1]}-fisher2.6-awq.orka",
    ]
    rows = [{"name": "bf16 reference", "ppl": ref_ppl, "bytes": None}]
    for nm in names:
        art = Path("/data/artifacts") / nm
        if not art.exists():
            print(f"  MISSING {art}", flush=True)
            continue
        # Reconstruct ONCE per artifact into the volume and reuse. orka eval otherwise
        # rebuilds a 5.4 GB dense checkpoint into a container tmpdir that dies with the
        # container - 7 rebuilds this session, 40-70 min of pure I/O, all avoidable.
        rec = Path("/data/reconstructed") / nm
        cmd = [sys.executable, "-m", "orka", "eval", str(art),
               "--prompts", str(prompts), "--out", str(out := Path(f"/data/eval-{nm}.json")),
               "--model-dir", str(src), "--device", "cuda", "--max-length", str(max_len)]
        if (rec / "model.safetensors").exists():
            print(f"  reusing cached reconstruction {rec}", flush=True)
            cmd += ["--reconstructed-model-dir", str(rec)]
        else:
            rec.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, "-m", "orka", "reconstruct", str(art),
                            "--out", str(rec / "model.safetensors"),
                            "--format", "safetensors"], cwd="/root", check=True)
            for f in src.iterdir():
                if f.is_file() and not f.name.endswith((".safetensors", ".bin")):
                    (rec / f.name).write_bytes(f.read_bytes())
            data_vol.commit()
            cmd += ["--reconstructed-model-dir", str(rec)]
        subprocess.run(cmd, cwd="/root", check=False)
        q = json.loads(out.read_text()) if out.exists() else {}
        size = sum(p.stat().st_size for p in art.rglob("*") if p.is_file())
        rows.append({"name": nm, "bytes": size, "eval": q})
        print(f"  EVAL {nm} -> {json.dumps(q)[:220]}", flush=True)
        data_vol.commit()
    print("\nRESULT " + json.dumps(rows), flush=True)
    return rows
