"""Serve a .orka artifact through VQLinear - no dense reconstruction.

Every eval so far rebuilt a 5.4 GB dense checkpoint, so the measured artifact was a download
saving only: resident memory stayed at full bf16. export_orka_hf_repo builds a
transformers-loadable repo whose Linears ARE VQLinear, which keeps the packed payload
resident. That is the actual product claim and it has never been tested here.

Requires group_size=8 AND block_scale_size=32 (hf_quantizer KERNEL_* constants). Artifacts
packed at block 64 - every one before this - are rejected by that guard.

    modal run deploy/modal/serve_native.py::serve --artifact <name>
"""

import json
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
app = modal.App("orka-serve")
data_vol = modal.Volume.from_name("orka-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("orka-hf-cache", create_if_missing=True)
VOLUMES = {"/data": data_vol, "/hf": hf_vol}
ENV = {"HF_HOME": "/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

PROMPTS = [
    "The Industrial Revolution began in Britain in the late eighteenth century because",
    "Photosynthesis is the process by which plants",
]
FEWSHOT = [("France -> Paris\nJapan -> Tokyo\nItaly -> Rome\nGermany ->", "Berlin"),
           ("2 + 3 = 5\n7 + 4 = 11\n10 + 5 = 15\n8 + 6 =", "14"),
           ("happy -> sad\nhot -> cold\nbig -> small\nfast ->", "slow"),
           ("dog -> puppy\ncat -> kitten\nhorse -> foal\ncow ->", "calf")]
FACTUAL = [("The capital city of Australia is", "Canberra"),
           ("Water boils at a temperature of 100 degrees", "Celsius"),
           ("The largest planet in our solar system is", "Jupiter"),
           ("William Shakespeare wrote a play called Romeo and", "Juliet"),
           ("The chemical symbol for gold is", "Au")]


@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=3 * 3600, env=ENV)
def serve(artifact: str, repo: str = "LiquidAI/LFM2.5-2.6B-Base"):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from orka.integrations.hf_quantizer import export_orka_hf_repo, register_orka_quantizer

    src = Path(snapshot_download(repo))
    tok = AutoTokenizer.from_pretrained(repo)
    art = Path("/data/artifacts") / artifact
    out = Path("/data/served") / artifact.replace(".orka", "-vqhf")

    t0 = time.perf_counter()
    if not (out / "config.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        info = export_orka_hf_repo(art, src, out)
        print(f"[{time.perf_counter()-t0:.0f}s] exported: {json.dumps(info)[:300]}", flush=True)
        data_vol.commit()
    else:
        print("  reusing exported repo", flush=True)
    repo_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"  served repo on disk: {repo_bytes/1e9:.2f} GB", flush=True)

    register_orka_quantizer()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t1 = time.perf_counter()
    m = AutoModelForCausalLM.from_pretrained(
        str(out), dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True).to("cuda").eval()
    resident = torch.cuda.memory_allocated() - base
    print(f"[{time.perf_counter()-t0:.0f}s] loaded natively in {time.perf_counter()-t1:.0f}s"
          f"   RESIDENT {resident/1e9:.2f} GB", flush=True)

    @torch.no_grad()
    def gen(p, n):
        ids = tok(p, return_tensors="pt").to("cuda")
        o = m.generate(**ids, max_new_tokens=n, do_sample=False,
                       pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    outs = [{"prompt": p, "out": gen(p, 50)} for p in PROMPTS]
    for o in outs:
        print(f"  > {o['prompt'][:55]}\n    {o['out'][:190]!r}", flush=True)
    fs = sum(w.lower() in gen(p, 6).lower() for p, w in FEWSHOT)
    fa = sum(w.lower() in gen(p, 8).lower() for p, w in FACTUAL)

    # throughput, compressed-resident
    b = tok(PROMPTS[0], return_tensors="pt").to("cuda")
    with torch.no_grad():
        for _ in range(2):
            m.generate(**b, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        for _ in range(3):
            m.generate(**b, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
    tps = 3 * 32 / (time.perf_counter() - t2)

    res = {"artifact": artifact, "served_repo_bytes": repo_bytes,
           "resident_bytes": resident, "fewshot": f"{fs}/{len(FEWSHOT)}",
           "factual": f"{fa}/{len(FACTUAL)}", "tok_per_s": round(tps, 1),
           "generations": outs}
    print(f"\n  few-shot {fs}/{len(FEWSHOT)}   factual {fa}/{len(FACTUAL)}   "
          f"{tps:.1f} tok/s   resident {resident/1e9:.2f} GB", flush=True)
    print("\nRESULT " + json.dumps(res)[:800], flush=True)
    return res
