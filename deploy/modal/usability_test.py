"""Is the compressed model actually usable, or just scoring well?

Perplexity is an average over token likelihoods - a model can sit at 1.17x and still loop,
drift, or lose factual recall. This generates from the SAME prompts with the SAME seed on
bf16 and on the artifact, prints both for reading, and scores three things a base LM must do:

  continuation   does it stay coherent over 80 tokens
  few-shot       does it follow a pattern from 3 examples (the core base-model capability)
  factual        does it complete well-known facts correctly

Greedy decoding throughout, so any difference is the weights, not sampling luck.

    modal run deploy/modal/usability_test.py::usability --artifact <name>
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
app = modal.App("orka-usability")
data_vol = modal.Volume.from_name("orka-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("orka-hf-cache", create_if_missing=True)
VOLUMES = {"/data": data_vol, "/hf": hf_vol}
ENV = {"HF_HOME": "/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

CONTINUATION = [
    "The Industrial Revolution began in Britain in the late eighteenth century because",
    "To make bread you need flour, water, yeast and salt. The first step is",
    "Photosynthesis is the process by which plants",
]
FEWSHOT = [
    ("France -> Paris\nJapan -> Tokyo\nItaly -> Rome\nGermany ->", "Berlin"),
    ("2 + 3 = 5\n7 + 4 = 11\n10 + 5 = 15\n8 + 6 =", "14"),
    ("happy -> sad\nhot -> cold\nbig -> small\nfast ->", "slow"),
    ("dog -> puppy\ncat -> kitten\nhorse -> foal\ncow ->", "calf"),
]
FACTUAL = [
    ("The capital city of Australia is", "Canberra"),
    ("Water boils at a temperature of 100 degrees", "Celsius"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("William Shakespeare wrote a play called Romeo and", "Juliet"),
    ("The chemical symbol for gold is", "Au"),
]


@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=3 * 3600, env=ENV)
def usability(artifact: str, repo: str = "LiquidAI/LFM2.5-2.6B-Base"):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = Path(snapshot_download(repo))
    tok = AutoTokenizer.from_pretrained(repo)

    def build(which):
        if which == "bf16":
            return AutoModelForCausalLM.from_pretrained(
                repo, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
        import subprocess
        import sys
        rec = Path("/tmp/rec")
        subprocess.run([sys.executable, "-m", "orka", "reconstruct",
                        str(Path("/data/artifacts") / artifact),
                        "--out", str(rec / "model.safetensors"), "--format", "safetensors"],
                       cwd="/root", check=True)
        for f in src.iterdir():
            if f.is_file() and not f.name.endswith((".safetensors", ".bin")):
                (rec / f.name).write_bytes(f.read_bytes())
        return AutoModelForCausalLM.from_pretrained(
            str(rec), dtype=torch.bfloat16, trust_remote_code=True,
            local_files_only=True).to("cuda").eval()

    @torch.no_grad()
    def gen(m, prompt, n):
        ids = tok(prompt, return_tensors="pt").to("cuda")
        out = m.generate(**ids, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    results = {}
    for which in ("bf16", "orka"):
        t0 = time.perf_counter()
        m = build(which)
        r = {"continuation": [], "fewshot": [], "factual": []}
        for p in CONTINUATION:
            r["continuation"].append({"prompt": p, "out": gen(m, p, 60)})
        fs_ok = 0
        for p, want in FEWSHOT:
            o = gen(m, p, 6)
            hit = want.lower() in o.lower()
            fs_ok += hit
            r["fewshot"].append({"prompt": p.split(chr(10))[-1], "want": want,
                                 "got": o.strip()[:30], "ok": hit})
        fa_ok = 0
        for p, want in FACTUAL:
            o = gen(m, p, 8)
            hit = want.lower() in o.lower()
            fa_ok += hit
            r["factual"].append({"prompt": p[-38:], "want": want,
                                 "got": o.strip()[:30], "ok": hit})
        r["fewshot_score"] = f"{fs_ok}/{len(FEWSHOT)}"
        r["factual_score"] = f"{fa_ok}/{len(FACTUAL)}"
        # degeneracy check: unique-token fraction over a long greedy continuation
        long = gen(m, CONTINUATION[0], 120)
        toks = tok(long)["input_ids"]
        r["repetition"] = round(len(set(toks)) / max(len(toks), 1), 3)
        results[which] = r
        print(f"\n===== {which}  ({time.perf_counter()-t0:.0f}s) =====", flush=True)
        print(f"  few-shot {r['fewshot_score']}   factual {r['factual_score']}   "
              f"unique-token frac {r['repetition']}", flush=True)
        for c in r["continuation"]:
            print(f"  > {c['prompt'][:60]}\n    {c['out'][:200]!r}", flush=True)
        del m
        torch.cuda.empty_cache()

    print("\nRESULT " + json.dumps(results), flush=True)
    return results
