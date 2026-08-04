"""Find the operating point for ONE model by measuring it, not by copying another model's.

Every knob used on LFM2.5-2.6B so far was validated on LFM2.5-Encoder-230M: a different
architecture class, 12x smaller, scored on a proxy task MTEB later showed was too easy (the
same config went from "parity" to -9.1%). Transplanting 2.6 bpw across that gap is a guess.

This sweeps target_bpw on the actual model and scores each artifact with orka's native
perplexity eval, which is the metric that applies to a causal LM. fp16 is the reference.

    modal run deploy/modal/bpw_sweep.py::sweep --repo LiquidAI/LFM2.5-2.6B-Base
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
app = modal.App("orka-bpw-sweep")
data_vol = modal.Volume.from_name("orka-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("orka-hf-cache", create_if_missing=True)
VOLUMES = {"/data": data_vol, "/hf": hf_vol}
ENV = {"HF_HOME": "/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=6 * 3600, env=ENV,
              secrets=[modal.Secret.from_name("huggingface")])
def sweep(repo: str, bpws: str = "2.6,3.0,3.4,3.8", n_calib: int = 48,
          n_eval: int = 64, max_len: int = 256):
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from orka.autoquant.roles import classify_role
    from orka.pipeline.pack import pack_checkpoint
    from orka.quant.curvature import achieved_bpw, fisher_diagonal, waterfill_with_roles

    t0 = time.perf_counter()
    src = Path(snapshot_download(repo))
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    longs = [t.strip() for t in ds["text"] if len(t.strip()) > 400]
    calib, evals = longs[:n_calib], longs[n_calib:n_calib + n_eval]

    prompts = Path("/data/wikitext_eval.txt")
    prompts.write_text("\n".join(t.replace("\n", " ") for t in evals))

    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").train()

    def loss_fn(m, text):
        b = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to("cuda")
        return m(input_ids=b["input_ids"], labels=b["input_ids"]).loss

    stats = fisher_diagonal(model, calib, loss_fn)
    print(f"[{time.perf_counter()-t0:.0f}s] fisher {len(stats)} tensors", flush=True)
    del model
    torch.cuda.empty_cache()

    from safetensors import safe_open
    sh = {}
    for f in sorted(src.glob("*.safetensors")):
        with safe_open(f, framework="np") as h:
            for k in h.keys():
                sh[k] = tuple(h.get_slice(k).get_shape())
    role_of = lambda n: classify_role(n, sh.get(n, (8, 8)))[0]  # noqa: E731

    acts = {}
    m2 = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    raw, handles = {}, []
    for name, mod in m2.named_modules():
        if isinstance(mod, torch.nn.Linear):
            def hook(_m, inp, _o, _n=name):
                x = inp[0]
                if x.dim() > 2:
                    x = x.reshape(-1, x.shape[-1])
                raw.setdefault(_n, []).append(x.detach().float().cpu())
            handles.append(mod.register_forward_hook(hook))
    CAP = 4096
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for t in calib:
            b = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to("cuda")
            m2(input_ids=b["input_ids"])
            for n, xs in list(raw.items()):
                if sum(x.shape[0] for x in xs) > 2 * CAP:
                    full = torch.cat(xs, 0)
                    raw[n] = [full[torch.randperm(full.shape[0], generator=gen)[:CAP]]]
    for h in handles:
        h.remove()
    for n, xs in raw.items():
        full = torch.cat(xs, 0) if len(xs) > 1 else xs[0]
        acts[n + ".weight"] = full[:CAP]
    raw.clear()
    del m2
    torch.cuda.empty_cache()
    print(f"[{time.perf_counter()-t0:.0f}s] activations {len(acts)}", flush=True)

    rows = []
    for target in [float(x) for x in bpws.split(",")]:
        stages, dense = waterfill_with_roles(stats, target, role_of)
        out = Path("/data/artifacts") / f"{repo.split('/')[-1]}-sweep{target}.orka"
        pack_checkpoint(source=src, out_dir=out, group_size=8, normalization="block-max",
                        backend="torch", device="cuda", em_aq_passes=3,
                        tensor_stages_map=stages,
                        only_tensors=list(stages), only_tensors_passthrough=True,
                        awq_activations=acts, awq_alpha=0.5,
                        codebook_dtype="int8", block_scale_size=64)
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        ev = Path(f"/data/eval-{target}.json")
        subprocess.run([sys.executable, "-m", "orka", "eval", str(out),
                        "--prompts", str(prompts), "--out", str(ev),
                        "--model-dir", str(src), "--device", "cuda",
                        "--max-length", str(max_len)], cwd="/root", check=False)
        q = json.loads(ev.read_text()) if ev.exists() else {}
        rows.append({"target_bpw": target,
                     "achieved_bpw": achieved_bpw(stages, {k: stats[k] for k in stages}),
                     "bytes": size, "quantized": len(stages), "dense": len(dense),
                     "eval": q})
        print(f"SWEEP {json.dumps(rows[-1])}", flush=True)
        data_vol.commit()
    print("\nRESULT " + json.dumps(rows), flush=True)
    return rows
