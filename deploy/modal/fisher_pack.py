"""Curvature-allocated pack for models too large to backprop locally.

fisher_diagonal needs weights AND gradients resident. At 2.7B in bf16 that is 10.8 GB, past
what an 11.6 GB consumer card has left under a 10 GB cap, so the allocator is unreachable on
the dev box for anything much above 1B. An A10G (24 GB) clears it for ~$0.40 a run.

Self-contained: importing a sibling module fails inside Modal's execution context, so the
image/app/volumes are declared here rather than pulled from orka_modal.

    modal run deploy/modal/fisher_pack.py::fisher_pack --repo LiquidAI/LFM2.5-2.6B-Base
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

app = modal.App("orka-fisher")
data_vol = modal.Volume.from_name("orka-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("orka-hf-cache", create_if_missing=True)
VOLUMES = {"/data": data_vol, "/hf": hf_vol}
ENV = {"HF_HOME": "/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=4 * 3600, env=ENV)
def fisher_pack(repo: str, target_bpw: float = 2.6, n_calib: int = 48,
                max_len: int = 256, use_roles: bool = True, tag: str = "",
                error_compensation: bool = False, use_awq: bool = True,
                min_sqnr_db: float = 14.0):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from orka.autoquant.roles import classify_role
    from orka.pipeline.pack import pack_checkpoint
    from orka.quant.curvature import achieved_bpw, fisher_diagonal, waterfill_stages

    t0 = time.perf_counter()
    src = Path(snapshot_download(repo))
    print(f"[{time.perf_counter()-t0:.0f}s] snapshot {src}", flush=True)

    from datasets import load_dataset
    # bare "wikitext" is rejected by newer huggingface_hub: repo ids must be namespaced
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t.strip() for t in ds["text"] if len(t.strip()) > 400][:n_calib]

    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").train()
    print(f"[{time.perf_counter()-t0:.0f}s] model on GPU "
          f"{torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    def causal_loss(m, text):
        b = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to("cuda")
        return m(input_ids=b["input_ids"], labels=b["input_ids"]).loss

    stats = fisher_diagonal(model, texts, causal_loss)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{time.perf_counter()-t0:.0f}s] fisher {len(stats)} tensors, peak {peak:.2f} GB",
          flush=True)
    del model
    torch.cuda.empty_cache()

    if use_roles:
        from safetensors import safe_open

        from orka.quant.curvature import waterfill_with_roles
        sh = {}
        for f in sorted(src.glob("*.safetensors")):
            with safe_open(f, framework="np") as h:
                for k in h.keys():
                    sh[k] = tuple(h.get_slice(k).get_shape())
        stages, dense = waterfill_with_roles(
            stats, target_bpw, lambda n: classify_role(n, sh.get(n, (8, 8)))[0],
            min_sqnr_db=min_sqnr_db)
        print(f"  roles: {len(stages)} quantized, {len(dense)} dense", flush=True)
    else:
        stages, dense = waterfill_stages(stats, target_bpw,
                                         min_sqnr_db=min_sqnr_db), {}
    bpw = achieved_bpw(stages, {k: stats[k] for k in stages})
    print(f"  allocated {bpw:.3f} bpw", flush=True)

    # AWQ activation calibration: measured +0.131 cosine at identical bytes on
    # LFM2.5-Encoder-230M. Data-free packing is the configuration that scored 0.813.
    acts = {}
    if not use_awq:
        print("  AWQ disabled", flush=True)
    m2 = None if not use_awq else AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    raw, handles = {}, []
    for name, mod in ([] if m2 is None else m2.named_modules()):
        if isinstance(mod, torch.nn.Linear):
            def hook(_m, inp, _o, _n=name):
                x = inp[0]
                if x.dim() > 2:
                    x = x.reshape(-1, x.shape[-1])
                raw.setdefault(_n, []).append(x.detach().float().cpu())
            handles.append(mod.register_forward_hook(hook))
    CAP = 4096
    gen = torch.Generator().manual_seed(0)

    def trim(n):
        xs = raw.get(n)
        if not xs or sum(t.shape[0] for t in xs) <= 2 * CAP:
            return
        full = torch.cat(xs, 0)
        raw[n] = [full[torch.randperm(full.shape[0], generator=gen)[:CAP]]]

    with torch.no_grad():
        for t in (texts if m2 is not None else []):
            b = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to("cuda")
            m2(input_ids=b["input_ids"])
            for n in list(raw):
                trim(n)
    for h in handles:
        h.remove()
    for n, xs in raw.items():
        full = torch.cat(xs, 0) if len(xs) > 1 else xs[0]
        if full.shape[0] > CAP:
            full = full[torch.randperm(full.shape[0], generator=gen)[:CAP]]
        acts[n + ".weight"] = full
    raw.clear()
    del m2
    torch.cuda.empty_cache()
    print(f"[{time.perf_counter()-t0:.0f}s] activations {len(acts)} tensors", flush=True)

    suffix = f"-{tag}" if tag else ""
    out = Path("/data/artifacts") / f"{repo.split('/')[-1]}-fisher{target_bpw}{suffix}.orka"
    out.parent.mkdir(parents=True, exist_ok=True)
    pack_checkpoint(source=src, out_dir=out, group_size=8, normalization="block-max",
                    backend="torch", device="cuda", em_aq_passes=3,
                    tensor_stages_map=stages,
                    only_tensors=list(stages), only_tensors_passthrough=True,
                    awq_activations=acts or None, awq_alpha=0.5,
                    error_compensation=error_compensation,
                    codebook_dtype="int8", block_scale_size=64)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    orig = sum(p.stat().st_size for p in src.glob("*.safetensors"))
    res = {"repo": repo, "tag": tag or "data-free", "artifact": str(out), "bytes": size, "orig_bytes": orig,
           "ratio_vs_bf16": orig / size, "achieved_bpw": bpw,
           "quantized": len(stages), "dense": len(dense), "awq_tensors": len(acts), "error_compensation": error_compensation,
           "min_sqnr_db": min_sqnr_db,
           "fisher_peak_gb": peak, "seconds": time.perf_counter() - t0}
    print("\nRESULT " + json.dumps(res), flush=True)
    data_vol.commit()
    return res
