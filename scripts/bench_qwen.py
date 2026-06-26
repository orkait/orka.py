"""Qwen2.5-0.5B compression+quality bench: fp16 | bnb-nf4 | orka (linears RVQ, fp16 tied head).

Qwen ties embed<->lm_head, so the (huge 152k) embedding IS the output head - keep it fp16
per the head-precision lesson. Only the transformer linears are RVQ'd.
"""
import os, sys, time, json, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch, numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from orka.inference.vq_linear import build_vq_linear

Q = "/home/kai/ai-models/hf-cache/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
ART = Path("/home/kai/ai-models/qwen05_g8.orka")
tok = AutoTokenizer.from_pretrained(Q)
TXT = open("/tmp/claude-1000/-mnt-storage-codespace-code-orkait-orka-compiler/6e173d66-a10f-4ee7-88d8-e0cc77e0223f/scratchpad/ppl.txt").read()
ids = tok(TXT, return_tensors="pt").input_ids[:, :512].cuda()
PROMPT = tok("The history of artificial intelligence", return_tensors="pt").input_ids.cuda()

def ppl(m):
    with torch.no_grad(): return float(torch.exp(m(ids, labels=ids).loss))
def decode_ts(m, n=64):
    m.generate(PROMPT, max_new_tokens=4, do_sample=False); torch.cuda.synchronize()
    t = time.time(); m.generate(PROMPT, max_new_tokens=n, do_sample=False); torch.cuda.synchronize()
    return n / (time.time() - t)
def dsize(p):
    p = Path(p)
    if p.is_file(): return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

res = {}
def emit(name, m, disk):
    torch.cuda.reset_peak_memory_stats()
    p, d = ppl(m), decode_ts(m)
    v = torch.cuda.max_memory_allocated() / 1e6
    res[name] = dict(ppl=round(p,3), decode=round(d,1), vram_mb=round(v,1), disk_mb=round(disk/1e6,1))
    print(f"{name:12s} disk={disk/1e6:7.1f}MB vram={v:7.1f}MB decode={d:7.1f} ppl={p:.3f}", flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()

# fp16
emit("fp16", AutoModelForCausalLM.from_pretrained(Q, dtype=torch.float16).cuda().eval(), dsize(Q+"/model.safetensors"))
# bnb-nf4
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
emit("bnb-nf4", AutoModelForCausalLM.from_pretrained(Q, quantization_config=bnb, device_map="cuda").eval(), dsize(Q+"/model.safetensors")//4)
# orka: RVQ linears, fp16 tied embedding/head
import json as _j
man = _j.load(open(ART/"manifest.json"))
m = AutoModelForCausalLM.from_pretrained(Q, dtype=torch.float16).cuda().eval(); sd = dict(m.named_parameters())
lin = {t["name"] for t in man["tensors"] if t["name"].endswith(".weight") and "embed" not in t["name"]
       and "lm_head" not in t["name"] and len(t["shape"]) == 2 and "norm" not in t["name"]}
nrep = 0
for t in man["tensors"]:
    nm = t["name"]
    if nm in lin and nm in sd:
        with torch.no_grad():
            sd[nm].copy_(build_vq_linear(ART, t, bias=None, device="cpu").reconstruct_weight().to(torch.float16).cuda())
        nrep += 1
print(f"orka: reconstructed {nrep} linears (embedding kept fp16)", flush=True)
# orka disk = quantized linears (tensors/) + fp16 embedding
emb_fp16 = man and 151936*896*2
orka_disk = dsize(ART/"tensors") + emb_fp16
emit("orka-rvq", m, orka_disk)

json.dump(res, open("/tmp/claude-1000/-mnt-storage-codespace-code-orkait-orka-compiler/6e173d66-a10f-4ee7-88d8-e0cc77e0223f/scratchpad/qwen_results.json", "w"), indent=2)
print("DONE")
