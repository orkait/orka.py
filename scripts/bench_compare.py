"""Apples-to-apples compression+speed+behaviour bench on identical pythia-160m (GPU).

Configs: fp16 dense | bitsandbytes-4bit (unsloth's quant backend) | orka RVQ.
Metrics: disk size, bits/param, VRAM, prefill tok/s, decode tok/s, perplexity (behaviour).
"""
import os, sys, time, json, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/home/kai/ai-models/hf-cache/hub/models--EleutherAI--pythia-160m/snapshots/50f5173d932e8e61f858120bcb800b97af589f46"
ORKA = "/home/kai/ai-models/pythia_orka_hf"
ORKA_ART = "/home/kai/ai-models/pythia_g8.orka"
DEV = "cuda"

tok = AutoTokenizer.from_pretrained(BASE)
PROMPT = "The history of artificial intelligence began in antiquity, with myths and stories of artificial beings endowed with intelligence by master craftsmen."
PPL_TEXT = (PROMPT + " Modern machine learning systems learn patterns from large amounts of data and "
            "improve their performance over time without being explicitly programmed for each task.")

def dsize(path):
    if os.path.isfile(path): return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(path) for f in fs)

@torch.no_grad()
def perplexity(model):
    ids = tok(PPL_TEXT, return_tensors="pt").input_ids.to(DEV)
    out = model(ids, labels=ids)
    return float(torch.exp(out.loss))

@torch.no_grad()
def bench(model, n_new=64):
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEV)
    model.generate(ids, max_new_tokens=8, do_sample=False)  # warmup
    torch.cuda.synchronize()
    # prefill: time first forward
    t0=time.time(); model(ids); torch.cuda.synchronize(); prefill_t=time.time()-t0
    n_prompt=ids.shape[1]
    t0=time.time(); out=model.generate(ids, max_new_tokens=n_new, do_sample=False); torch.cuda.synchronize()
    dt=time.time()-t0
    gen=out.shape[1]-n_prompt
    return n_prompt/prefill_t, gen/dt, out[0,n_prompt:n_prompt+10].tolist()

def vram_mb():
    torch.cuda.synchronize(); return torch.cuda.max_memory_allocated()/1e6

results=[]
def run(name, loader, disk, n_params_lin):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = loader().eval()
    pre, dec, toks = bench(m)
    ppl = perplexity(m)
    v = vram_mb()
    np_total = sum(p.numel() for p in m.parameters())
    results.append(dict(name=name, disk_mb=disk/1e6, vram_mb=v, prefill=pre, decode=dec,
                        ppl=ppl, first10=toks))
    print(f"{name:14s} disk={disk/1e6:7.1f}MB vram={v:7.1f}MB prefill={pre:7.1f} decode={dec:7.1f} ppl={ppl:7.3f} {toks[:5]}")
    del m; gc.collect(); torch.cuda.empty_cache()

# fp16 dense
run("fp16", lambda: AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to(DEV), dsize(BASE)*0+ 2*sum(p.numel() for p in AutoModelForCausalLM.from_pretrained(BASE).parameters()), 0)

# bnb 4-bit (unsloth backend)
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
def bnb_size():
    # 4-bit linears + fp16 embeds approx; report measured param bytes
    return None
run("bnb-nf4", lambda: AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map=DEV), dsize(BASE)//4, 0)

# orka RVQ
import orka.integrations.hf_quantizer  # registers
run("orka-rvq", lambda: AutoModelForCausalLM.from_pretrained(ORKA, dtype=torch.float16).to(DEV), dsize(ORKA_ART), 0)

json.dump(results, open("/tmp/claude-1000/-mnt-storage-codespace-code-orkait-orka-compiler/6e173d66-a10f-4ee7-88d8-e0cc77e0223f/scratchpad/bench_results.json","w"), indent=2)
print("\nWROTE results")
