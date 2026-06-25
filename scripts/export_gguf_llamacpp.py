"""Convert an .orka artifact into a llama.cpp-native GPTNeoX GGUF.

Unlike orka/export_gguf.py (HF names, for the standalone runner), this emits the tensor
names + KV that llama.cpp's gptneox loader expects (blk.N.attn_qkv.weight.idx0, ...), plus
a GPT2 BPE vocab, so a compressed model runs through the real engine (KV cache, llama-bench).

Two layout fixes vs the HF artifact:
  * QKV output rows are permuted from HF head-interleaved [q,k,v per head] to llama.cpp
    contiguous [all-Q | all-K | all-V]. RVQ output rows are independent, so we just reorder
    the M axis of idx/scales + the bias.
  * codebooks/scales stay fp16, indices int16 (the loader + custom op read those dtypes).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orka.inference.vq_linear import build_vq_linear
from orka.pipeline.decode import _decode_tensor

ART = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/kai/ai-models/pythia_g8.orka")
CFG = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/kai/ai-models/pythia_orka_hf")
OUT = sys.argv[3] if len(sys.argv) > 3 else "/home/kai/ai-models/pythia_orka_llamacpp.gguf"

from gguf import GGUFWriter, TokenType

manifest = json.loads((ART / "manifest.json").read_text())
cfg = json.loads((CFG / "config.json").read_text())
n_layer = cfg["num_hidden_layers"]; n_head = cfg["num_attention_heads"]
n_embd = cfg["hidden_size"]; n_ff = cfg["intermediate_size"]; n_vocab = cfg["vocab_size"]
head_dim = n_embd // n_head
rot = int(cfg.get("rotary_pct", cfg.get("partial_rotary_factor", 1.0)) * head_dim)

w = GGUFWriter(OUT, "gptneox")
w.add_context_length(cfg["max_position_embeddings"])
w.add_embedding_length(n_embd)
w.add_block_count(n_layer)
w.add_feed_forward_length(n_ff)
w.add_head_count(n_head)
w.add_layer_norm_eps(float(cfg.get("layer_norm_eps", 1e-5)))
w.add_rope_dimension_count(rot)
w.add_rope_freq_base(float(cfg.get("rotary_emb_base", 10000.0)))
w.add_parallel_residual(bool(cfg.get("use_parallel_residual", True)))
w.add_file_type(1)  # mostly F16

# --- orka KV ---
GROUP, BLOCK = None, None  # filled from first linear
N_STAGES = None
orka_meta_pending = True

# --- vocab (GPT2 BPE from tokenizer.json) ---
tj = json.loads((CFG / "tokenizer.json").read_text())
vocab = tj["model"]["vocab"]                       # token -> id
merges = tj["model"].get("merges", [])
id2tok = {v: k for k, v in vocab.items()}
added = {a["id"]: a["content"] for a in tj.get("added_tokens", [])}
toks, types = [], []
for i in range(n_vocab):
    if i in added:
        toks.append(added[i]); types.append(TokenType.USER_DEFINED)
    elif i in id2tok:
        toks.append(id2tok[i]); types.append(TokenType.NORMAL)
    else:
        toks.append(f"[PAD{i}]"); types.append(TokenType.UNUSED)
w.add_tokenizer_model("gpt2")
w.add_token_list(toks)
w.add_token_types(types)
if merges:
    w.add_token_merges([m if isinstance(m, str) else " ".join(m) for m in merges])
w.add_bos_token_id(0); w.add_eos_token_id(0)
w.add_add_bos_token(False)

def qkv_perm():
    p = np.empty(3 * n_embd, dtype=np.int64)
    for part in range(3):
        for h in range(n_head):
            for d in range(head_dim):
                lc = part * n_embd + h * head_dim + d
                hf = h * (3 * head_dim) + part * head_dim + d
                p[lc] = hf
    return p

NAME = {  # HF linear -> llama.cpp base name
    "attention.query_key_value": "attn_qkv",
    "attention.dense": "attn_output",
    "mlp.dense_h_to_4h": "ffn_up",
    "mlp.dense_4h_to_h": "ffn_down",
}
PERM = qkv_perm()

def passthrough(name):
    from safetensors import safe_open
    with safe_open(str(ART / "passthrough.safetensors"), "pt") as f:
        return f.get_tensor(name).float().numpy() if name in f.keys() else None

# dense tensors
emb = np.asarray(_decode_tensor(ART, next(t for t in manifest["tensors"] if t["name"]=="gpt_neox.embed_in.weight")), dtype=np.float32).reshape(n_vocab, n_embd)
w.add_tensor("token_embd.weight", emb.astype(np.float16))
# Output head is extremely precision-sensitive (a few % weight error => ppl blows up).
# Standard practice (llama.cpp keeps `output` at Q6/F16) - keep it fp16, never RVQ it.
# Prefer an fp16 head from ORKA_FP16_HEAD (the base model dir) if provided; else the
# artifact's reconstruction (which, if the packer quantized the head, will be degraded).
import os as _os
head_dir = _os.environ.get("ORKA_FP16_HEAD")
if head_dir:
    from safetensors import safe_open as _so
    import glob as _glob
    sf = _glob.glob(_os.path.join(head_dir, "*.safetensors"))[0]
    with _so(sf, "pt") as f:
        out = f.get_tensor("embed_out.weight").float().numpy().reshape(n_vocab, n_embd)
else:
    out = np.asarray(_decode_tensor(ART, next(t for t in manifest["tensors"] if t["name"]=="embed_out.weight")), dtype=np.float32).reshape(n_vocab, n_embd)
w.add_tensor("output.weight", out.astype(np.float16))
def pf(hf, lc):
    a = passthrough(hf)
    if a is not None: w.add_tensor(lc, a.astype(np.float32))
pf("gpt_neox.final_layer_norm.weight", "output_norm.weight")
pf("gpt_neox.final_layer_norm.bias",   "output_norm.bias")

n_quant = 0
for i in range(n_layer):
    P = f"gpt_neox.layers.{i}."
    pf(P+"input_layernorm.weight", f"blk.{i}.attn_norm.weight")
    pf(P+"input_layernorm.bias",   f"blk.{i}.attn_norm.bias")
    pf(P+"post_attention_layernorm.weight", f"blk.{i}.ffn_norm.weight")
    pf(P+"post_attention_layernorm.bias",   f"blk.{i}.ffn_norm.bias")
    for hf_sub, lc in NAME.items():
        tm = next(t for t in manifest["tensors"] if t["name"] == P+hf_sub+".weight")
        layer = build_vq_linear(ART, tm, bias=None, device="cpu")
        M, K = layer.out_features, layer.in_features
        GPR, BPR = K // layer.group_size, K // layer.block_size
        is_qkv = (lc == "attn_qkv")
        for s in range(layer.n_stages):
            idx = layer._stage_indices_int(s).cpu().numpy().astype(np.int16).reshape(M, GPR)
            if is_qkv: idx = idx[PERM]
            w.add_tensor(f"blk.{i}.{lc}.weight.idx{s}", idx.reshape(-1))
            cb = getattr(layer, f"codebook_{s}").cpu().numpy().astype(np.float16)
            w.add_tensor(f"blk.{i}.{lc}.weight.cb{s}", cb.reshape(-1))
        sc = layer.scales.cpu().numpy().astype(np.float16).reshape(M, BPR)
        if is_qkv: sc = sc[PERM]
        w.add_tensor(f"blk.{i}.{lc}.weight.scales", sc.reshape(-1))
        # bias (permute for qkv)
        b = passthrough(P+hf_sub+".bias")
        if b is not None:
            if is_qkv: b = b[PERM]
            bn = {"attn_qkv":"attn_qkv","attn_output":"attn_output","ffn_up":"ffn_up","ffn_down":"ffn_down"}[lc]
            w.add_tensor(f"blk.{i}.{bn}.bias", b.astype(np.float32))
        if orka_meta_pending:
            GROUP, BLOCK, N_STAGES = layer.group_size, layer.block_size, layer.n_stages
        n_quant += 1
    orka_meta_pending = False

w.add_uint32("orka.rvq", 1)
w.add_uint32("orka.n_stages", N_STAGES)
w.add_uint32("orka.group_size", GROUP)
w.add_uint32("orka.block_size", BLOCK)
w.add_uint32("orka.group_major", 0)
# codebook sizes per stage (read from layer 0 qkv)
tm0 = next(t for t in manifest["tensors"] if t["name"]=="gpt_neox.layers.0.attention.query_key_value.weight")
l0 = build_vq_linear(ART, tm0, bias=None, device="cpu")
for s in range(l0.n_stages):
    cb = getattr(l0, f"codebook_{s}")
    w.add_uint32(f"orka.cb_size.{s}", int(cb.shape[0]))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote {OUT}: {n_quant} quantized linears, group={GROUP} block={BLOCK} stages={N_STAGES} rot={rot}")
