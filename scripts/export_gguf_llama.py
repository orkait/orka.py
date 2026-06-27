"""Convert an .orka artifact (llama-arch model: SmolLM/Qwen/Llama) to a llama.cpp-native
GGUF with bit-plane RVQ linears. Mirrors export_gguf_llamacpp.py (gptneox) for LLM_ARCH_LLAMA.

llama specifics vs gptneox:
  * separate q/k/v/o (no fused qkv), gate/up/down MLP
  * q_proj/k_proj output rows are PERMUTED for llama.cpp's rope convention (the classic
    convert_hf permute) - RVQ output rows are independent so we reorder the M axis
  * tied output head (= token embedding) kept Q8 (never RVQ)
  * RMSNorm eps, rope.freq_base, GQA head_count_kv
"""
from __future__ import annotations
import sys, json, os, glob
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orka.inference.vq_linear import build_vq_linear
from orka.pipeline.decode import _decode_tensor
from orka.core._format import _pack_index_planes
from gguf import GGUFWriter, TokenType, GGMLQuantizationType
from gguf.quants import quantize as _gq

ART = Path(sys.argv[1]); CFG = Path(sys.argv[2]); OUT = sys.argv[3]
BASE = os.environ.get("ORKA_BASE", str(CFG))   # base model dir for embeddings/head (fp16 source)

manifest = json.loads((ART / "manifest.json").read_text())
cfg = json.loads((CFG / "config.json").read_text())
n_layer = cfg["num_hidden_layers"]; n_head = cfg["num_attention_heads"]
n_head_kv = cfg.get("num_key_value_heads", n_head)
n_embd = cfg["hidden_size"]; n_ff = cfg["intermediate_size"]; n_vocab = cfg["vocab_size"]
head_dim = n_embd // n_head

w = GGUFWriter(OUT, "llama")
w.add_context_length(cfg.get("max_position_embeddings", 2048))
w.add_embedding_length(n_embd); w.add_block_count(n_layer); w.add_feed_forward_length(n_ff)
w.add_head_count(n_head); w.add_head_count_kv(n_head_kv)
w.add_layer_norm_rms_eps(float(cfg.get("rms_norm_eps", 1e-5)))
w.add_rope_freq_base(float(cfg.get("rope_theta", 10000.0)))
w.add_rope_dimension_count(head_dim)
w.add_file_type(1)

def _base_tensor(name):
    sf = glob.glob(os.path.join(BASE, "*.safetensors"))[0]
    from safetensors import safe_open
    with safe_open(sf, "pt") as f:
        return f.get_tensor(name).float().numpy() if name in f.keys() else None

# --- vocab (gpt2/BPE from tokenizer.json) ---
tj = json.loads((CFG / "tokenizer.json").read_text())
vocab = tj["model"]["vocab"]; merges = tj["model"].get("merges", [])
id2tok = {v: k for k, v in vocab.items()}
added = {a["id"]: a["content"] for a in tj.get("added_tokens", [])}
toks, types = [], []
for i in range(n_vocab):
    if i in added: toks.append(added[i]); types.append(TokenType.USER_DEFINED)
    elif i in id2tok: toks.append(id2tok[i]); types.append(TokenType.NORMAL)
    else: toks.append(f"[PAD{i}]"); types.append(TokenType.UNUSED)
w.add_tokenizer_model("gpt2")
w.add_token_list(toks); w.add_token_types(types)
if merges: w.add_token_merges([m if isinstance(m, str) else " ".join(m) for m in merges])
w.add_bos_token_id(int(cfg.get("bos_token_id", 1))); w.add_eos_token_id(int(cfg.get("eos_token_id", 2)))

def perm_rows(n_h):
    # llama.cpp rope permute on output rows: [n_h,2,hd/2] -> swapaxes -> [n_h,hd/2,2]
    p = np.empty(n_h * head_dim, dtype=np.int64)
    for h in range(n_h):
        for s in range(2):
            for j in range(head_dim // 2):
                old = h * head_dim + s * (head_dim // 2) + j
                new = h * head_dim + j * 2 + s
                p[new] = old
    return p
PERM_Q = perm_rows(n_head); PERM_K = perm_rows(n_head_kv)

def _q8(name, arr):
    w.add_tensor(name, _gq(arr.astype(np.float32), GGMLQuantizationType.Q8_0), raw_dtype=GGMLQuantizationType.Q8_0)

# token_embd + output (tied): Q8 from base fp16
emb = _base_tensor("model.embed_tokens.weight").reshape(n_vocab, n_embd)
_q8("token_embd.weight", emb)
_q8("output.weight", emb)   # tied head

def pf(hf, lc):
    a = _base_tensor(hf)
    if a is not None: w.add_tensor(lc, a.astype(np.float32))
pf("model.norm.weight", "output_norm.weight")

NAME = {"self_attn.q_proj": ("attn_q", PERM_Q), "self_attn.k_proj": ("attn_k", PERM_K),
        "self_attn.v_proj": ("attn_v", None), "self_attn.o_proj": ("attn_output", None),
        "mlp.gate_proj": ("ffn_gate", None), "mlp.up_proj": ("ffn_up", None), "mlp.down_proj": ("ffn_down", None)}
GROUP = BLOCK = N_STAGES = None
n_quant = 0
for i in range(n_layer):
    P = f"model.layers.{i}."
    pf(P + "input_layernorm.weight", f"blk.{i}.attn_norm.weight")
    pf(P + "post_attention_layernorm.weight", f"blk.{i}.ffn_norm.weight")
    for hf_sub, (lc, perm) in NAME.items():
        tm = next(t for t in manifest["tensors"] if t["name"] == P + hf_sub + ".weight")
        layer = build_vq_linear(ART, tm, bias=None, device="cpu")
        M, K = layer.out_features, layer.in_features
        GPR, BPR = K // layer.group_size, K // layer.block_size
        for s in range(layer.n_stages):
            cb = getattr(layer, f"codebook_{s}").cpu().numpy().astype(np.float16)
            width = max(1, int(round(np.log2(cb.shape[0]))))
            idx = layer._stage_indices_int(s).cpu().numpy().astype(np.int64).reshape(M, GPR)
            if perm is not None: idx = idx[perm]
            lo, hi = _pack_index_planes(idx.reshape(-1), width)
            w.add_tensor(f"blk.{i}.{lc}.weight.idxlo{s}", lo.view(np.int8))
            w.add_tensor(f"blk.{i}.{lc}.weight.idxhi{s}", (hi if hi.size else np.zeros(1, np.uint8)).view(np.int8))
            w.add_tensor(f"blk.{i}.{lc}.weight.cb{s}", cb.reshape(-1))
        sc = layer.scales.cpu().numpy().astype(np.float16).reshape(M, BPR)
        if perm is not None: sc = sc[perm]
        w.add_tensor(f"blk.{i}.{lc}.weight.scales", sc.reshape(-1))
        GROUP, BLOCK, N_STAGES = layer.group_size, layer.block_size, layer.n_stages
        n_quant += 1

w.add_uint32("orka.rvq", 1); w.add_uint32("orka.n_stages", N_STAGES)
w.add_uint32("orka.group_size", GROUP); w.add_uint32("orka.block_size", BLOCK); w.add_uint32("orka.group_major", 0)
l0 = build_vq_linear(ART, next(t for t in manifest["tensors"] if t["name"] == "model.layers.0.self_attn.q_proj.weight"), bias=None, device="cpu")
for s in range(l0.n_stages):
    w.add_uint32(f"orka.cb_size.{s}", int(getattr(l0, f"codebook_{s}").shape[0]))

w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"wrote {OUT}: {n_quant} linears, group={GROUP} stages={N_STAGES} n_head_kv={n_head_kv}")
