"""Convert an .orka artifact of a Qwen3-family model into an orka.cpp GGUF.

Counterpart of export_gguf_llamacpp.py (gptneox) for the qwen3 llama.cpp arch:
separate q/k/v/o + gate/up/down linears (no QKV permute - qwen uses NEOX rope),
per-layer q_norm/k_norm passthroughs, RMSNorm, GQA head counts, qwen2 BPE pre-tokenizer.

Usage:
    python scripts/export_gguf_llamacpp_qwen3.py <artifact.orka> <hf_model_dir> <out.gguf>

Embeddings/head: token_embd from the artifact (passthrough when tied / decoded when
packed) as Q8_0; output.weight emitted explicitly as Q8_0 from the same source (RVQ
heads are catastrophic; int8 is lossless - see the gptneox converter rationale).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orka.inference.vq_linear import build_vq_linear  # noqa: E402
from orka.pipeline.decode import _decode_tensor  # noqa: E402

ART = Path(sys.argv[1])
CFG = Path(sys.argv[2])
OUT = sys.argv[3]

from gguf import GGMLQuantizationType, GGUFWriter, TokenType  # noqa: E402
from gguf.quants import quantize as _gq  # noqa: E402
from safetensors import safe_open  # noqa: E402

manifest = json.loads((ART / "manifest.json").read_text())
cfg = json.loads((CFG / "config.json").read_text())

n_layer = cfg["num_hidden_layers"]
n_head = cfg["num_attention_heads"]
n_kv = cfg["num_key_value_heads"]
n_embd = cfg["hidden_size"]
head_dim = cfg.get("head_dim") or n_embd // n_head
n_ff = cfg["intermediate_size"]
n_vocab = cfg["vocab_size"]

w = GGUFWriter(OUT, "qwen3")
w.add_context_length(cfg["max_position_embeddings"])
w.add_embedding_length(n_embd)
w.add_block_count(n_layer)
w.add_feed_forward_length(n_ff)
w.add_head_count(n_head)
w.add_head_count_kv(n_kv)
w.add_key_length(head_dim)
w.add_value_length(head_dim)
w.add_rope_freq_base(float(cfg.get("rope_theta", 1e6)))
w.add_layer_norm_rms_eps(float(cfg.get("rms_norm_eps", 1e-6)))
w.add_file_type(1)

# --- vocab: qwen2 BPE ---
tj = json.loads((CFG / "tokenizer.json").read_text())
vocab = tj["model"]["vocab"]
merges = tj["model"].get("merges", [])
id2tok = {v: k for k, v in vocab.items()}
added = {a["id"]: a["content"] for a in tj.get("added_tokens", [])}
toks, types = [], []
for i in range(n_vocab):
    if i in added:
        toks.append(added[i]); types.append(TokenType.CONTROL)
    elif i in id2tok:
        toks.append(id2tok[i]); types.append(TokenType.NORMAL)
    else:
        toks.append(f"[PAD{i}]"); types.append(TokenType.UNUSED)
w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre("qwen2")
w.add_token_list(toks)
w.add_token_types(types)
if merges:
    w.add_token_merges([m if isinstance(m, str) else " ".join(m) for m in merges])
gen = {}
gp = CFG / "generation_config.json"
if gp.exists():
    gen = json.loads(gp.read_text())
eos = gen.get("eos_token_id", cfg.get("eos_token_id", 151643))
if isinstance(eos, list):
    eos = eos[0]
w.add_eos_token_id(int(eos))
w.add_bos_token_id(int(gen.get("bos_token_id") or cfg.get("bos_token_id") or eos))
w.add_add_bos_token(False)


def passthrough(name):
    with safe_open(str(ART / "passthrough.safetensors"), "pt") as f:
        return f.get_tensor(name).float().numpy() if name in f.keys() else None


def dense_full(hf_name, rows, cols):
    """Full-precision tensor from the artifact: passthrough first, else RVQ decode."""
    a = passthrough(hf_name)
    if a is not None:
        return a.reshape(rows, cols)
    tm = next((t for t in manifest["tensors"] if t["name"] == hf_name), None)
    if tm is None:
        raise KeyError(f"{hf_name} neither passthrough nor packed")
    return np.asarray(_decode_tensor(ART, tm), dtype=np.float32).reshape(rows, cols)


def add_q8(name, arr):
    w.add_tensor(name, _gq(arr.astype(np.float32), GGMLQuantizationType.Q8_0),
                 raw_dtype=GGMLQuantizationType.Q8_0)


embed = dense_full("model.embed_tokens.weight", n_vocab, n_embd)
add_q8("token_embd.weight", embed)
add_q8("output.weight",
       embed if cfg.get("tie_word_embeddings") else dense_full("lm_head.weight", n_vocab, n_embd))


def pf(hf, lc):
    a = passthrough(hf)
    if a is not None:
        w.add_tensor(lc, a.astype(np.float32))


pf("model.norm.weight", "output_norm.weight")

NAME = {  # HF linear suffix -> llama.cpp base name (no permutes: NEOX rope)
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}

GROUP = BLOCK = N_STAGES = None
n_quant = 0
for i in range(n_layer):
    P = f"model.layers.{i}."
    pf(P + "input_layernorm.weight", f"blk.{i}.attn_norm.weight")
    pf(P + "post_attention_layernorm.weight", f"blk.{i}.ffn_norm.weight")
    pf(P + "self_attn.q_norm.weight", f"blk.{i}.attn_q_norm.weight")
    pf(P + "self_attn.k_norm.weight", f"blk.{i}.attn_k_norm.weight")
    for hf_sub, lc in NAME.items():
        tm = next(t for t in manifest["tensors"] if t["name"] == P + hf_sub + ".weight")
        layer = build_vq_linear(ART, tm, bias=None, device="cpu")
        assert layer.corr_col.numel() == 0, f"{tm['name']}: sidecars unsupported in gguf format"
        M, K = layer.out_features, layer.in_features
        GPR = K // layer.group_size
        from orka.core._format import _pack_index_planes
        for s in range(layer.n_stages):
            cb = getattr(layer, f"codebook_{s}").cpu().numpy().astype(np.float16)
            width = max(1, int(round(np.log2(cb.shape[0]))))
            idx = layer._stage_indices_int(s).cpu().numpy().astype(np.int64).reshape(M, GPR)
            lo, hi = _pack_index_planes(idx.reshape(-1), width)
            w.add_tensor(f"blk.{i}.{lc}.weight.idxlo{s}", lo.view(np.int8))
            hi = hi if hi.size else np.zeros(1, np.uint8)
            w.add_tensor(f"blk.{i}.{lc}.weight.idxhi{s}", hi.view(np.int8))
            w.add_tensor(f"blk.{i}.{lc}.weight.cb{s}", cb.reshape(-1))
        # Flat scales = numel/block, matching orka.llama qwen3.cpp's
        # sc_len = M*K/o_block. Reshaping to (M, K//block) first only equals this
        # when block divides K; slrq-block (block 32) always does, block-max may not.
        sc = layer.scales.cpu().numpy().astype(np.float16).reshape(-1)
        w.add_tensor(f"blk.{i}.{lc}.weight.scales", sc)
        if GROUP is None:
            GROUP, BLOCK, N_STAGES = layer.group_size, layer.block_size, layer.n_stages
        n_quant += 1

w.add_uint32("orka.rvq", 1)
w.add_uint32("orka.n_stages", N_STAGES)
w.add_uint32("orka.group_size", GROUP)
w.add_uint32("orka.block_size", BLOCK)
w.add_uint32("orka.group_major", 0)
tm0 = next(t for t in manifest["tensors"] if t["name"].endswith("self_attn.q_proj.weight"))
l0 = build_vq_linear(ART, tm0, bias=None, device="cpu")
for s in range(l0.n_stages):
    w.add_uint32(f"orka.cb_size.{s}", int(getattr(l0, f"codebook_{s}").shape[0]))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote {OUT}: {n_quant} quantized linears, group={GROUP} block={BLOCK} stages={N_STAGES}")
