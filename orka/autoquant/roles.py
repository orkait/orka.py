"""Arch-agnostic tensor role classifier. Refines classify_tensor_family by splitting the
output head from the input embedding (they need opposite treatment) and resolving sub-roles.
Returns (role, confidence). Unknown names fall to ('unknown', low) -> escalation."""
from __future__ import annotations
from orka.quant import classify_tensor_family

_OUT_HEAD = ("lm_head", "embed_out", "output.weight")
_IN_EMBED = ("embed_in", "wte", "embed_tokens", "word_embeddings", "embedding")


def classify_role(name: str, shape: tuple[int, ...], tied: bool = False) -> tuple[str, float]:
    n = name.lower()
    if n.endswith(".bias") or n.endswith("_bias"):
        return "bias", 1.0
    if "norm" in n or "ln_" in n or n.endswith(".ln.weight"):
        return "norm", 1.0

    fam = classify_tensor_family(name)
    if fam == "embedding":
        if any(m in n for m in _OUT_HEAD):
            return "out-head", 1.0
        if any(m in n for m in _IN_EMBED):
            return "in-embed", 1.0
        return "unknown", 0.3
    if fam == "attention":
        for k in ("q_proj", "query"):
            if k in n:
                return "attn.q", 0.9
        for k in ("k_proj", "key"):
            if k in n:
                return "attn.k", 0.9
        for k in ("v_proj", "value"):
            if k in n:
                return "attn.v", 0.9
        for k in ("o_proj", "out_proj", "c_proj", ".dense"):
            if k in n:
                return "attn.o", 0.9
        return "attn.o", 0.5  # fused qkv / unknown attn linear -> conservative
    if fam == "mlp":
        if "down" in n or "fc2" in n or "fc_out" in n or ".wo" in n or ".w2" in n:
            return "mlp.down", 0.9
        if "gate" in n or ".w1" in n:
            return "mlp.gate", 0.9
        if "up" in n or "fc1" in n or "fc_in" in n or ".wi" in n or "c_fc" in n or ".w3" in n:
            return "mlp.up", 0.9
        return "mlp.up", 0.5
    return "unknown", 0.3
