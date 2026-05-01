"""Tensor name -> family classifier (embedding/attention/mlp/other)."""

from __future__ import annotations


def classify_tensor_family(name: str) -> str:
    lowered = name.lower()
    if any(marker in lowered for marker in ("embed", "embedding", "wte", "wpe")):
        return "embedding"
    if any(
        marker in lowered
        for marker in (".mlp.", "mlp", "gate_proj", "up_proj", "down_proj", "c_fc")
    ):
        return "mlp"
    if any(
        marker in lowered
        for marker in (
            "attn",
            "attention",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "c_attn",
        )
    ):
        return "attention"
    return "other"
