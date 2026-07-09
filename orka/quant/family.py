"""Tensor name -> family classifier (embedding/attention/mlp/other).

Scope: this is the name->family taxonomy used for codebook GROUPING in family-mode packing
only. The quantize / keep-dense / skip-OBS identification (output head, recurrent/SSM) lives
in ``orka.quant.arch`` (ArchProfile); do not duplicate that logic here.
"""

from __future__ import annotations


def classify_tensor_family(name: str) -> str:
    lowered = name.lower()

    if (
        any(marker in lowered for marker in ("embed", "embedding", "wte", "wpe", "lm_head", "embed_out"))
        or lowered == "output.weight"
        or (lowered.endswith(".output.weight") and not any(x in lowered for x in ("attn", "attention", "mlp", "layer")))
    ):
        return "embedding"

    # MoE structure must be matched before the generic MLP markers below.
    if any(marker in lowered for marker in ("shared_expert", "sharedexpert")):
        return "shared_expert"

    if any(marker in lowered for marker in (".experts.", "experts/")):
        return "expert"

    if any(
        marker in lowered
        for marker in (
            "gate_proj",
            "up_proj",
            "down_proj",
            "c_fc",
            "fc1",
            "fc2",
            "fc_in",
            "fc_out",
            ".wi",
            ".wo",
            ".w1",
            ".w2",
            ".w3",
        )
    ):
        return "mlp"

    if any(marker in lowered for marker in (".gate", ".router", ".gating")):
        return "router"

    if any(
        marker in lowered
        for marker in (
            ".mlp.", "mlp", "gate_proj", "up_proj", "down_proj", "c_fc",
            "fc1", "fc2", "fc_in", "fc_out", ".wi", ".wo", ".w1", ".w2", ".w3"
        )
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
            "qkv",
            "query_key_value",
            "c_attn",
            "c_proj",
            "out_proj",
        )
    ):
        return "attention"

    return "other"
