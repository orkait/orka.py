"""Tensor name -> family classifier (embedding/attention/mlp/other)."""

from __future__ import annotations


def classify_tensor_family(name: str) -> str:
    lowered = name.lower()
    
    # 1. Linguistic/LAVA Pillars & Output Heads
    if (
        any(marker in lowered for marker in ("embed", "embedding", "wte", "wpe", "lm_head", "embed_out"))
        or lowered == "output.weight"
        or (lowered.endswith(".output.weight") and not any(x in lowered for x in ("attn", "attention", "mlp", "layer")))
    ):
        return "embedding"
    
    # 2. MoE Specialized Structure (Checked before generic MLP)
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

    # 3. Standard Logic Components
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


# Single source of truth for two name-based predicates used by both the weight
# quantizers (VQ pack / E8 lattice) and the error-compensation gate, so they can
# never drift on what counts as "the output head" or "a recurrent/SSM block".
# Kept as a thin substring test (the pack pipeline sees a state_dict, not live
# modules) but centralized + documented + extensible, rather than copied inline.

_OUTPUT_HEAD_MARKERS = ("lm_head", "embed_out", "output.weight")
# Documented state-space / recurrent block names. Mamba/Mamba2 register the block as
# ``mixer`` in the reference impl; FalconH1 names it ``mamba``. Extend here (one place)
# for new recurrent families - over-matching only keeps plain VQ (safe), under-matching
# corrupts output (block-OBS over a nonlinear scan).
_RECURRENT_MARKERS = ("mamba", "mixer", "ssm", "state_space", "state-space")


def is_output_head(name: str) -> bool:
    """NAME-BASED FALLBACK for the output head - prefer the structural ``output_head_names``
    (detects by vocab-width shape) whenever shapes are available. Use this only when all
    you have is a name (e.g. a direct call without checkpoint shapes). True for the final
    logit projection (lm_head / embed_out / output.weight), which must stay high-precision:
    quantizing it explodes perplexity, and block-OBS over a softmax is invalid."""
    return any(m in name.lower() for m in _OUTPUT_HEAD_MARKERS)


def is_recurrent_block(name: str) -> bool:
    """NAME-BASED FALLBACK for recurrent/SSM tensors - prefer the structural
    ``recurrent_block_names`` (detects by sibling recurrence STATE params) whenever the
    full tensor-name set is available. This covers the documented SSM block names
    (mamba/mixer/ssm) but, being name-based, cannot know an arbitrary new one. Used to
    skip block-OBS error compensation (the SSM scan is nonlinear downstream, so the
    linear-output-error proxy is wrong); plain weight quant of the same matrices is fine."""
    return any(m in name.lower() for m in _RECURRENT_MARKERS)


# Recurrence STATE parameters: their presence as a sibling of a Linear marks that Linear
# as living inside a state-space / linear-recurrent block, INDEPENDENT of the block's name.
# These are the math-defining params (log state-transition matrix, selective timestep,
# token-shift decay) - Mamba/Mamba2 carry A_log + dt_bias; RWKV carries time_decay/time_mix.
# Far more robust than matching the block's name (mamba vs mixer vs ...). Lowercased.
_RECURRENCE_STATE_PARAMS = frozenset(
    {"a_log", "dt_bias", "dt_proj", "time_decay", "time_mix", "time_maa", "time_faaaa"}
)


def recurrent_block_names(tensor_names) -> set[str]:
    """STRUCTURAL recurrent/SSM detection: the set of tensor names that live inside a
    module which also owns a recurrence state param (``A_log``/``dt_bias``/...). Keys on
    the recurrence math, not the block name, so it holds for Mamba (``...mamba.in_proj``),
    pure-Mamba (``...mixer.in_proj``), or any future naming. Block-OBS error compensation
    must be skipped for these (nonlinear scan downstream); plain weight quant is still fine.

    Takes the FULL tensor-name set (including 1-D params like ``A_log`` that are not
    quantization candidates) and returns the names under each detected recurrent module."""
    names = list(tensor_names)
    blocks = set()  # module prefixes that directly own a recurrence state param
    for n in names:
        leaf = n.rsplit(".", 1)[-1].lower()
        if leaf in _RECURRENCE_STATE_PARAMS:
            blocks.add(n.rsplit(".", 1)[0])
    if not blocks:
        return set()
    prefixes = tuple(b + "." for b in blocks)
    return {n for n in names if n.startswith(prefixes)}


def output_head_names(tensor_shapes: dict, vocab_size: int | None = None) -> set[str]:
    """STRUCTURAL output-head / vocab-projection detection: the set of 2-D weight names
    whose output dimension equals ``vocab_size`` (the logit / embedding width). Catches
    both ``lm_head`` and the input embedding regardless of name. When ``vocab_size`` is
    unknown (no config), uses the dominant large output dim - vocab >> hidden in every
    standard LM, so the widest 2-D output is the vocab projection. These weights stay
    fp16 (quantizing the logit projection explodes perplexity)."""
    twod = {n: tuple(s) for n, s in tensor_shapes.items() if len(s) == 2}
    if not twod:
        return set()
    vocab = vocab_size if vocab_size else max(s[0] for s in twod.values())
    return {n for n, s in twod.items() if s[0] == vocab}
