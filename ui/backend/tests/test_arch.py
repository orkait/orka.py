from ui.backend.arch import build_architecture

CONFIG = {"vocab_size": 32, "tie_word_embeddings": True}
SHAPES = {
    "model.embed_tokens.weight": (32, 8),
    "lm_head.weight": (32, 8),
    "model.layers.0.self_attn.q_proj.weight": (8, 8),
    "model.layers.0.mlp.down_proj.weight": (8, 16),
    "model.layers.0.mamba.A_log": (4,),
    "model.layers.0.mamba.in_proj.weight": (16, 8),
}


def test_moe_detected_from_config_not_just_names():
    # granitemoe names experts 'block_sparse_moe.*' (no 'expert' substring) - must still
    # detect MoE via the config's expert count.
    a = build_architecture(
        {"vocab_size": 32, "num_local_experts": 8},
        {"model.layers.0.block_sparse_moe.input_linear.weight": (8, 32)},
    )
    assert a.flags["has_moe"] is True
    assert a.arch_class == "moe"


def test_flags_and_treatment():
    a = build_architecture(CONFIG, SHAPES)
    assert a.flags["tied_head"] is True
    assert a.flags["has_ssm"] is True
    assert a.arch_class == "hybrid"
    treat = {m.name: m.treatment for blk in a.layers for m in blk.modules}
    assert treat["lm_head.weight"] == "keep_fp16"
    assert treat["model.layers.0.mamba.in_proj.weight"] == "skip_error_comp"
    assert treat["model.layers.0.mlp.down_proj.weight"] == "quantize"
    assert any(f.family == "embedding" for f in a.param_breakdown)
