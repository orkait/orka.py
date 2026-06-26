from orka.autoquant.roles import classify_role


def test_output_head_split_from_embedding():
    assert classify_role("embed_out.weight", (50304, 768))[0] == "out-head"
    assert classify_role("lm_head.weight", (50304, 768))[0] == "out-head"
    assert classify_role("gpt_neox.embed_in.weight", (50304, 768))[0] == "in-embed"


def test_attention_subroles():
    assert classify_role("model.layers.0.self_attn.v_proj.weight", (768, 768))[0] == "attn.v"
    assert classify_role("model.layers.0.self_attn.o_proj.weight", (768, 768))[0] == "attn.o"


def test_mlp_subroles():
    assert classify_role("model.layers.0.mlp.down_proj.weight", (768, 3072))[0] == "mlp.down"
    assert classify_role("model.layers.0.mlp.gate_proj.weight", (3072, 768))[0] == "mlp.gate"


def test_norm_and_bias():
    assert classify_role("model.layers.0.input_layernorm.weight", (768,))[0] == "norm"
    assert classify_role("model.layers.0.mlp.down_proj.bias", (768,))[0] == "bias"


def test_unknown_low_confidence():
    role, conf = classify_role("mystery.tensor.foo", (123, 456))
    assert role == "unknown"
    assert conf < 0.5
