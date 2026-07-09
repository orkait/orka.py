import numpy as np

from orka.autoquant.orchestrator import derive_config


def test_derives_int8_head_and_rvq_linears_no_llm():
    rng = np.random.default_rng(0)
    tensors = {
        "embed_out.weight": (np.float32, rng.standard_normal((512, 64)).astype("float32")),
        "model.layers.0.self_attn.q_proj.weight": (np.float32, rng.standard_normal((64, 64)).astype("float32")),
        "model.layers.0.input_layernorm.weight": (np.float32, rng.standard_normal((64,)).astype("float32")),
    }
    cfg = derive_config({n: w for n, (_, w) in tensors.items()}, objective="min-bits", use_llm=False)
    assert cfg["embed_out.weight"].method == "int8"
    assert cfg["model.layers.0.self_attn.q_proj.weight"].method == "rvq"
    assert cfg["model.layers.0.input_layernorm.weight"].keep_fp16
