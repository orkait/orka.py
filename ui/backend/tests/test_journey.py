from ui.backend import journey as J
from ui.backend.schema import Journey

CONFIG = {"vocab_size": 32, "tie_word_embeddings": True, "torch_dtype": "bfloat16"}
SHAPES = {"model.embed_tokens.weight": (32, 8), "lm_head.weight": (32, 8),
          "model.layers.0.mlp.down_proj.weight": (64, 64)}


def test_build_static_journey(monkeypatch):
    monkeypatch.setattr(J, "fetch_config", lambda m, token=None: CONFIG)
    monkeypatch.setattr(J, "fetch_shapes", lambda m, token=None: SHAPES)
    j = J.build_static_journey("x/y", bpw=3.0)
    assert isinstance(j, Journey)
    assert j.model.name == "x/y"
    assert j.model.tie_word_embeddings is True
    assert j.result.source == "estimated"
    assert len(j.pipeline) == 7
    assert any(t.id == "keep_head_fp16" for t in j.tricks)
