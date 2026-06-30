from ui.backend.arch import build_architecture
from ui.backend.pipeline_steps import build_pipeline, build_tricks

DENSE = build_architecture({"vocab_size": 32, "tie_word_embeddings": False},
                           {"model.layers.0.mlp.down_proj.weight": (64, 64)})
HYBRID = build_architecture({"vocab_size": 32},
                            {"model.layers.0.mamba.A_log": (4,),
                             "model.layers.0.mamba.in_proj.weight": (16, 8)})


def test_pipeline_has_ordered_stages():
    ids = [s.id for s in build_pipeline(DENSE)]
    assert ids == ["load", "transform", "allocate", "codebook", "quantize", "strategies", "pack"]


def test_keep_head_trick_gated_by_tie():
    tricks = {t.id: t for t in build_tricks(HYBRID)}
    assert tricks["keep_head_fp16"].gated_by == "tied_head"
    assert tricks["lattice"].warn is not None
    assert "recurrent" in tricks["error_comp"].why.lower()
