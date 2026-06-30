from ui.backend.arch import build_architecture
from ui.backend.estimator import estimate
from ui.backend.schema import ModelMeta

# Realistic sizes so the per-tensor codebook overhead (~131 KB) is negligible vs params
# (a 4608-param toy is dominated by overhead - not representative of any real model).
CONFIG = {"vocab_size": 2048, "tie_word_embeddings": True}
SHAPES = {"model.embed_tokens.weight": (2048, 512), "lm_head.weight": (2048, 512),
          "model.layers.0.mlp.down_proj.weight": (512, 2048)}


def _meta():
    total = 2048 * 512 + 2048 * 512 + 512 * 2048
    return ModelMeta(name="x/y", params_total=total, dtype="bfloat16",
                     vocab_size=2048, tie_word_embeddings=True, fp16_bytes=total * 2)


def test_estimate_monotonic_and_labeled():
    meta, arch = _meta(), build_architecture(CONFIG, SHAPES)
    r3 = estimate(meta, arch, bpw=3.0, keep_head=True)
    r25 = estimate(meta, arch, bpw=2.5, keep_head=True)
    assert r3.source == "estimated"
    assert r3.ratio > 1.0
    assert r25.ppl_ratio > r3.ppl_ratio       # lower bpw -> worse ppl
    assert r3.notes                            # carries provenance


def test_keep_head_costs_ratio():
    meta, arch = _meta(), build_architecture(CONFIG, SHAPES)
    keep = estimate(meta, arch, bpw=3.0, keep_head=True)
    drop = estimate(meta, arch, bpw=3.0, keep_head=False)
    assert drop.ratio > keep.ratio             # fp16 head is big -> protecting it lowers ratio


def test_lattice_on_hybrid_warns_worse():
    meta = _meta()
    arch = build_architecture({"vocab_size": 32}, {**SHAPES, "model.layers.0.mamba.A_log": (4,),
                              "model.layers.0.mamba.in_proj.weight": (16, 8)})
    base = estimate(meta, arch, bpw=3.0)
    lat = estimate(meta, arch, bpw=3.0, lattice=True)
    assert lat.ppl_ratio > base.ppl_ratio
