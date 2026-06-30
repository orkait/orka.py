import pytest

from ui.backend import live
from ui.backend.schema import Journey


def _static_journey():
    from ui.backend.arch import build_architecture
    from ui.backend.estimator import estimate
    from ui.backend.pipeline_steps import build_pipeline, build_tricks
    from ui.backend.schema import Journey, ModelMeta
    cfg = {"vocab_size": 32, "tie_word_embeddings": False, "torch_dtype": "bfloat16"}
    shapes = {"lm_head.weight": (32, 8), "model.layers.0.mlp.down_proj.weight": (64, 64)}
    meta = ModelMeta(name="x/y", params_total=4352, dtype="bfloat16", vocab_size=32,
                     tie_word_embeddings=False, fp16_bytes=8704)
    arch = build_architecture(cfg, shapes)
    return Journey(schema_version=1, model=meta, architecture=arch,
                   pipeline=build_pipeline(arch), tricks=build_tricks(arch),
                   result=estimate(meta, arch))


@pytest.mark.asyncio
async def test_run_live_emits_measured(monkeypatch):
    monkeypatch.setattr(live, "build_static_journey", lambda m, **k: _static_journey())
    monkeypatch.setattr(live, "_pack_and_eval",
                        lambda model, emit: {"ratio": 4.3, "fp16_mb": 988.0, "orka_mb": 230.0,
                                             "ppl_base": 20.9, "ppl_orka": 28.3,
                                             "ppl_ratio": 1.354, "trusted": True,
                                             "trust_reason": None})
    events = []
    j = await live.run_live("x/y", "jobid", lambda e: events.append(e))
    assert isinstance(j, Journey)
    assert j.result.source == "measured"
    assert j.result.ratio == 4.3
    assert j.result.trusted is True
