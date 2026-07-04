from ui.backend.schema import Journey

EXAMPLE = {
    "schema_version": 1,
    "model": {"name": "x/y", "params_total": 100, "dtype": "bfloat16",
              "vocab_size": 32, "tie_word_embeddings": True, "fp16_bytes": 200},
    "architecture": {"arch_class": "dense", "flags": {"tied_head": True, "has_moe": False, "has_ssm": False},
                     "param_breakdown": [{"family": "mlp", "params": 100, "pct": 100.0, "role": ""}],
                     "layers": [{"index": 0, "modules": [
                         {"name": "model.layers.0.mlp.down_proj", "shape": [8, 8],
                          "family": "mlp", "treatment": "quantize"}]}],
                     "partial": False},
    "pipeline": [{"id": "load", "title": "Load", "summary": "..."}],
    "tricks": [{"id": "bpw", "label": "Bits/weight", "kind": "scalar", "default": 3.0,
                "applies": True, "why": "", "warn": None, "gated_by": None}],
    "result": {"source": "estimated", "bpw": 3.0, "ratio": 4.3, "fp16_mb": 0.2, "orka_mb": 0.05,
               "ppl_base": None, "ppl_orka": None, "ppl_ratio": 1.35,
               "trusted": None, "trust_reason": None, "notes": ["estimated"]},
}


def test_round_trip():
    j = Journey.model_validate(EXAMPLE)
    assert j.schema_version == 1
    assert j.model.tie_word_embeddings is True
    assert j.architecture.layers[0].modules[0].treatment == "quantize"
    assert j.result.source == "estimated"
    assert Journey.model_validate(j.model_dump()).result.ratio == 4.3
