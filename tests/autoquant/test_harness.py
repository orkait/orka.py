import json

import pytest

from orka.autoquant.harness import decide_with_llm
from orka.autoquant.probes import Signals


def _sig():
    return Signals(sqnr_curve={3: 31.0, 8: 45.0}, rd_knee_bits=3, sensitivity=0.02)


def test_uses_llm_verdict_when_valid():
    def fake_llm(messages):
        return json.dumps({"method": "rvq", "bits": 4, "stages": 3,
                           "normalization": "block-max", "keep_fp16": False,
                           "rationale": "needs headroom"})
    cfg = decide_with_llm("mlp.down", (768, 3072), _sig(), "min-bits", llm_fn=fake_llm)
    assert cfg.method == "rvq" and cfg.bits == 4 and cfg.source == "llm"


def test_invalid_llm_output_raises():
    def bad_llm(messages):
        return "not json"
    with pytest.raises(ValueError):
        decide_with_llm("mlp.down", (768, 3072), _sig(), "min-bits", llm_fn=bad_llm)
