from orka.autoquant.policy import decide
from orka.autoquant.probes import Signals


def _sig(knee=3):
    return Signals(sqnr_curve={3: 31.0, 8: 45.0}, rd_knee_bits=knee, sensitivity=0.02)


def test_out_head_is_int8_high_confidence():
    cfg, conf = decide("out-head", _sig())
    assert cfg.method == "int8" and conf == 1.0


def test_norm_kept_fp16():
    cfg, conf = decide("norm", _sig())
    assert cfg.keep_fp16 and cfg.method == "fp16"


def test_default_uses_rd_knee_bits():
    cfg, _ = decide("attn.q", _sig(knee=4))
    assert cfg.method == "rvq" and cfg.bits == 4


def test_sensitive_role_gets_extra_stage():
    cfg, _ = decide("mlp.down", _sig())
    base, _ = decide("attn.q", _sig())
    assert cfg.stages >= base.stages + 1


def test_unknown_is_low_confidence_fp16():
    cfg, conf = decide("unknown", _sig())
    assert cfg.keep_fp16 and conf < 0.5
