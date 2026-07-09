from orka.autoquant.probes import Signals
from orka.autoquant.refine import Quality, attribute, escalate_cfg, meets, refine
from orka.autoquant.schema import TensorConfig


def _rvq(bits=3, stages=2):
    return TensorConfig("rvq", bits, stages, "block-max", False, "policy", 0.7, "x")


def _sig(sqnr):
    return Signals(sqnr_curve={2: sqnr, 3: sqnr, 4: sqnr + 5, 6: sqnr + 10, 8: sqnr + 15},
                   rd_knee_bits=3, sensitivity=0.02)


def test_meets_min_bits():
    assert meets(Quality(kl=0.01, top1=0.99), "min-bits", target=0.02)
    assert not meets(Quality(kl=0.05, top1=0.9), "min-bits", target=0.02)


def test_attribute_ranks_lowest_sqnr():
    cfg = {"a": _rvq(), "b": _rvq()}
    sig = {"a": _sig(10.0), "b": _sig(40.0)}
    assert attribute(cfg, sig, k=1) == ["a"]


def test_escalate_bumps_bits_then_stage():
    c = escalate_cfg(_rvq(bits=3, stages=2))
    assert c.bits == 4 and c.source == "refine"
    c8 = escalate_cfg(_rvq(bits=8, stages=2))
    assert c8.bits == 8 and c8.stages == 3


def test_refine_improves_then_converges():
    cfg = {"a": _rvq(bits=2)}
    sig = {"a": _sig(8.0)}
    # pulse improves as bits go up; pack_fn identity returns the cfg
    def pack_fn(c): return c
    def pulse_fn(c): return Quality(kl=max(0.0, 0.1 - 0.03 * c["a"].bits), top1=0.9)
    final, q, rounds = refine(cfg, sig, "min-bits", target=0.02,
                              pack_fn=pack_fn, pulse_fn=pulse_fn, max_rounds=5)
    assert q.kl <= 0.02
    assert final["a"].bits > 2


def test_refine_caps_when_stuck():
    cfg = {"a": _rvq(bits=2)}
    sig = {"a": _sig(8.0)}
    def pack_fn(c): return c
    def pulse_fn(c): return Quality(kl=0.5, top1=0.5)   # never improves
    final, q, rounds = refine(cfg, sig, "min-bits", target=0.02,
                              pack_fn=pack_fn, pulse_fn=pulse_fn, max_rounds=5)
    assert rounds <= 5 and q.kl == 0.5
