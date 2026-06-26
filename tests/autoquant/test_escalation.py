from orka.autoquant.escalation import should_escalate, signature, Cache
from orka.autoquant.probes import Signals


def _sig(knee=3):
    return Signals(sqnr_curve={3: 31.0}, rd_knee_bits=knee, sensitivity=0.02)


def test_low_confidence_triggers():
    assert should_escalate(role_conf=0.3, signals=_sig(), policy_conf=0.3, regressed=False)


def test_high_confidence_no_trigger():
    assert not should_escalate(role_conf=1.0, signals=_sig(), policy_conf=1.0, regressed=False)


def test_regression_triggers():
    assert should_escalate(role_conf=1.0, signals=_sig(), policy_conf=1.0, regressed=True)


def test_signature_stable_and_bucketed():
    a = signature("attn.q", (768, 768), _sig(3), "min-bits")
    b = signature("attn.q", (768, 768), _sig(3), "min-bits")
    assert a == b


def test_cache_roundtrip(tmp_path):
    c = Cache(tmp_path / "cache.json")
    c.put("sig1", {"method": "rvq", "bits": 3})
    assert Cache(tmp_path / "cache.json").get("sig1")["bits"] == 3
