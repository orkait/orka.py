import numpy as np
from orka.autoquant.probes import probe_tensor, Signals


def test_signals_shape_and_monotonic_sqnr():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((256, 512)).astype("float32")
    s = probe_tensor(W)
    assert isinstance(s, Signals)
    # more bits -> not-worse SQNR
    assert s.sqnr_at(8) >= s.sqnr_at(2) - 1e-6


def test_rd_knee_reaches_target():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((256, 512)).astype("float32")
    s = probe_tensor(W, sqnr_target_db=30.0)
    assert 1 <= s.rd_knee_bits <= 16
