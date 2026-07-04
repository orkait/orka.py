import numpy as np

from ui.backend.probe import probe_tensor


def test_probe_shapes_and_rd_monotonic():
    rng = np.random.default_rng(0)
    flat = (rng.standard_normal(16384).astype("float32")) * 0.02
    p = probe_tensor(flat)

    assert len(p["distribution"]) == 33
    assert len(p["weights_block"]) == 11 and len(p["weights_block"][0]) == 40
    assert len(p["error_block"]) == 7 and len(p["error_block"][0]) == 40
    assert [d["bpw"] for d in p["rd"]] == [1.0, 2.0, 3.0, 4.0]

    sqnrs = [d["sqnr"] for d in p["rd"]]
    assert sqnrs == sorted(sqnrs)  # SQNR must improve as bpw rises

    assert all(len(v) == 3 for v in p["vectors3d"])
    assert all(len(c) == 3 for c in p["centroids3d"])
    assert 0.0 <= p["entropy_bits"] <= p["entropy_max"] == 8.0
    assert p["dist_range"][0] < p["dist_range"][1]


def test_probe_handles_tiny_input():
    # fewer vectors than the codebook size must not crash (K is capped to N)
    p = probe_tensor(np.linspace(-0.1, 0.1, 200, dtype="float32"))
    assert len(p["rd"]) == 4
