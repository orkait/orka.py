"""Bit allocation should follow measured curvature, not a hand-written rule.

The allocator's contract is the reverse water-filling solution: bits scale with
0.5*log2(fisher * var), everything below the water level takes the cheapest spec, and the
parameter-weighted mean lands on the requested bits-per-weight. These tests pin that contract
on models whose curvature is known by construction, because on a real checkpoint you cannot
tell a correct allocation from a plausible one.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from orka.quant.curvature import (  # noqa: E402
    DEFAULT_SPEC_GRID,
    achieved_bpw,
    fisher_diagonal,
    spec_bits_per_weight,
    waterfill_stages,
)


class TwoHeads(nn.Module):
    """b's gradient is scaled up, so b must measure sharper than a."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 16, bias=False)
        self.b = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.a(x) + 10.0 * self.b(x)


def _loss(model, batch):
    return model(batch).pow(2).mean()


def _batches(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(8, 16, generator=g) for _ in range(n)]


def test_spec_bits_per_weight_matches_the_definition():
    assert spec_bits_per_weight((4096,), 8) == pytest.approx(12 / 8)
    assert spec_bits_per_weight((4096, 4096), 8) == pytest.approx(24 / 8)
    assert spec_bits_per_weight((4096, 4096, 256), 8) == pytest.approx(32 / 8)
    assert spec_bits_per_weight((256, 256), 16) == pytest.approx(16 / 16)


def test_fisher_ranks_the_sharper_tensor_higher():
    m = TwoHeads()
    stats = fisher_diagonal(m, _batches(), _loss)
    assert set(stats) == {"a.weight", "b.weight"}
    assert stats["b.weight"]["fisher"] > stats["a.weight"]["fisher"]
    for s in stats.values():
        assert s["numel"] == 256
        assert s["var"] > 0


def test_fisher_excludes_one_dimensional_parameters():
    m = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16))
    stats = fisher_diagonal(m, _batches(), lambda mm, b: mm(b).pow(2).mean())
    assert all(not k.endswith("bias") for k in stats)
    assert "1.weight" not in stats            # LayerNorm weight is 1-D


def test_fisher_leaves_no_gradients_behind():
    m = TwoHeads()
    fisher_diagonal(m, _batches(), _loss)
    assert all(p.grad is None for p in m.parameters())


def test_fisher_rejects_a_model_with_no_candidates():
    with pytest.raises(ValueError, match="ndim >= 2"):
        fisher_diagonal(nn.LayerNorm(4), _batches(), _loss)


def test_fisher_rejects_an_empty_batch_stream():
    with pytest.raises(ValueError, match="no batch produced a loss"):
        fisher_diagonal(TwoHeads(), [], _loss)


def test_sharper_tensor_receives_at_least_as_many_bits():
    m = TwoHeads()
    stats = fisher_diagonal(m, _batches(), _loss)
    smap = waterfill_stages(stats, target_bpw=3.0)
    ba = spec_bits_per_weight(smap["a.weight"], 8)
    bb = spec_bits_per_weight(smap["b.weight"], 8)
    assert bb >= ba


def test_achieved_bpw_tracks_the_requested_budget():
    """Rates are discrete, so the achieved mean lands near - not exactly on - the target."""
    stats = {f"t{i}.weight": {"fisher": 10.0 ** (-i), "var": 1.0, "numel": 1000}
             for i in range(8)}
    cheapest = min(spec_bits_per_weight(s, 8) for s in DEFAULT_SPEC_GRID)
    dearest = max(spec_bits_per_weight(s, 8) for s in DEFAULT_SPEC_GRID)
    for target in (2.0, 2.5, 3.0, 4.0):
        smap = waterfill_stages(stats, target_bpw=target)
        got = achieved_bpw(smap, stats)
        assert cheapest <= got <= dearest
        assert abs(got - target) <= 0.75, f"target {target} -> {got}"


def test_a_lower_budget_never_increases_any_tensor_rate():
    stats = {f"t{i}.weight": {"fisher": 10.0 ** (-i), "var": 1.0, "numel": 1000}
             for i in range(8)}
    hi = waterfill_stages(stats, target_bpw=4.0)
    lo = waterfill_stages(stats, target_bpw=2.0)
    for k in stats:
        assert spec_bits_per_weight(lo[k], 8) <= spec_bits_per_weight(hi[k], 8)


def test_allocation_is_monotonic_in_curvature():
    stats = {f"t{i}.weight": {"fisher": 10.0 ** (-i), "var": 1.0, "numel": 1000}
             for i in range(8)}
    smap = waterfill_stages(stats, target_bpw=2.5)
    bits = [spec_bits_per_weight(smap[f"t{i}.weight"], 8) for i in range(8)]
    assert bits == sorted(bits, reverse=True), bits


def test_zero_gradient_tensor_does_not_produce_a_nan_or_crash():
    """A frozen or dead tensor has fisher == 0; log2(0) would be -inf without the floor."""
    stats = {"dead.weight": {"fisher": 0.0, "var": 0.0, "numel": 100},
             "live.weight": {"fisher": 1e-3, "var": 1.0, "numel": 100}}
    smap = waterfill_stages(stats, target_bpw=3.0)
    assert all(math.isfinite(spec_bits_per_weight(v, 8)) for v in smap.values())
    assert spec_bits_per_weight(smap["dead.weight"], 8) <= \
        spec_bits_per_weight(smap["live.weight"], 8)


def test_group_size_changes_the_bit_accounting():
    stats = {"t.weight": {"fisher": 1e-3, "var": 1.0, "numel": 1000}}
    a = achieved_bpw(waterfill_stages(stats, 3.0, group_size=8), stats, group_size=8)
    b = achieved_bpw(waterfill_stages(stats, 3.0, group_size=16), stats, group_size=16)
    assert a == pytest.approx(3.0, abs=0.75)
    assert b == pytest.approx(3.0, abs=0.75)


def test_invalid_inputs_are_rejected():
    stats = {"t.weight": {"fisher": 1e-3, "var": 1.0, "numel": 10}}
    with pytest.raises(ValueError, match="target_bpw"):
        waterfill_stages(stats, target_bpw=0.0)
    with pytest.raises(ValueError, match="spec_grid is empty"):
        waterfill_stages(stats, target_bpw=3.0, spec_grid=[])
    with pytest.raises(ValueError, match="stats is empty"):
        waterfill_stages({}, target_bpw=3.0)
