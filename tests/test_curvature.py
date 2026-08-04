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
        smap = waterfill_stages(stats, target_bpw=target, min_sqnr_db=None)
        got = achieved_bpw(smap, stats)
        assert cheapest <= got <= dearest
        assert abs(got - target) <= 0.75, f"target {target} -> {got}"


def test_a_lower_budget_never_increases_any_tensor_rate():
    stats = {f"t{i}.weight": {"fisher": 10.0 ** (-i), "var": 1.0, "numel": 1000}
             for i in range(8)}
    hi = waterfill_stages(stats, target_bpw=4.0, min_sqnr_db=None)
    lo = waterfill_stages(stats, target_bpw=2.0, min_sqnr_db=None)
    for k in stats:
        assert spec_bits_per_weight(lo[k], 8) <= spec_bits_per_weight(hi[k], 8)


def test_allocation_is_monotonic_in_curvature():
    stats = {f"t{i}.weight": {"fisher": 10.0 ** (-i), "var": 1.0, "numel": 1000}
             for i in range(8)}
    smap = waterfill_stages(stats, target_bpw=2.5, min_sqnr_db=None)
    bits = [spec_bits_per_weight(smap[f"t{i}.weight"], 8) for i in range(8)]
    assert bits == sorted(bits, reverse=True), bits


def test_zero_gradient_tensor_does_not_produce_a_nan_or_crash():
    """A frozen or dead tensor has fisher == 0; log2(0) would be -inf without the floor."""
    stats = {"dead.weight": {"fisher": 0.0, "var": 0.0, "numel": 100},
             "live.weight": {"fisher": 1e-3, "var": 1.0, "numel": 100}}
    smap = waterfill_stages(stats, target_bpw=3.0, min_sqnr_db=None)
    assert all(math.isfinite(spec_bits_per_weight(v, 8)) for v in smap.values())
    assert spec_bits_per_weight(smap["dead.weight"], 8) <= \
        spec_bits_per_weight(smap["live.weight"], 8)


def test_group_size_changes_the_bit_accounting():
    stats = {"t.weight": {"fisher": 1e-3, "var": 1.0, "numel": 1000}}
    a = achieved_bpw(waterfill_stages(stats, 3.0, group_size=8, min_sqnr_db=None),
                     stats, group_size=8)
    b = achieved_bpw(waterfill_stages(stats, 3.0, group_size=16, min_sqnr_db=None),
                     stats, group_size=16)
    assert a == pytest.approx(3.0, abs=0.75)
    assert b == pytest.approx(3.0, abs=0.75)


def test_invalid_inputs_are_rejected():
    stats = {"t.weight": {"fisher": 1e-3, "var": 1.0, "numel": 10}}
    with pytest.raises(ValueError, match="target_bpw"):
        waterfill_stages(stats, target_bpw=0.0, min_sqnr_db=None)
    with pytest.raises(ValueError, match="spec_grid is empty"):
        waterfill_stages(stats, target_bpw=3.0, spec_grid=[], min_sqnr_db=None)
    with pytest.raises(ValueError, match="stats is empty"):
        waterfill_stages({}, target_bpw=3.0, min_sqnr_db=None)


from orka.autoquant.roles import classify_role  # noqa: E402
from orka.quant.curvature import waterfill_with_roles  # noqa: E402

SHAPES = {
    "model.embed_tokens.weight": (4000, 1000),
    "model.layers.0.self_attn.q_proj.weight": (1000, 1000),
    "model.layers.0.mlp.down_proj.weight": (1000, 1000),
    "model.layers.0.conv.in_proj.weight": (2000, 1000),
    "model.layers.0.conv.conv.weight": (1000, 1, 3),
    "lm_head.weight": (4000, 1000),
    "model.norm.weight": (1024,),
}


def _mixed_stats():
    return {
        "model.embed_tokens.weight": {"fisher": 1e-7, "var": 1.0, "numel": 4_000_000},
        "model.layers.0.self_attn.q_proj.weight": {"fisher": 1e-4, "var": 1.0, "numel": 1_000_000},
        "model.layers.0.mlp.down_proj.weight": {"fisher": 1e-4, "var": 1.0, "numel": 1_000_000},
        "model.layers.0.conv.in_proj.weight": {"fisher": 1e-5, "var": 1.0, "numel": 2_000_000},
        "model.layers.0.conv.conv.weight": {"fisher": 1e-2, "var": 1.0, "numel": 3_000},
        "lm_head.weight": {"fisher": 1e-3, "var": 1.0, "numel": 4_000_000},
        "model.norm.weight": {"fisher": 1e-3, "var": 1.0, "numel": 1_024},
    }


def _role(n):
    return classify_role(n, SHAPES.get(n, (8, 8)))[0]


def test_conv_block_projections_are_no_longer_unknown():
    assert classify_role("lfm2.layers.0.conv.in_proj.weight", (3072, 1024))[0] == "conv.in"
    assert classify_role("lfm2.layers.0.conv.out_proj.weight", (1024, 1024))[0] == "conv.out"
    assert classify_role("backbone.layers.3.mixer.in_proj.weight", (4096, 2048))[0] == "conv.in"


def test_depthwise_kernels_match_on_shape_not_name():
    for nm in ("lfm2.layers.0.conv.conv.weight", "backbone.layers.0.mixer.conv1d.weight"):
        assert classify_role(nm, (1024, 1, 3))[0] == "conv.depthwise"


def test_ordinary_roles_are_not_captured_by_the_conv_matcher():
    assert classify_role("model.layers.0.self_attn.o_proj.weight", (1024, 1024))[0] == "attn.o"
    assert classify_role("model.layers.0.mlp.down_proj.weight", (1024, 4096))[0] == "mlp.down"
    assert classify_role("model.embed_tokens.weight", (32000, 1024))[0] == "in-embed"
    assert classify_role("lm_head.weight", (32000, 1024))[0] == "out-head"


def test_rails_keep_head_norm_and_depthwise_out_of_the_budget():
    stages, dense = waterfill_with_roles(_mixed_stats(), 3.0, _role)
    assert "lm_head.weight" in dense
    assert "model.norm.weight" in dense
    assert "model.layers.0.conv.conv.weight" in dense
    assert set(stages) == {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.conv.in_proj.weight",
    }


def test_conv_projections_now_get_bits_instead_of_fp16():
    stages, dense = waterfill_with_roles(_mixed_stats(), 3.0, _role)
    assert "model.layers.0.conv.in_proj.weight" in stages
    assert "model.layers.0.conv.in_proj.weight" not in dense


def test_sensitive_roles_receive_an_extra_stage():
    stages, _ = waterfill_with_roles(_mixed_stats(), 2.0, _role, min_sqnr_db=None)
    q = spec_bits_per_weight(stages["model.layers.0.self_attn.q_proj.weight"], 8)
    d = spec_bits_per_weight(stages["model.layers.0.mlp.down_proj.weight"], 8)
    assert d > q, f"sensitive mlp.down got {d} bpw vs q_proj {q}"


def test_extra_stage_is_a_no_op_at_the_grid_ceiling():
    stages, _ = waterfill_with_roles(_mixed_stats(), 4.5, _role)
    top = max(spec_bits_per_weight(s, 8) for s in DEFAULT_SPEC_GRID)
    d = stages["model.layers.0.mlp.down_proj.weight"]
    assert spec_bits_per_weight(d, 8) <= top
    assert tuple(d) in {tuple(x) for x in DEFAULT_SPEC_GRID}


def test_budget_applies_to_quantizable_tensors_only():
    stats = _mixed_stats()
    stages, dense = waterfill_with_roles(stats, 2.5, _role, min_sqnr_db=None)
    got = achieved_bpw(stages, {k: stats[k] for k in stages})
    assert 1.5 <= got <= 5.0
    assert dense


def test_floor_prevents_starving_a_large_tensor():
    """Reproduces the LFM2.5-2.6B shape: a minority of very large, low-curvature tensors
    alongside many higher-curvature ones. Water-filling starved the large ones to the grid
    floor (1.5 bpw / 7.7 dB) because the rate rule carries no tensor-size term."""
    stats = {}
    for i in range(10):                       # the FFN band that got starved
        stats[f"ffn{i}.weight"] = {"fisher": 1e-9, "var": 1.0, "numel": 22_000_000}
    for i in range(20):                       # everything else, higher curvature
        stats[f"other{i}.weight"] = {"fisher": 1e-4, "var": 1.0, "numel": 60_000_000}

    unfloored = waterfill_stages(stats, 3.0, min_sqnr_db=None)
    starved = [k for k, v in unfloored.items() if spec_bits_per_weight(v, 8) < 3.0]
    assert starved, "expected water-filling to starve the low-curvature band"
    assert any(k.startswith("ffn") for k in starved)

    floored = waterfill_stages(stats, 3.5, min_sqnr_db=14.0)
    assert all(spec_bits_per_weight(v, 8) >= 3.0 for v in floored.values())


def test_floor_is_applied_to_every_tensor():
    stats = {f"t{i}.weight": {"fisher": 10.0 ** (-i - 4), "var": 1.0, "numel": 1_000_000}
             for i in range(8)}
    smap = waterfill_stages(stats, 3.0, min_sqnr_db=14.0)
    assert all(spec_bits_per_weight(v, 8) >= 3.0 for v in smap.values())


def test_target_below_the_floor_is_refused_not_silently_starved():
    stats = {"a.weight": {"fisher": 1e-5, "var": 1.0, "numel": 1_000_000}}
    with pytest.raises(ValueError, match="below the .* floor"):
        waterfill_stages(stats, 2.0, min_sqnr_db=14.0)


def test_floor_can_be_disabled():
    stats = {"a.weight": {"fisher": 1e-9, "var": 1.0, "numel": 9_000_000},
             "b.weight": {"fisher": 1e-2, "var": 1.0, "numel": 1_000_000}}
    smap = waterfill_stages(stats, 2.0, min_sqnr_db=None)
    assert min(spec_bits_per_weight(v, 8) for v in smap.values()) < 2.5


def test_unreachable_sqnr_floor_reports_the_grid_ceiling():
    stats = {"a.weight": {"fisher": 1e-5, "var": 1.0, "numel": 1_000}}
    with pytest.raises(ValueError, match="richest spec"):
        waterfill_stages(stats, 4.5, min_sqnr_db=60.0)


def test_roles_wrapper_applies_the_floor():
    stats = _mixed_stats()
    stages, _ = waterfill_with_roles(stats, 3.0, _role, min_sqnr_db=14.0)
    assert all(spec_bits_per_weight(v, 8) >= 3.0 for v in stages.values())
