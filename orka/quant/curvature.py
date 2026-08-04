"""Curvature-driven bit allocation: reverse water-filling over a diagonal Fisher estimate.

Orka's allocator spends bits by heuristic (family/depth rules, or a uniform spec). What the
task actually cares about is loss curvature. Expanding the task loss around trained weights,
with ``D = w_hat - w`` and ``grad ~ 0`` at a minimum::

    L(w_hat) - L(w)  ~  0.5 * D^T H D

so the distortion that matters is Hessian-weighted, not Euclidean. Combining that with the
Gaussian rate-distortion law ``D(R) = var * 2^(-2R)`` gives total task damage::

    D_total = sum_t  F_t * W_t * var_t * 2^(-2 R_t)

for per-tensor rate ``R_t``, mean diagonal Fisher ``F_t`` and parameter count ``W_t``.
Minimising under a budget ``sum_t W_t R_t <= B`` yields, by Lagrange::

    dD/dR_t = -2 ln2 * F_t W_t var_t 2^(-2R_t) = -lambda W_t
    =>  R_t = 0.5 * log2(F_t * var_t) + c

Bits scale with the LOG of curvature x variance, and everything below the water level takes
the cheapest spec. ``c`` is solved by bisection to hit a target bits-per-weight.

MEASURED on LiquidAI/LFM2.5-Encoder-230M (229.7M params, 918.8 MB fp32), paired retrieval
Recall@1 over 1024 pairs with a McNemar test against fp32:

    hand-tuned depth/module heuristic   95.0 MB   9.67x   R@1 0.326   (worse AND bigger)
    this allocator at 2.6 bpw           84.2 MB  10.91x   R@1 0.297   p=0.844, identical
    this allocator at 2.2 bpw           69.5 MB  13.22x   R@1 0.271   p=0.034, degraded

It also corrected two heuristic errors: the tiny depthwise conv kernels (0.01% of that model)
are its SHARPEST tensors and deserve max bits at no cost, while ``embed_tokens`` is among the
flattest - which independently reproduced an ablation finding that the embedding contributes
no measurable damage.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable

#: Measured RVQ rate-distortion on LFM2.5-2.6B-Base: SQNR ~= 5.45*bpw - 0.5 dB across
#: 1.5-4.5 bpw. Matches the ~5.5 dB/bpw slope seen on earlier models. Used only to turn an
#: SQNR floor into a bits floor; pass explicit values if a model's curve differs.
RD_SLOPE_DB_PER_BIT = 5.45
RD_INTERCEPT_DB = -0.5

#: Below this, orka's own priors call the result catastrophic at model scale.
DEFAULT_MIN_SQNR_DB = 14.0

#: Candidate RVQ specs -> bits per weight at a given group size. Stage k costs log2(K) bits
#: per group, so bits-per-weight is sum(log2(K)) / group_size.
DEFAULT_SPEC_GRID: tuple[tuple[int, ...], ...] = (
    (4096,),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (4096, 4096, 256),
    (4096, 4096, 4096),
)


def spec_bits_per_weight(spec: Iterable[int], group_size: int) -> float:
    return sum(math.log2(int(k)) for k in spec) / float(group_size)


def fisher_diagonal(
    model,
    batches: Iterable,
    loss_fn: Callable,
    *,
    min_ndim: int = 2,
) -> dict[str, dict]:
    """Mean diagonal empirical Fisher ``E[(dL/dw)^2]`` per parameter tensor.

    Architecture-agnostic by construction: the caller supplies ``loss_fn(model, batch) ->
    scalar loss``, so this works for causal LMs, masked LMs, encoders, or anything else with a
    differentiable objective. Returns ``{name: {"fisher", "var", "numel"}}`` for tensors of at
    least ``min_ndim`` dimensions - 1-D norms and biases are excluded because they are not
    quantization candidates.

    Backprop is required, so the model must not be under ``torch.no_grad``. Gradients are
    zeroed per batch and left cleared on exit.
    """
    import torch

    params = {n: p for n, p in model.named_parameters() if p.ndim >= min_ndim}
    if not params:
        raise ValueError(f"model has no parameters with ndim >= {min_ndim}")
    acc = {n: torch.zeros((), device=p.device, dtype=torch.float64)
           for n, p in params.items()}

    seen = 0
    for batch in batches:
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch)
        if loss is None:
            continue
        loss.backward()
        for n, p in params.items():
            if p.grad is not None:
                acc[n] += (p.grad.detach().double() ** 2).mean()
        seen += 1
    model.zero_grad(set_to_none=True)
    if seen == 0:
        raise ValueError("no batch produced a loss; nothing to estimate curvature from")

    out = {}
    for n, p in params.items():
        out[n] = {
            "fisher": float(acc[n] / seen),
            "var": float(p.detach().float().var()),
            "numel": int(p.numel()),
        }
    return out


def waterfill_stages(
    stats: dict[str, dict],
    target_bpw: float,
    *,
    group_size: int = 8,
    spec_grid: Iterable[Iterable[int]] = DEFAULT_SPEC_GRID,
    iterations: int = 64,
    min_sqnr_db: float | None = DEFAULT_MIN_SQNR_DB,
    rd_slope_db_per_bit: float = RD_SLOPE_DB_PER_BIT,
    rd_intercept_db: float = RD_INTERCEPT_DB,
) -> dict[str, list[int]]:
    """Reverse water-filling with a distortion ceiling.

    Pure water-filling minimises TOTAL distortion, which lets it starve an individual tensor
    to buy accuracy elsewhere. On LFM2.5-2.6B that put 22M-parameter feed_forward tensors at
    1.5 bpw / 7.7 dB SQNR while 1M-parameter attention v_projs got 4.5 bpw / 25 dB - 56.6% of
    parameters ended up below 14 dB and perplexity went 16.3 -> 187.

    The rate rule ``R_t = 0.5*log2(F_t*var_t) + c`` ranks by MEAN curvature per weight and
    carries no tensor-size term, so a big tensor with moderate curvature loses to a small one
    with high curvature no matter how much total distortion that costs. The Lagrangian this
    came from assumes every tensor sits above the water level; tensors pinned to the grid
    floor violate it.

    ``min_sqnr_db`` restores the missing constraint as a per-tensor distortion ceiling
    (D_i <= D_max), which is the standard form of reverse water-filling under a maximum
    distortion bound. The floor is converted to bits through a linear RD model; only the
    budget above the floor is water-filled.
    """
    if target_bpw <= 0:
        raise ValueError(f"target_bpw must be positive, got {target_bpw}")
    grid = [(tuple(int(k) for k in spec), spec_bits_per_weight(spec, group_size))
            for spec in spec_grid]
    if not grid:
        raise ValueError("spec_grid is empty")
    grid.sort(key=lambda kv: kv[1])

    names = list(stats)
    if not names:
        raise ValueError("stats is empty")
    total_w = sum(stats[n]["numel"] for n in names)

    floor_bpw = 0.0
    if min_sqnr_db is not None:
        floor_bpw = (min_sqnr_db - rd_intercept_db) / rd_slope_db_per_bit
        reachable = [g for g in grid if g[1] >= floor_bpw]
        if not reachable:
            raise ValueError(
                f"min_sqnr_db={min_sqnr_db} needs {floor_bpw:.2f} bpw but the richest spec "
                f"in the grid is {grid[-1][1]:.2f} bpw")
        floor_bpw = reachable[0][1]
        if floor_bpw > target_bpw:
            raise ValueError(
                f"target_bpw={target_bpw} is below the {floor_bpw:.2f} bpw floor implied by "
                f"min_sqnr_db={min_sqnr_db}. Raise the target or lower the SQNR floor - a "
                f"budget under the floor cannot be met without starving tensors.")

    score = {
        n: 0.5 * math.log2(max(stats[n]["fisher"] * stats[n]["var"], 1e-45))
        for n in names
    }

    def pick(offset: float, name: str) -> tuple[tuple[int, ...], float]:
        want = score[name] + offset
        best = min(grid, key=lambda kv: abs(kv[1] - want))
        if best[1] < floor_bpw:
            best = next(g for g in grid if g[1] >= floor_bpw)
        return best

    def mean_bpw(offset: float) -> float:
        return sum(pick(offset, n)[1] * stats[n]["numel"] for n in names) / total_w

    lo, hi = -128.0, 128.0
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if mean_bpw(mid) > target_bpw:
            hi = mid
        else:
            lo = mid
    offset = (lo + hi) / 2
    return {n: list(pick(offset, n)[0]) for n in names}


def achieved_bpw(
    stages_map: dict[str, list[int]],
    stats: dict[str, dict],
    *,
    group_size: int = 8,
) -> float:
    """Parameter-weighted mean bits per weight of a stages map - what the map really costs."""
    total_w = sum(stats[n]["numel"] for n in stages_map if n in stats)
    if total_w == 0:
        return 0.0
    return sum(
        spec_bits_per_weight(spec, group_size) * stats[n]["numel"]
        for n, spec in stages_map.items()
        if n in stats
    ) / total_w


def waterfill_with_roles(
    stats: dict[str, dict],
    target_bpw: float,
    role_of,
    *,
    group_size: int = 8,
    spec_grid: Iterable[Iterable[int]] = DEFAULT_SPEC_GRID,
    min_sqnr_db: float | None = DEFAULT_MIN_SQNR_DB,
) -> tuple[dict[str, list[int]], dict[str, str]]:
    """Curvature-driven rates behind autoquant\'s role rails.

    Role priors decide WHETHER a tensor may be quantized (the output head is
    catastrophic under RVQ, norms stay fp16); water-filling decides HOW MANY bits
    the rest get. Returns (stages_map, dense_map).
    """
    from orka.autoquant.priors import ROLE_PRIORS

    quantizable, dense = {}, {}
    for name in stats:
        role = role_of(name)
        prior = ROLE_PRIORS.get(role, ROLE_PRIORS["unknown"])
        if prior.get("allow_rvq"):
            quantizable[name] = stats[name]
        else:
            dense[name] = f"{role}: {prior['method']} prior"
    if not quantizable:
        return {}, dense

    stages = waterfill_stages(quantizable, target_bpw, group_size=group_size,
                              spec_grid=spec_grid, min_sqnr_db=min_sqnr_db)

    # sensitive roles get one rung richer; no-op at the grid ceiling
    grid = sorted((tuple(int(k) for k in s) for s in spec_grid),
                  key=lambda s: spec_bits_per_weight(s, group_size))
    for name, spec in list(stages.items()):
        prior = ROLE_PRIORS.get(role_of(name), {})
        if not prior.get("extra_stage"):
            continue
        cur = spec_bits_per_weight(spec, group_size)
        richer = [s for s in grid if spec_bits_per_weight(s, group_size) > cur]
        if richer:
            stages[name] = list(richer[0])
    return stages, dense
