"""Closed-loop refine: pack the current config, measure quality (pulse-check KL/top1),
attribute any regression to the worst tensors, escalate those (bump bits / add a stage),
repeat until the objective is met or the loop stops improving. pack_fn and pulse_fn are
injected so the loop logic is testable without a real pack."""
from __future__ import annotations

from dataclasses import dataclass

from orka.autoquant.probes import Signals
from orka.autoquant.schema import TensorConfig

_BITS_LADDER = [2, 3, 4, 6, 8]


@dataclass
class Quality:
    kl: float        # mean KL divergence vs fp16 (lower better)
    top1: float      # top-1 agreement (higher better)


def meets(q: Quality, objective: str, target: float | None) -> bool:
    if target is None:
        return False
    if objective == "min-bits":
        return q.kl <= target
    if objective == "max-quality":
        return True  # budget enforced elsewhere; quality just maximized within it
    if objective == "knee":
        return q.kl <= target
    return False


def attribute(cfg: dict[str, TensorConfig], signals: dict[str, Signals], k: int = 3) -> list[str]:
    """Rank the RVQ tensors most likely driving regression: lowest SQNR at their assigned bits."""
    scored = [
        (name, signals[name].sqnr_at(c.bits))
        for name, c in cfg.items()
        if c.method == "rvq" and name in signals
    ]
    scored.sort(key=lambda x: x[1])           # lowest SQNR first
    return [name for name, _ in scored[:k]]


def escalate_cfg(c: TensorConfig) -> TensorConfig:
    """Spend more bits on a tensor: step bits up the ladder, then add a stage at the ceiling."""
    if c.method != "rvq":
        return c
    higher = [b for b in _BITS_LADDER if b > c.bits]
    if higher:
        return TensorConfig(c.method, higher[0], c.stages, c.normalization, c.keep_fp16,
                            "refine", c.confidence, c.rationale + f"; bumped bits->{higher[0]}")
    return TensorConfig(c.method, c.bits, c.stages + 1, c.normalization, c.keep_fp16,
                        "refine", c.confidence, c.rationale + f"; +stage->{c.stages + 1}")


def refine(cfg: dict[str, TensorConfig], signals: dict[str, Signals], objective: str,
           target: float | None, *, pack_fn, pulse_fn, max_rounds: int = 5):
    """Returns (final_cfg, final_quality, rounds_used)."""
    cfg = dict(cfg)
    quality = pulse_fn(pack_fn(cfg))
    for r in range(max_rounds):
        if meets(quality, objective, target):
            return cfg, quality, r
        offenders = attribute(cfg, signals)
        if not offenders:
            break
        for name in offenders:
            cfg[name] = escalate_cfg(cfg[name])
        new_q = pulse_fn(pack_fn(cfg))
        if new_q.kl >= quality.kl:           # no improvement -> stop (stuck)
            return cfg, quality, r + 1
        quality = new_q
    return cfg, quality, max_rounds
