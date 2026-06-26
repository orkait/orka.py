"""Deterministic policy core: (role, signals) -> (TensorConfig, confidence). Table-driven
from priors. RVQ roles take their bit count from the rate-distortion knee; sensitive roles
(attn.v, mlp.down) get one extra stage; head/norm/bias follow their fixed priors."""
from __future__ import annotations
from orka.autoquant.priors import ROLE_PRIORS
from orka.autoquant.probes import Signals
from orka.autoquant.schema import TensorConfig


def decide(role: str, signals: Signals) -> tuple[TensorConfig, float]:
    p = ROLE_PRIORS.get(role, ROLE_PRIORS["unknown"])
    conf = float(p["confidence"])
    if p["method"] == "fp16":
        return TensorConfig("fp16", 16, 0, "none", True, "policy", conf,
                            f"{role}: fp16 prior"), conf
    if p["method"] == "int8":
        return TensorConfig("int8", 8, 0, "block-max", False, "policy", conf,
                            f"{role}: int8 prior (RVQ-fragile)"), conf
    bits = int(signals.rd_knee_bits)
    stages = 2 + (1 if p.get("extra_stage") else 0)
    return TensorConfig("rvq", bits, stages, "block-max", False, "policy", conf,
                        f"{role}: rvq {bits}b/{stages}st at SQNR knee"), conf
