"""The autoquant chain: for each tensor, classify role -> probe -> policy decide -> escalate
to the LLM if a trigger fires (unless use_llm=False) -> cached verdict. Returns the per-tensor
config map. The pack -> pulse-check -> refine loop lives in cmd_autoquant (Task 9/10)."""
from __future__ import annotations
import numpy as np
from orka.autoquant.roles import classify_role
from orka.autoquant.probes import probe_tensor
from orka.autoquant.policy import decide
from orka.autoquant.escalation import should_escalate, signature, Cache, default_cache_path
from orka.autoquant.harness import decide_with_llm
from orka.autoquant.schema import TensorConfig


def derive_config(weights: dict[str, np.ndarray], objective: str, *, use_llm: bool = True,
                  llm_fn=None, cache: Cache | None = None) -> dict[str, TensorConfig]:
    if use_llm and cache is None:
        cache = Cache(default_cache_path())
    out: dict[str, TensorConfig] = {}
    for name, W in weights.items():
        role, role_conf = classify_role(name, tuple(np.shape(W)))
        signals = probe_tensor(W)
        cfg, policy_conf = decide(role, signals)
        if use_llm and should_escalate(role_conf, signals, policy_conf, regressed=False):
            sig = signature(role, tuple(np.shape(W)), signals, objective)
            cached = cache.get(sig)
            if cached is not None:
                cfg = TensorConfig(source="cache", confidence=0.75,
                                   rationale="cached", **{k: cached[k] for k in
                                   ("method", "bits", "stages", "normalization", "keep_fp16")})
            elif llm_fn is not None:
                try:
                    cfg = decide_with_llm(role, tuple(np.shape(W)), signals, objective, llm_fn=llm_fn)
                    cache.put(sig, {k: getattr(cfg, k) for k in
                              ("method", "bits", "stages", "normalization", "keep_fp16")})
                except ValueError:
                    pass  # invalid LLM output -> keep policy cfg
        out[name] = cfg
    return out
