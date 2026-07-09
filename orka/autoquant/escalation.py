"""Escalation triggers + global signature cache. A tensor goes to the LLM when any trigger
fires: low policy/role confidence, probe conflict (RD says cheap but SQNR fragile), a
post-pack regression, or unknown role. Verdicts are cached by signature (role + shape
bucket + sensitivity bucket + objective) in a global JSON, so they are reproducible and
reused across models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from orka.autoquant.probes import Signals

CONF_THRESHOLD = 0.6


def _conflict(s: Signals) -> bool:
    # cheap knee but a stage where SQNR is still poor -> conflicting signal
    return s.rd_knee_bits <= 3 and min(s.sqnr_curve.values()) < 10.0


def should_escalate(role_conf: float, signals: Signals, policy_conf: float, regressed: bool) -> bool:
    return (role_conf < CONF_THRESHOLD or policy_conf < CONF_THRESHOLD
            or _conflict(signals) or regressed)


def _bucket(x: float, edges=(0.005, 0.02, 0.05, 0.1)) -> int:
    return sum(1 for e in edges if x > e)


def signature(role: str, shape: tuple[int, ...], s: Signals, objective: str) -> str:
    key = f"{role}|{tuple(shape)}|knee{s.rd_knee_bits}|sens{_bucket(s.sensitivity)}|{objective}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


class Cache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, sig: str):
        return self.data.get(sig)

    def put(self, sig: str, verdict: dict) -> None:
        self.data[sig] = verdict
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


def default_cache_path() -> Path:
    return Path.home() / ".orka" / "autoquant-cache.json"
