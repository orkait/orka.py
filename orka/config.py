"""Environment-driven runtime knobs, resolved in one place.

Accessors read the environment per call. Callers that need import-time constants
(``orka._runtime.limits``) bind the result once at import, preserving their
existing behaviour.
"""
from __future__ import annotations

import os

DEFAULT_PREFLIGHT_MIN_AVAIL_GB = 5.0
DEFAULT_PREFLIGHT_MAX_SWAP_GB = 4.0
DEFAULT_HARD_CEILING_GB = 25.0

#: Values of ORKA_ENABLE_AWQ that turn the legacy AWQ path on. Anything else, "0"
#: and "false" included, leaves it off.
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def preflight_min_avail_gb() -> float:
    return _float("ORKA_PREFLIGHT_MIN_AVAIL_GB", DEFAULT_PREFLIGHT_MIN_AVAIL_GB)


def preflight_max_swap_gb() -> float:
    return _float("ORKA_PREFLIGHT_MAX_SWAP_GB", DEFAULT_PREFLIGHT_MAX_SWAP_GB)


def hard_ceiling_gb() -> float:
    return _float("ORKA_HARD_CEILING_GB", DEFAULT_HARD_CEILING_GB)


def kmeans_iters(default: int) -> int:
    raw = os.environ.get("ORKA_KMEANS_ITERS")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def awq_enabled() -> bool:
    return os.environ.get("ORKA_ENABLE_AWQ", "").strip().lower() in _TRUTHY


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


def cuda_visible_devices() -> str | None:
    return os.environ.get("CUDA_VISIBLE_DEVICES")
