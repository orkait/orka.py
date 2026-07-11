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

DEFAULT_LLM_LITE_MODEL = "claude-sonnet-4-6"
DEFAULT_LLM_STRONG_MODEL = "claude-opus-4-8"

#: H2D transfer budget per chunk for the tiled (giant-tensor) assign, in MB.
#: 65536-row chunks were 2MB at group_size 8 - PCIe-latency-bound (~1940 copies
#: + syncs on the 1B vocab head). 128MB keeps the loop bandwidth-bound.
DEFAULT_ASSIGN_CHUNK_MB = 128

#: zlib level for index/sidecar streams. Decode is level-agnostic (zlib.decompress),
#: so lowering it trades a few percent of artifact size for ~4x compression speed
#: (measured: level 1 = 80 MB/s vs level 6 = 19 MB/s on compressible index streams).
DEFAULT_ZLIB_LEVEL = 6

#: Values of ORKA_ENABLE_AWQ that turn the legacy AWQ path on. Anything else, "0"
#: and "false" included, leaves it off.
_TRUTHY_AWQ = frozenset({"1", "true", "yes", "on"})

#: ORKA_KMEANS_FAISS accepts a narrower set than ORKA_ENABLE_AWQ: "on" is not
#: recognised. The two are kept distinct because widening this one would silently
#: enable faiss for anyone who had set it to "on" expecting it to be ignored.
_TRUTHY_FAISS = frozenset({"1", "true", "yes"})


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


def assign_chunk_mb() -> int:
    raw = os.environ.get("ORKA_ASSIGN_CHUNK_MB")
    if raw is None:
        return DEFAULT_ASSIGN_CHUNK_MB
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_ASSIGN_CHUNK_MB


def zlib_level() -> int:
    raw = os.environ.get("ORKA_ZLIB_LEVEL")
    if not raw:
        return DEFAULT_ZLIB_LEVEL
    try:
        return min(9, max(0, int(raw)))
    except ValueError:
        return DEFAULT_ZLIB_LEVEL


def awq_enabled() -> bool:
    return os.environ.get("ORKA_ENABLE_AWQ", "").strip().lower() in _TRUTHY_AWQ


def kmeans_faiss_enabled() -> bool:
    return os.environ.get("ORKA_KMEANS_FAISS", "").strip().lower() in _TRUTHY_FAISS


def llm_lite_model() -> str:
    return os.environ.get("ORKA_LLM_LITE", DEFAULT_LLM_LITE_MODEL)


def llm_strong_model() -> str:
    return os.environ.get("ORKA_LLM_STRONG", DEFAULT_LLM_STRONG_MODEL)


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


def cuda_visible_devices() -> str | None:
    return os.environ.get("CUDA_VISIBLE_DEVICES")
