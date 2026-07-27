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

#: Byte budget for producer read-ahead (streamed per-tensor packs), in GB. The
#: prefetch queue caps candidate count; this caps the bytes those candidates
#: retain (source_flat + vectors, ~8x numel). <= 0 disables the budget.
DEFAULT_PREFETCH_BUDGET_GB = 4.0

#: H2D transfer budget per chunk for the tiled (giant-tensor) assign, in MB.
#: 65536-row chunks were 2MB at group_size 8 - PCIe-latency-bound (~1940 copies
#: + syncs on the 1B vocab head). 128MB keeps the loop bandwidth-bound.
DEFAULT_ASSIGN_CHUNK_MB = 128

#: zlib level for index/sidecar streams. Decode is level-agnostic (zlib.decompress),
#: so lowering it trades a few percent of artifact size for ~4x compression speed
#: (measured: level 1 = 80 MB/s vs level 6 = 19 MB/s on compressible index streams).
DEFAULT_ZLIB_LEVEL = 6

#: Largest codebook that still gets the scalable k-means++ (k-means||) seeding; above it
#: the init falls back to a uniform random sample of rows.
#:
#: The default deliberately excludes the flagship rvq-12-12 recipe (K=4096 per stage):
#: k-means||'s final reduction is a k-step sequential loop, so at K=4096 it costs ~1.6s of
#: init per codebook while the whole random-seeded fit takes ~0.02s, and it buys only
#: 0.1-2.7% lower MSE (measured on SmolLM2-135M down_proj / q_proj / gate_proj, 8 Lloyd
#: iterations, 3 seeds). That is ~80x the fit time for a sub-3% quality move, so raise
#: this only when init time is free relative to the rest of the pack.
DEFAULT_KMEANS_PP_MAX_K = 2048

#: Rows per chunk when scanning an embedding matrix for semantic hubs. Bounds the
#: [chunk, vocab] similarity tile; the scan itself stays sequential, so this only
#: trades memory for matmul batching.
DEFAULT_SEMANTIC_HUB_CHUNK = 512

#: Token count at or above which the N>1 prefill path decodes the weight to dense and
#: hands it to cuBLAS instead of running the Triton gather-GEMM. Below it the one-time
#: decode is not amortized (measured ~2x in favour of dense at 256+ tokens).
DEFAULT_PREFILL_MIN_TOKENS = 256

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


def prefetch_budget_gb() -> float:
    return _float("ORKA_PREFETCH_BUDGET_GB", DEFAULT_PREFETCH_BUDGET_GB)


def assign_chunk_mb() -> int:
    raw = os.environ.get("ORKA_ASSIGN_CHUNK_MB")
    if raw is None:
        return DEFAULT_ASSIGN_CHUNK_MB
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_ASSIGN_CHUNK_MB


def _int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def kmeans_pp_max_k() -> int:
    return _int("ORKA_KMEANS_PP_MAX_K", DEFAULT_KMEANS_PP_MAX_K, minimum=1)


def semantic_hub_chunk() -> int:
    return _int("ORKA_SEMANTIC_HUB_CHUNK", DEFAULT_SEMANTIC_HUB_CHUNK, minimum=1)


def prefill_min_tokens() -> int:
    return _int("ORKA_PREFILL_MIN_TOKENS", DEFAULT_PREFILL_MIN_TOKENS, minimum=1)


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
