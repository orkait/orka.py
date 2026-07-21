"""Generic utilities. No orka-specific semantics. Stdlib only."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path


def _derive_seed(parts: Sequence[object]) -> int:
    import hashlib

    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)
def _report_progress(path: Path | None, message: str):
    if path:
        try:
            with path.open("w") as f:
                f.write(message + "\n")
        except Exception:
            pass
    print(message)


_WARNED_ONCE: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """RuntimeWarning on the first sighting of ``key``, silent after.

    Accelerator fallbacks fire per layer and per token, so warning unconditionally
    floods the log and warning never hides multi-x slowdowns.
    """
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    import warnings

    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _source_signature(source: Path) -> str:
    try:
        st = Path(source).resolve().stat()
        return f"{st.st_size}-{st.st_mtime_ns}"
    except OSError:
        return str(source)
def _index_bits_for_size(codebook_size: int) -> int:
    return math.ceil(math.log2(codebook_size)) if codebook_size > 1 else 1


def _safe_tensor_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return safe.strip("._") or "tensor"
def _product(values: Sequence[int]) -> int:
    total = 1
    for value in values:
        total *= value
    return total


def _reshape_flat(values: Sequence[float], shape: Sequence[int]) -> object:
    if not shape:
        if len(values) != 1:
            raise ValueError("scalar reshape requires exactly one value")
        return float(values[0])
    if len(shape) == 1:
        width = int(shape[0])
        return [float(v) for v in values[:width]]

    step = _product(shape[1:])
    return [
        _reshape_flat(values[i : i + step], shape[1:])
        for i in range(0, int(shape[0]) * step, step)
    ]

def _dir_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total

def _require_non_empty(name: str, values: Sequence[object]) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
def _safe_exp(value: float) -> float:
    if value > 700:
        return float("inf")
    return math.exp(value)
def _parse_params(value: str) -> int:
    text = value.strip().lower().replace("_", "")
    suffixes = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    suffix = text[-1]
    if suffix in suffixes:
        return int(Decimal(text[:-1]) * suffixes[suffix])
    return int(text)


def _human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1000 or unit == units[-1]:
            return f"{value:.3f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1000
    return f"{value:.3f} TB"

def _best_run(runs: Sequence[dict], key: str, reverse: bool) -> dict | None:
    if not runs:
        return None
    return dict(sorted(runs, key=lambda run: float(run[key]), reverse=reverse)[0])

