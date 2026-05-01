"""Core types, primitives, GPU resolution, OOM wrapping.

No internal dependencies. Imported by every other orka module.
"""

from __future__ import annotations

import math
import re
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import queue
import threading
from typing import Sequence


ORKA_VERSION = 1
@dataclass(frozen=True)
class PayloadEstimate:
    params: int
    group_size: int
    codebook_size: int
    index_bits: int
    vector_count: int
    index_bytes: int
    scale_block_vectors: int
    scale_bytes: int
    bits_per_weight: float
    total_payload_bytes: int


class BackgroundWriter:
    def __init__(self):
        self.queue = queue.Queue(maxsize=128)
        self.errors: list[tuple[str, str]] = []
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            task = self.queue.get()
            if task is None:
                self.queue.task_done()
                break
            fn, args = task
            try:
                fn(*args)
            except Exception as e:
                self.errors.append((fn.__name__, repr(e)))
            finally:
                self.queue.task_done()

    def submit(self, fn, *args):
        self.queue.put((fn, args))

    def wait(self):
        self.queue.join()
        if self.errors:
            detail = "; ".join(f"{name}: {err}" for name, err in self.errors)
            raise RuntimeError(
                f"background writes failed ({len(self.errors)} error(s)): {detail}"
            )

    def stop(self):
        self.queue.put(None)
        if self.thread.is_alive():
            self.thread.join()

_BG_WRITER = BackgroundWriter()

def estimate_payload(
    params: int,
    group_size: int,
    codebook_size: int,
    scale_block_vectors: int = 64,
    scale_bits: int = 0,
) -> PayloadEstimate:
    if params <= 0:
        raise ValueError("params must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if codebook_size <= 1:
        raise ValueError("codebook_size must be greater than 1")
    if scale_block_vectors <= 0:
        raise ValueError("scale_block_vectors must be positive")
    if scale_bits < 0:
        raise ValueError("scale_bits must be non-negative")

    index_bits = math.ceil(math.log2(codebook_size))
    vector_count = math.ceil(params / group_size)
    index_bytes = math.ceil(vector_count * index_bits / 8)
    scale_count = math.ceil(vector_count / scale_block_vectors)
    scale_bytes = math.ceil(scale_count * scale_bits / 8)
    return PayloadEstimate(
        params=params,
        group_size=group_size,
        codebook_size=codebook_size,
        index_bits=index_bits,
        vector_count=vector_count,
        index_bytes=index_bytes,
        scale_block_vectors=scale_block_vectors,
        scale_bytes=scale_bytes,
        bits_per_weight=(index_bytes + scale_bytes) * 8 / params,
        total_payload_bytes=index_bytes + scale_bytes,
    )

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


def _source_signature(source: Path) -> str:
    try:
        st = Path(source).resolve().stat()
        return f"{st.st_size}-{st.st_mtime_ns}"
    except OSError:
        return str(source)

def _is_numpy_array(value: object) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "reshape")
        and hasattr(value, "astype")
        and hasattr(value, "tolist")
    )


def _is_torch_tensor(value: object) -> bool:
    return hasattr(value, "detach") and hasattr(value, "to")


def _supports_numpy_backend(value: object) -> bool:
    return _is_numpy_array(value) or hasattr(value, "detach")


def _resolve_torch_device(device: str):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch backend requires torch") from exc

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "torch backend requested CUDA, but torch.cuda.is_available() is false"
        )
    return resolved


def _cuda_sm_major(device_index: int = 0) -> int | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability(device_index)[0]
    except Exception:
        return None


def _maybe_fallback_cuda_to_cpu(device: str, backend: str) -> str:
    """Fall back to CPU if the CUDA device is sm_60 (P100 etc.) — PyTorch requires sm_70+."""
    if backend != "torch":
        return device
    dev_lower = device.lower()
    if "cuda" not in dev_lower and dev_lower != "auto":
        return device
    sm = _cuda_sm_major()
    if sm is not None and sm < 7:
        import sys
        print(
            f"WARNING: CUDA device is SM {sm}.x (< 7.0) — not supported by this PyTorch build. "
            f"Falling back to CPU automatically.",
            file=sys.stderr,
            flush=True,
        )
        return "cpu"
    return device

def _torch_f32(tensor, device):
    import torch

    resolved = _resolve_torch_device(device)
    if _is_torch_tensor(tensor):
        return resolved, tensor.detach().to(device=resolved, dtype=torch.float32)
    return resolved, torch.as_tensor(tensor, dtype=torch.float32, device=resolved)

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

class CappedOutOfMemoryError(RuntimeError):
    pass


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch
    except Exception:
        return False
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or ("cuda error" in msg and "memory" in msg)


def _wrap_capped_oom(cap_gb: float | None, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        if cap_gb and cap_gb > 0 and _is_cuda_oom(exc):
            raise CappedOutOfMemoryError(
                f"GPU memory cap exceeded ({cap_gb} GB): {exc}"
            ) from exc
        raise


CUBLAS_WORKSPACE_SLOP_BYTES = 256 * 1024 * 1024

def _apply_gpu_memory_cap(
    backend: str, device: str, max_gpu_mem_gb: float | None
) -> None:
    if not max_gpu_mem_gb or max_gpu_mem_gb <= 0:
        return
    if backend != "torch":
        return
    try:
        import torch
    except Exception:
        return
    resolved = _resolve_torch_device(device)
    if resolved.type != "cuda":
        return
    device_index = (
        resolved.index if resolved.index is not None else torch.cuda.current_device()
    )
    total_bytes = torch.cuda.get_device_properties(device_index).total_memory
    cap_bytes = int(max_gpu_mem_gb * 1024 * 1024 * 1024)
    fraction = max(0.05, min(1.0, cap_bytes / total_bytes))
    torch.cuda.set_per_process_memory_fraction(fraction, device_index)
    effective_gb = (cap_bytes - CUBLAS_WORKSPACE_SLOP_BYTES) / (1024**3)
    print(
        f"INFO: GPU memory cap = {max_gpu_mem_gb:.2f} GB on cuda:{device_index} "
        f"(fraction={fraction:.4f} of {total_bytes / (1024**3):.2f} GB). "
        f"Cap covers torch caching allocator only; cuBLAS/cuDNN workspace (~{CUBLAS_WORKSPACE_SLOP_BYTES // (1024 * 1024)} MB) "
        f"lives outside it, so plan for ≈{effective_gb:.2f} GB of usable headroom.",
        file=os.sys.stderr,
    )
def _read_prompt_file(path: Path, max_prompts: int | None = None) -> list[str]:
    if max_prompts is not None and max_prompts <= 0:
        raise ValueError("max_prompts must be positive")
    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise ValueError("prompt file must contain at least one non-empty prompt")
    return prompts

