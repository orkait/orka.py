"""Device resolution, GPU memory caps, OOM wrapping, async I/O."""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Sequence

import psutil
try:
    import torch
except ImportError:
    torch = None

class SystemRAMExceededError(RuntimeError):
    pass


def _get_system_memory_info():
    vm = psutil.virtual_memory()
    return vm.total, vm.available, vm.used


def _calculate_default_ram_cap_gb() -> float | None:
    total, available, used = _get_system_memory_info()
    # User rule: only use 80% of currently available remaining memory
    # Safe limit (total system used) = current_used + 0.8 * available
    safe_limit_bytes = used + (0.8 * available)
    return safe_limit_bytes / (1024**3)


def _monitor_ram_task(cap_gb: float, stop_event: threading.Event, interval: float = 0.5):
    cap_bytes = cap_gb * 1024**3
    while not stop_event.is_set():
        used = psutil.virtual_memory().used
        if used > cap_bytes:
            # We can't easily "interrupt" the main thread with an exception in Python
            # without signals or checking a flag. Orka's pack loop checks progress.
            # We'll set a global flag that the loops can check.
            _set_ram_exceeded(f"System RAM usage ({used/(1024**3):.2f} GB) exceeded cap ({cap_gb:.2f} GB)")
            break
        time.sleep(interval)


_RAM_EXCEEDED_MSG: str | None = None
_RAM_LOCK = threading.Lock()

def _set_ram_exceeded(msg: str):
    global _RAM_EXCEEDED_MSG
    with _RAM_LOCK:
        _RAM_EXCEEDED_MSG = msg

def _check_ram_cap():
    global _RAM_EXCEEDED_MSG
    with _RAM_LOCK:
        if _RAM_EXCEEDED_MSG:
            msg = _RAM_EXCEEDED_MSG
            _RAM_EXCEEDED_MSG = None
            # Aggressive exit if we are way over
            raise SystemRAMExceededError(msg)


_MONITOR_STOP_EVENT = threading.Event()
_MONITOR_THREAD: threading.Thread | None = None

def _apply_system_ram_cap(max_ram_gb: float | None) -> None:
    global _MONITOR_THREAD
    if max_ram_gb is None:
        max_ram_gb = _calculate_default_ram_cap_gb()
    
    if max_ram_gb is None:
        return

    import sys
    print(f"INFO: System RAM cap = {max_ram_gb:.2f} GB", file=sys.stderr)
    
    _MONITOR_STOP_EVENT.clear()
    _MONITOR_THREAD = threading.Thread(
        target=_monitor_ram_task, 
        args=(max_ram_gb, _MONITOR_STOP_EVENT), 
        daemon=True
    )
    _MONITOR_THREAD.start()


def _stop_ram_monitor():
    _MONITOR_STOP_EVENT.set()
    if _MONITOR_THREAD:
        _MONITOR_THREAD.join(timeout=2.0)


def _apply_cpu_cap(max_threads: int | None) -> None:
    if not max_threads or max_threads <= 0:
        return
    if torch:
        torch.set_num_threads(max_threads)
        torch.set_num_interop_threads(max_threads)
    
    # Also set common environment variables for libraries that respect them
    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)
    
    import sys
    print(f"INFO: CPU thread cap = {max_threads}", file=sys.stderr)



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
