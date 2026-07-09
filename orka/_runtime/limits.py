"""Host resource governance: RAM caps + monitoring + preflight, and the CPU thread cap.

The RAM monitor is a daemon thread polling RSS; pipeline checkpoints call
``_check_ram_cap`` to raise ``SystemRAMExceededError`` after an overage. All RAM
state (the exceeded flag, the monitor thread) lives in this module.
"""

from __future__ import annotations

import os
import threading
import time

from orka import config

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None
    _HAS_PSUTIL = False

# torch is imported lazily in _apply_cpu_cap (its only use here). Importing it at
# module scope pulled ~1s of torch startup into every `import orka` - including
# `orka --help`, arg-validation errors, and pure-numpy-backend runs that never
# touch torch. Deferring it keeps those paths fast.


# Never let RAM_CAP exceed this regardless of user input. Calibrated for 32GB
# systems: leaves 5GB for kernel + UI + swap-pressure margin. Bound at import;
# raise via ORKA_HARD_CEILING_GB on larger machines (e.g. 64GB).
HARD_CEILING_GB = config.hard_ceiling_gb()

PREFLIGHT_MIN_AVAIL_GB = config.preflight_min_avail_gb()
PREFLIGHT_MAX_SWAP_GB = config.preflight_max_swap_gb()


class SystemRAMExceededError(RuntimeError):
    """Raised at pipeline checkpoints when the RAM monitor flagged overage,
    or when preflight checks refuse to start a job."""


def _get_system_memory_info():
    if not _HAS_PSUTIL:
        return None, None, None
    vm = psutil.virtual_memory()
    return vm.total, vm.available, vm.used


def _calculate_default_ram_cap_gb() -> float | None:
    """Auto-cap = currently_used + 80% of currently_available."""
    total, available, used = _get_system_memory_info()
    if total is None:
        return None
    safe_limit_bytes = used + (0.8 * available)
    return safe_limit_bytes / (1024 ** 3)


def _monitor_ram_task(cap_gb: float, stop_event: threading.Event, interval: float = 0.1):
    """Daemon thread: poll RSS every ``interval`` seconds; flag overage.

    100ms default tightens worst-case overshoot vs the previous 500ms. Pipeline
    checkpoints (``_check_ram_cap``) raise ``SystemRAMExceededError`` on the
    next call after the flag is set.
    """
    if not _HAS_PSUTIL:
        return
    cap_bytes = cap_gb * 1024 ** 3
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        try:
            used = process.memory_info().rss
            if used > cap_bytes:
                # Keep polling: _check_ram_cap clears the flag when it raises,
                # so a long job stays protected after the first overage.
                _set_ram_exceeded(
                    f"Process RAM usage ({used / (1024 ** 3):.2f} GB) exceeded cap ({cap_gb:.2f} GB)"
                )
        except Exception:
            pass
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
            raise SystemRAMExceededError(msg)


_MONITOR_STOP_EVENT = threading.Event()
_MONITOR_THREAD: threading.Thread | None = None


def _preflight_memory_check(workload_budget_gb: float = 5.0) -> None:
    """Refuse to start work if system is already under memory pressure.

    Raises ``SystemRAMExceededError`` if:
      - MemAvailable is below workload_budget + safety margin
      - Swap usage exceeds threshold (system already thrashing-prone)

    These checks STOP the job before any allocation happens. They protect
    against running heavy workloads on a system that will freeze under load.
    """
    if not _HAS_PSUTIL:
        return
    vm = psutil.virtual_memory()
    avail_gb = vm.available / (1024 ** 3)
    sw = psutil.swap_memory()
    swap_used_gb = sw.used / (1024 ** 3)
    required_avail = workload_budget_gb + PREFLIGHT_MIN_AVAIL_GB

    if avail_gb < required_avail:
        raise SystemRAMExceededError(
            f"REFUSING to start: MemAvailable={avail_gb:.1f}GB < "
            f"workload_budget={workload_budget_gb:.1f}GB + safety_margin={PREFLIGHT_MIN_AVAIL_GB:.1f}GB. "
            f"Free memory before retrying (close VM/browser/other processes), "
            f"or override via ORKA_PREFLIGHT_MIN_AVAIL_GB if you accept the risk."
        )
    if swap_used_gb > PREFLIGHT_MAX_SWAP_GB:
        raise SystemRAMExceededError(
            f"REFUSING to start: SwapUsed={swap_used_gb:.1f}GB > "
            f"threshold={PREFLIGHT_MAX_SWAP_GB:.1f}GB. "
            f"System under prior memory pressure - will thrash if pushed further. "
            f"Run 'sudo swapoff -a && sudo swapon -a' to clear swap, "
            f"or override via ORKA_PREFLIGHT_MAX_SWAP_GB."
        )


def _apply_hard_ram_cap(max_ram_gb: float) -> None:
    """OBSOLETE: OS-enforced hard cap on process address space (RLIMIT_AS).

    This has been disabled because capping Virtual Memory (VSZ) crashes
    CUDA/PyTorch which aggressively maps 20GB+ of virtual addresses for context
    regardless of physical RAM usage. The polling monitor handles physical RAM safety.
    """
    return


def _enforce_hard_ceiling(requested_gb: float) -> float:
    """Clamp requested cap to HARD_CEILING_GB. Logs if clamped."""
    import sys
    if requested_gb > HARD_CEILING_GB:
        print(
            f"WARNING: requested RAM cap {requested_gb:.1f}GB exceeds "
            f"HARD_CEILING_GB={HARD_CEILING_GB:.1f}GB. Clamping. "
            f"Override only via ORKA_HARD_CEILING_GB env var (you accept system-freeze risk).",
            file=sys.stderr,
        )
        return HARD_CEILING_GB
    return requested_gb


def _apply_system_ram_cap(
    max_ram_gb: float | None, workload_budget_gb: float | None
) -> None:
    """Apply layered RAM protection. Both args required when cap is requested.

      1. Preflight check  - refuse if MemAvailable < budget+5GB or SwapUsed > 4GB
      2. Hard ceiling     - clamp max_ram_gb to HARD_CEILING_GB (25GB)
      3. RLIMIT_AS        - DISABLED (capping VSZ crashes CUDA/PyTorch; see _apply_hard_ram_cap)
      4. Polling monitor  - 100ms RSS checkpoint; the only active runtime enforcement

    Pass max_ram_gb=None to skip cap entirely (NOT recommended).
    When max_ram_gb is set, workload_budget_gb is required (no default - caller
    MUST think about expected process budget).

    Per-workload budget table (process delta, validated):
      tiny script         : 0.5 GB
      SmolLM2-135M torch  : 5 GB    (numpy: 2.5)
      Pythia-160M         : 5 GB    (numpy: 2.5)
      Bloom-560m          : 7 GB    (numpy: 4)
      Qwen3-0.6B torch    : 9 GB    (numpy: 5)
      SmolLM2-360M        : 7 GB    (numpy: 3.5)
    Add +2 if EM-AQ enabled, +1 if family_stages_map, +1 if prefetch>2.
    """
    global _MONITOR_THREAD
    import sys

    if max_ram_gb is None:
        return

    if not _HAS_PSUTIL:
        raise RuntimeError(
            "psutil required for RAM cap enforcement. install via `uv pip install psutil`."
        )

    if workload_budget_gb is None:
        raise ValueError(
            "workload_budget_gb is required when max_ram_gb is set. "
            "Pass --workload-budget-gb on CLI. See orka._runtime per-workload table."
        )

    _preflight_memory_check(workload_budget_gb=workload_budget_gb)
    max_ram_gb = _enforce_hard_ceiling(max_ram_gb)
    _apply_hard_ram_cap(max_ram_gb)

    print(
        f"INFO: System RAM cap = {max_ram_gb:.2f} GB (poll 100ms, hard ceiling {HARD_CEILING_GB:.1f}GB, "
        f"workload_budget={workload_budget_gb:.1f}GB)",
        file=sys.stderr,
    )
    _MONITOR_STOP_EVENT.clear()
    _MONITOR_THREAD = threading.Thread(
        target=_monitor_ram_task,
        args=(max_ram_gb, _MONITOR_STOP_EVENT),
        daemon=True,
    )
    _MONITOR_THREAD.start()


def _stop_ram_monitor() -> None:
    _MONITOR_STOP_EVENT.set()
    if _MONITOR_THREAD:
        _MONITOR_THREAD.join(timeout=2.0)


def _apply_cpu_cap(max_threads: int | None) -> None:
    """Cap CPU concurrency three ways: torch threads, BLAS env vars, OS affinity.

    OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS / VECLIB_MAXIMUM_
    THREADS / NUMEXPR_NUM_THREADS are read by their respective libraries at
    LOAD time. Setting them here only affects libraries that re-read the env
    or respawn workers (e.g. via subprocess). For full BLAS cap, callers
    should also set these vars BEFORE python startup OR rely on
    ``orka/__main__.py`` which does that pre-import.

    ``os.sched_setaffinity`` is the OS-level cap: kernel scheduler will only
    run this process on the named cores. Hard cap regardless of library
    behaviour.
    """
    if not max_threads or max_threads <= 0:
        return

    import sys

    try:
        import torch
    except Exception:  # ImportError, or OSError from a broken/partial native install
        torch = None
    if torch is not None:
        torch.set_num_threads(max_threads)
        try:
            torch.set_num_interop_threads(max_threads)
        except RuntimeError:
            # Already set elsewhere; setting again raises. Non-fatal.
            pass

    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(max_threads)

    affinity_set = False
    try:
        if hasattr(os, "sched_setaffinity"):
            available = sorted(os.sched_getaffinity(0))
            chosen = set(available[:max_threads]) if available else set(range(max_threads))
            os.sched_setaffinity(0, chosen)
            affinity_set = True
    except (AttributeError, OSError) as exc:
        print(f"WARNING: could not set CPU affinity: {exc}", file=sys.stderr)

    print(
        f"INFO: CPU thread cap = {max_threads}"
        f"{' (affinity pinned)' if affinity_set else ''}",
        file=sys.stderr,
    )
