"""Runtime environment control, split by concern:

  limits.py   host resource governance - RAM caps + monitor + preflight, CPU thread cap
  device.py   torch device resolution, CUDA fallback, OOM wrapping, GPU memory cap
  io.py       async disk I/O (BackgroundWriter)

This package re-exports the public names so ``from orka._runtime import X`` keeps
working unchanged; import from the submodules directly only when adding new code.
"""

from orka._runtime.limits import (
    HARD_CEILING_GB,
    PREFLIGHT_MAX_SWAP_GB,
    PREFLIGHT_MIN_AVAIL_GB,
    SystemRAMExceededError,
    _apply_cpu_cap,
    _apply_hard_ram_cap,
    _apply_system_ram_cap,
    _calculate_default_ram_cap_gb,
    _check_ram_cap,
    _enforce_hard_ceiling,
    _get_system_memory_info,
    _monitor_ram_task,
    _preflight_memory_check,
    _set_ram_exceeded,
    _stop_ram_monitor,
)
from orka._runtime.device import (
    CUBLAS_WORKSPACE_SLOP_BYTES,
    CappedOutOfMemoryError,
    _apply_gpu_memory_cap,
    _cuda_sm_major,
    _is_cuda_oom,
    _maybe_fallback_cuda_to_cpu,
    _resolve_auto_backend,
    _resolve_torch_device,
    _wrap_capped_oom,
)
from orka._runtime.io import BackgroundWriter, _BG_WRITER

__all__ = [
    "HARD_CEILING_GB",
    "PREFLIGHT_MAX_SWAP_GB",
    "PREFLIGHT_MIN_AVAIL_GB",
    "SystemRAMExceededError",
    "_apply_cpu_cap",
    "_apply_hard_ram_cap",
    "_apply_system_ram_cap",
    "_calculate_default_ram_cap_gb",
    "_check_ram_cap",
    "_enforce_hard_ceiling",
    "_get_system_memory_info",
    "_monitor_ram_task",
    "_preflight_memory_check",
    "_set_ram_exceeded",
    "_stop_ram_monitor",
    "CUBLAS_WORKSPACE_SLOP_BYTES",
    "CappedOutOfMemoryError",
    "_apply_gpu_memory_cap",
    "_cuda_sm_major",
    "_is_cuda_oom",
    "_maybe_fallback_cuda_to_cpu",
    "_resolve_auto_backend",
    "_resolve_torch_device",
    "_wrap_capped_oom",
    "BackgroundWriter",
    "_BG_WRITER",
]
