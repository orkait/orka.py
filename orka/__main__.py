"""Entry for ``python -m orka``.

Pre-import resource caps run BEFORE numpy/torch import so OpenMP / BLAS
thread pools obey ``--max-cpu-threads``. After early caps,
``orka.cli.main`` is imported and dispatched.
"""

import os
import sys


def _early_cpu_cap() -> None:
    """Set OMP/MKL/OPENBLAS env vars from --max-cpu-threads BEFORE library load.

    BLAS thread pools fix at library load. Setting the env vars later is a no-op.
    Affinity + ``torch.set_num_threads`` still apply at runtime via _apply_cpu_cap.
    """
    argv = sys.argv
    cpu_cap = None
    for i, arg in enumerate(argv):
        if arg.startswith("--max-cpu-threads="):
            try:
                cpu_cap = int(arg.split("=", 1)[1])
            except ValueError:
                pass
            break
        if arg == "--max-cpu-threads" and i + 1 < len(argv):
            try:
                cpu_cap = int(argv[i + 1])
            except ValueError:
                pass
            break
    if not cpu_cap or cpu_cap <= 0:
        return
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, str(cpu_cap))


_early_cpu_cap()

from orka.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
