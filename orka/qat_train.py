"""Backwards-compat shim - implementation moved to orka.qat.train."""
from orka.qat.train import *  # noqa: F401,F403
from orka.qat.train import main  # noqa: F401
if __name__ == "__main__":
    raise SystemExit(main())
