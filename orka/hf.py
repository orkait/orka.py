"""Backwards-compat shim. The implementation moved to orka.integrations.hf.
Kept so the documented public path `from orka.hf import load_orka_model` works."""
from orka.integrations.hf import *  # noqa: F401,F403
from orka.integrations.hf import load_orka_model, load_orka_tokenizer  # noqa: F401
