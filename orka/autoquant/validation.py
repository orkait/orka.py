"""Hard safety rails for autonomous quantization decisions."""
from __future__ import annotations

from orka.autoquant.schema import TensorConfig


def validate_config(role: str, cfg: TensorConfig) -> None:
    if cfg.method not in {"rvq", "int8", "fp16"}:
        raise ValueError(f"unsupported method: {cfg.method}")
    if role == "out-head" and cfg.method == "rvq":
        raise ValueError("out-head must not use rvq")
    if role in {"norm", "bias"} and (cfg.method != "fp16" or not cfg.keep_fp16):
        raise ValueError(f"{role} must remain fp16")
    if cfg.method == "rvq" and (cfg.bits not in {2, 3, 4, 6, 8} or cfg.stages < 1):
        raise ValueError("unsupported rvq configuration")
    if cfg.method == "fp16" and not cfg.keep_fp16:
        raise ValueError("fp16 method requires keep_fp16")


def validate_configs(cfgs: dict[str, TensorConfig], roles: dict[str, str]) -> None:
    for name, cfg in cfgs.items():
        validate_config(roles.get(name, "unknown"), cfg)
