"""Feature gates for legacy or experimental code paths."""

from __future__ import annotations

from orka import config

AWQ_DISABLED_MESSAGE = (
    "AWQ support is disabled by default because it depends on external "
    "calibration data. Set ORKA_ENABLE_AWQ=1 to use the legacy AWQ path."
)


def awq_feature_enabled() -> bool:
    return config.awq_enabled()


def ensure_awq_feature_enabled() -> None:
    if not awq_feature_enabled():
        raise RuntimeError(AWQ_DISABLED_MESSAGE)
