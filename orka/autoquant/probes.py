"""Cheap per-tensor signals for autoquant, computed from one weight tensor (no full pack).
sqnr_at(bits): SQNR of a fast scalar-quant probe at `bits`. rd_knee_bits: smallest bits
hitting the SQNR target. These rank tensors and drive the policy's bit choice."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sqnr_db(W: np.ndarray, bits: int) -> float:
    # symmetric per-row scalar quant probe (cheap proxy for achievable distortion)
    qmax = (1 << (bits - 1)) - 1
    s = np.abs(W).max(axis=1, keepdims=True) / max(qmax, 1)
    s[s == 0] = 1.0
    Wq = np.clip(np.round(W / s), -qmax, qmax) * s
    sig = float((W ** 2).sum())
    err = float(((W - Wq) ** 2).sum()) or 1e-12
    return 10.0 * np.log10(sig / err)


@dataclass
class Signals:
    sqnr_curve: dict[int, float]
    rd_knee_bits: int
    sensitivity: float            # placeholder = weight energy proxy (refined later)

    def sqnr_at(self, bits: int) -> float:
        return self.sqnr_curve.get(bits, _nearest(self.sqnr_curve, bits))


def _nearest(curve: dict[int, float], bits: int) -> float:
    k = min(curve, key=lambda b: abs(b - bits))
    return curve[k]


def probe_tensor(W: np.ndarray, sqnr_target_db: float = 30.0) -> Signals:
    W = np.asarray(W, dtype=np.float32)
    if W.ndim == 1:
        W = W.reshape(1, -1)
    curve = {b: _sqnr_db(W, b) for b in (2, 3, 4, 6, 8)}
    knee = next((b for b in (2, 3, 4, 6, 8) if curve[b] >= sqnr_target_db), 16)
    return Signals(sqnr_curve=curve, rd_knee_bits=knee, sensitivity=float(np.abs(W).mean()))
