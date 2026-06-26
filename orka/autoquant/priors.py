"""Seeded role->rule priors for autoquant. Hard-won defaults from packing experiments:
the output head is catastrophic under RVQ (ppl 1.2M) but lossless as int8; norms/biases
must stay fp16; input embeddings tolerate RVQ. Target per-linear SQNR ~30 dB (14 dB was
catastrophic at model scale)."""
from __future__ import annotations

SQNR_TARGET_DB: float = 30.0

# method: default quant method for the role. allow_rvq: may RVQ ever be used here.
# confidence: how sure the policy is (1.0 = never escalate this role).
ROLE_PRIORS: dict[str, dict] = {
    "out-head":  {"method": "int8", "allow_rvq": False, "confidence": 1.0},
    "in-embed":  {"method": "rvq",  "allow_rvq": True,  "confidence": 0.8},
    "norm":      {"method": "fp16", "allow_rvq": False, "confidence": 1.0},
    "bias":      {"method": "fp16", "allow_rvq": False, "confidence": 1.0},
    "attn.q":    {"method": "rvq",  "allow_rvq": True,  "confidence": 0.7},
    "attn.k":    {"method": "rvq",  "allow_rvq": True,  "confidence": 0.7},
    "attn.v":    {"method": "rvq",  "allow_rvq": True,  "confidence": 0.6, "extra_stage": True},
    "attn.o":    {"method": "rvq",  "allow_rvq": True,  "confidence": 0.7},
    "mlp.up":    {"method": "rvq",  "allow_rvq": True,  "confidence": 0.7},
    "mlp.gate":  {"method": "rvq",  "allow_rvq": True,  "confidence": 0.7},
    "mlp.down":  {"method": "rvq",  "allow_rvq": True,  "confidence": 0.6, "extra_stage": True},
    "unknown":   {"method": "fp16", "allow_rvq": False, "confidence": 0.0},  # safe default
}
