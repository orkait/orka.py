"""Shared literal constants. No logic, no imports."""

from __future__ import annotations

#: Substrings that disqualify a tensor from quantization regardless of shape.
NON_CANDIDATE_MARKERS = (".bias", ".norm", ".layernorm", "rotary_emb", "attention.bias")
