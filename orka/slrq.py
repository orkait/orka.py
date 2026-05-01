"""SLRQ (Spherical Logarithmic Residual Quantization) experimental quantizer.

Block-wise salient-protected SLRQ: per block of N values, keep the absolute
max in fp32 and quantize the remaining N-1 with a power-of-two anchor.
"""

from orka._impl import (
    cmd_slrq_eval,
    quantize_block_salient_slrq_vectorized,
)
