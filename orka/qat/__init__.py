"""Quantization-aware training: QATVQLinear, the training loop, and distillation.

The public `orka.qat` symbols (build_qat_student, QATVQLinear, ...) are re-exported
here so the historical import path keeps working after the move to a package."""
from orka.qat._core import (  # noqa: F401
    QATVQLinear,
    _chunked_assign,
    _kmeans_fit,
    _kmeans_init,
    _pick_block_size,
    build_qat_student,
    collect_codebook_loss,
)
