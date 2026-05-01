"""Orka compiler package.

Thematic modules:
    core         - constants, dataclasses, BG writer, GPU helpers, OOM, primitives
    io_format    - tensor checkpoint loading + on-disk I/O for indices/codebooks/scales
    quant_spec   - vq-/rvq- spec parsing, family classification
    transforms   - normalization, rotation, outlier extraction
    metrics      - reconstruction quality metrics
    kmeans       - codebook learning, assignment, vector helpers
    activations  - AWQ activation calibration via Hugging Face
    pack         - pack_checkpoint + inspect_checkpoint
    decode       - decode/verify/reconstruct/report
    sweep        - pack/report matrix sweeps
    eval         - HF prompt-loss / perplexity evaluation
    kaggle       - kaggle-pack pipeline (download + pack + upload)
    slrq         - SLRQ experimental quantizer
    cli          - argparse + command dispatch + main entry
"""

from orka.cli import build_parser
from orka.core import (
    BackgroundWriter,
    CappedOutOfMemoryError,
    ORKA_VERSION,
    PayloadEstimate,
    _parse_params,
    estimate_payload,
)
from orka.decode import reconstruct_artifact, report_artifact, verify_artifact
from orka.eval import _summarize_eval_rows, eval_artifact, eval_sweep
from orka.kmeans import learn_codebook_auto, quantize_vectors_auto
from orka.metrics import quality_metrics_from_flat
from orka.pack import inspect_checkpoint, pack_checkpoint
from orka.quant_spec import (
    classify_tensor_family,
    is_rvq_mixed_spec,
    parse_quant_spec,
    quant_spec_from_sizes,
    rvq_mixed_family_stages,
)
from orka.slrq import quantize_block_salient_slrq_vectorized
from orka.sweep import sweep_checkpoint

__all__ = [
    "BackgroundWriter",
    "CappedOutOfMemoryError",
    "ORKA_VERSION",
    "PayloadEstimate",
    "build_parser",
    "classify_tensor_family",
    "estimate_payload",
    "eval_artifact",
    "eval_sweep",
    "inspect_checkpoint",
    "is_rvq_mixed_spec",
    "learn_codebook_auto",
    "pack_checkpoint",
    "parse_quant_spec",
    "quality_metrics_from_flat",
    "quant_spec_from_sizes",
    "quantize_block_salient_slrq_vectorized",
    "quantize_vectors_auto",
    "reconstruct_artifact",
    "report_artifact",
    "rvq_mixed_family_stages",
    "sweep_checkpoint",
    "verify_artifact",
]
