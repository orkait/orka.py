"""Orka compiler package.

This package is split into thematic modules for navigability. The original
single-file implementation lives in ``orka._impl`` and is re-exported here.
Future work migrates symbols out of ``_impl`` into their thematic modules.

Thematic modules:
    core         - constants, dataclasses, BG writer, GPU helpers, OOM
    io_format    - tensor loading + on-disk I/O for indices/codebooks/scales
    transforms   - normalization, rotation, outlier extraction
    kmeans       - codebook learning, assignment, vector helpers
    quant_spec   - vq-/rvq- spec parsing, family classification
    metrics      - reconstruction quality metrics
    activations  - AWQ activation calibration via Hugging Face
    pack         - pack_checkpoint + inspect_checkpoint
    decode       - decode/verify/reconstruct/report
    sweep        - pack/report matrix sweeps
    eval         - HF prompt-loss / perplexity evaluation
    kaggle       - kaggle-pack pipeline (download + pack + upload)
    slrq         - SLRQ experimental quantizer
    cli          - argparse + command dispatch + __main__ entry
"""

from orka._impl import (
    BackgroundWriter,
    CappedOutOfMemoryError,
    ORKA_VERSION,
    PayloadEstimate,
    build_parser,
    classify_tensor_family,
    estimate_payload,
    eval_artifact,
    eval_sweep,
    inspect_checkpoint,
    is_rvq_mixed_spec,
    learn_codebook_auto,
    pack_checkpoint,
    parse_quant_spec,
    quality_metrics_from_flat,
    quant_spec_from_sizes,
    quantize_block_salient_slrq_vectorized,
    quantize_vectors_auto,
    reconstruct_artifact,
    report_artifact,
    rvq_mixed_family_stages,
    sweep_checkpoint,
    verify_artifact,
)

# Test-private symbols (orka_test.py imports these)
from orka._impl import (
    _parse_params,
    _summarize_eval_rows,
)

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
