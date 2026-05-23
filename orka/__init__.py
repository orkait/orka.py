"""Orka compiler package.

Layout:
    _format        - .orka manifest + ALL sidecar I/O (single source of truth)
    _checkpoint    - source loaders (.safetensors / .pt / .bin / .json) + inspect
    _tensor        - backend primitives (numpy/torch dispatch, shape, sample, decode)
    _runtime       - device, GPU memory cap, OOM, BackgroundWriter
    _util          - generic stdlib helpers (numbers, fs, seeds, progress)
    quant/         - vq-/rvq- spec + family + payload size estimation
    transforms/    - normalize / rotate / outliers
    codebook/      - kmeans + cache + assign + learn
    metrics        - reconstruction quality
    activations    - AWQ activation calibration
    pipeline/      - pack_checkpoint + decode (numpy + torch) orchestrators
    verify         - verify_artifact
    report         - report_artifact
    reconstruct    - reconstruct_artifact
    sweep          - sweep_checkpoint
    eval/          - prompts + HF + eval orchestrators
    deploy/        - kaggle pack + upload + bootstrap
    cli/           - parser + commands + main()
"""

from orka._format import ORKA_VERSION
from orka._runtime import BackgroundWriter, CappedOutOfMemoryError
from orka._util import _parse_params
from orka._checkpoint import inspect_checkpoint
from orka.cli import build_parser, main
from orka.codebook import learn_codebook_auto, quantize_vectors_auto
from orka.eval import _summarize_eval_rows, eval_artifact, eval_sweep
from orka.metrics import quality_metrics_from_flat
from orka.pipeline.pack import pack_checkpoint
from orka.merge import merge_orka_artifacts
from orka.quant import (
    PayloadEstimate,
    classify_tensor_family,
    estimate_payload,
    is_rvq_mixed_spec,
    parse_quant_spec,
    quant_spec_from_sizes,
    rvq_mixed_family_stages,
)
from orka.reconstruct import reconstruct_artifact
from orka.report import report_artifact
from orka.sweep import sweep_checkpoint
from orka.verify import verify_artifact
try:
    from orka.layers import OrkaLinear, replace_linear_with_orka
except Exception:
    OrkaLinear = None

    def replace_linear_with_orka(*_args, **_kwargs):
        raise RuntimeError(
            "Torch is required for layer helpers. Install torch to use OrkaLinear."
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
    "main",
    "pack_checkpoint",
    "parse_quant_spec",
    "quality_metrics_from_flat",
    "quant_spec_from_sizes",
    "quantize_vectors_auto",
    "reconstruct_artifact",
    "merge_orka_artifacts",
    "report_artifact",
    "rvq_mixed_family_stages",
    "sweep_checkpoint",
    "verify_artifact",
    "OrkaLinear",
    "replace_linear_with_orka",
]
