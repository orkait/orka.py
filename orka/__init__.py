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

import importlib

# Lazy public API (PEP 562). Eagerly importing every submodule here pulled the
# full dependency tree - notably torch (~1s) via orka.integrations.layers - into
# every `import orka` and `python -m orka` (including `--help`, arg-validation
# errors, and numpy-backend runs that never touch torch). Each public name is
# resolved to its module on first access and then cached in globals().
_LAZY_ATTRS = {
    "ORKA_VERSION": "orka.core._format",
    "BackgroundWriter": "orka._runtime",
    "CappedOutOfMemoryError": "orka._runtime",
    "_parse_params": "orka.core._util",
    "inspect_checkpoint": "orka.core._checkpoint",
    "build_parser": "orka.cli",
    "main": "orka.cli",
    "learn_codebook_auto": "orka.codebook",
    "quantize_vectors_auto": "orka.codebook",
    "_summarize_eval_rows": "orka.eval",
    "eval_artifact": "orka.eval",
    "eval_sweep": "orka.eval",
    "quality_metrics_from_flat": "orka.eval.metrics",
    "pack_checkpoint": "orka.pipeline.pack",
    "merge_orka_artifacts": "orka.artifact.merge",
    "PayloadEstimate": "orka.quant",
    "classify_tensor_family": "orka.quant",
    "estimate_payload": "orka.quant",
    "is_rvq_mixed_spec": "orka.quant",
    "parse_quant_spec": "orka.quant",
    "quant_spec_from_sizes": "orka.quant",
    "rvq_mixed_family_stages": "orka.quant",
    "reconstruct_artifact": "orka.artifact.reconstruct",
    "report_artifact": "orka.eval.report",
    "sweep_checkpoint": "orka.eval.sweep",
    "verify_artifact": "orka.eval.verify",
    "OrkaLinear": "orka.integrations.layers",
    "replace_linear_with_orka": "orka.integrations.layers",
}


def __getattr__(name: str):
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'orka' has no attribute {name!r}")
    # Layer helpers degrade gracefully when torch is absent (parity with the
    # prior try/except stub).
    if name in ("OrkaLinear", "replace_linear_with_orka"):
        try:
            module = importlib.import_module(module_path)
        except Exception:
            if name == "OrkaLinear":
                return None

            def replace_linear_with_orka(*_args, **_kwargs):
                raise RuntimeError(
                    "Torch is required for layer helpers. Install torch to use OrkaLinear."
                )

            return replace_linear_with_orka
    else:
        module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value  # cache so __getattr__ won't fire again
    return value


def __dir__():
    return sorted(__all__)


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
