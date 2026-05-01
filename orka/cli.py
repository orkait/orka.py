"""argparse-based CLI: build_parser + cmd_* dispatchers + main entry."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from orka.activations import _load_awq_activations
from orka.core import (
    _apply_gpu_memory_cap,
    _human_bytes,
    _parse_params,
    _wrap_capped_oom,
    estimate_payload,
)
from orka.decode import (
    reconstruct_artifact,
    report_artifact,
    verify_artifact,
)
from orka.eval import eval_artifact, eval_sweep
from orka.kaggle import cmd_kaggle_pack
from orka.pack import inspect_checkpoint, pack_checkpoint
from orka.quant_spec import (
    is_rvq_mixed_spec,
    parse_quant_spec,
    quant_spec_from_sizes,
    rvq_mixed_family_stages,
    _resolve_quant_stages,
)
from orka.slrq import cmd_slrq_eval
from orka.sweep import sweep_checkpoint


def cmd_calc(args: argparse.Namespace) -> int:
    estimate = estimate_payload(
        params=_parse_params(args.params),
        group_size=args.group_size,
        codebook_size=args.codebook_size,
        scale_block_vectors=args.scale_block_vectors,
        scale_bits=args.scale_bits,
    )
    data = asdict(estimate)
    data["index_size"] = _human_bytes(estimate.index_bytes)
    data["scale_size"] = _human_bytes(estimate.scale_bytes)
    data["total_payload_size"] = _human_bytes(estimate.total_payload_bytes)
    print(json.dumps(data, indent=2))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    report = inspect_checkpoint(Path(args.source))
    report["baseline_vq8"] = asdict(estimate_payload(report["total_params"], 8, 256))
    print(json.dumps(report, indent=2))
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    _apply_gpu_memory_cap(args.backend, args.device, args.max_gpu_mem_gb)
    awq_activations = _load_awq_activations(args)

    if is_rvq_mixed_spec(args.quant_mode):
        family_map = rvq_mixed_family_stages()
        sizes = [family_map["other"][0]]
        codebook_mode = "per-tensor"
    else:
        family_map = None
        sizes = _resolve_quant_stages(
            args.quant_mode, args.codebook_sizes, args.codebook_size
        )
        codebook_mode = args.codebook_mode
    smap = None
    if getattr(args, "sensitivity_map", None):
        with open(args.sensitivity_map, "r") as f:
            smap = json.load(f)
    manifest = _wrap_capped_oom(
        args.max_gpu_mem_gb,
        pack_checkpoint,
        source=Path(args.source),
        out_dir=Path(args.out),
        group_size=args.group_size,
        codebook_size=sizes[0],
        iterations=args.iterations,
        max_values_per_tensor=args.max_values_per_tensor,
        codebook_mode=codebook_mode,
        sample_vectors=args.sample_vectors,
        backend=args.backend,
        normalization=args.normalization,
        device=args.device,
        codebook_sizes=sizes if family_map is None else None,
        family_stages_map=family_map,
        outlier_frac=args.outlier_frac,
        rotation=args.rotation,
        rotation_seed=args.rotation_seed,
        awq_activations=awq_activations,
        awq_alpha=args.awq_alpha,
        max_tensors=args.max_tensors,
        sensitivity_map=smap,
        progress_file=Path(args.progress_file) if args.progress_file else None,
        codebook_cache_dir=Path(args.codebook_cache).expanduser()
        if args.codebook_cache
        else None,
        block_scale_size=args.block_scale_size,
    )
    print(
        json.dumps(
            {
                "out": args.out,
                "tensor_count": manifest["tensor_count"],
                "total_index_bytes": manifest["total_index_bytes"],
            },
            indent=2,
        )
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report = report_artifact(Path(args.artifact))
    report["artifact_size"] = _human_bytes(report["artifact_bytes"])
    report["original_fp16_size"] = _human_bytes(report["original_fp16_bytes"])
    report["index_size"] = _human_bytes(report["total_index_bytes"])
    report["codebook_size"] = _human_bytes(report["total_codebook_bytes"])
    report["scale_size"] = _human_bytes(report["total_scale_bytes"])
    print(json.dumps(report, indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    result = verify_artifact(Path(args.artifact))
    print(json.dumps(result, indent=2))
    return 0


def cmd_reconstruct(args: argparse.Namespace) -> int:
    result = reconstruct_artifact(
        Path(args.artifact), Path(args.out), output_format=args.format
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    _apply_gpu_memory_cap(args.backend, args.device, args.max_gpu_mem_gb)
    awq_activations = _load_awq_activations(args)

    cb_sizes = list(args.codebook_sizes) if args.codebook_sizes else []
    qmodes = list(args.quant_modes) if args.quant_modes else []
    if not cb_sizes and not qmodes:
        cb_sizes = [256]

    smap = None
    if getattr(args, "sensitivity_map", None):
        with open(args.sensitivity_map, "r") as f:
            smap = json.load(f)

    result = _wrap_capped_oom(
        args.max_gpu_mem_gb,
        sweep_checkpoint,
        outlier_frac=args.outlier_frac,
        rotation=args.rotation,
        rotation_seed=args.rotation_seed,
        source=Path(args.source),
        out_path=Path(args.out),
        group_sizes=args.group_sizes,
        codebook_sizes=cb_sizes,
        codebook_modes=args.codebook_modes,
        normalizations=args.normalizations,
        iterations=args.iterations,
        max_values_per_tensor=args.max_values_per_tensor,
        sample_vectors=args.sample_vectors,
        backend=args.backend,
        device=args.device,
        verify_runs=args.verify,
        quant_modes=qmodes,
        awq_activations=awq_activations,
        awq_alpha=args.awq_alpha,
        awq_alphas=args.awq_alphas,
        max_tensors=args.max_tensors,
        sensitivity_map=smap,
        progress_file=Path(args.progress_file) if args.progress_file else None,
    )
    print(
        json.dumps(
            {
                "out": result["out"],
                "artifact_root": result["artifact_root"],
                "run_count": result["run_count"],
                "best_by_relative_rmse": result["best_by_relative_rmse"],
                "best_by_cosine_per_mb": result["best_by_cosine_per_mb"],
            },
            indent=2,
        )
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    try:
        result = eval_artifact(
            artifact_dir=Path(args.artifact),
            prompts_path=Path(args.prompts),
            out_path=Path(args.out),
            model_dir=Path(args.model_dir) if args.model_dir else None,
            max_prompts=args.max_prompts,
            max_length=args.max_length,
            device=args.device,
            reconstructed_model_dir=Path(args.reconstructed_model_dir)
            if args.reconstructed_model_dir
            else None,
            local_files_only=not args.allow_download,
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=os.sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "out": args.out,
                "artifact": result["artifact"],
                "prompt_count": result["prompt_count"],
                "token_count": result["token_count"],
                "original_loss": result["original_loss"],
                "orka_loss": result["orka_loss"],
                "loss_delta": result["loss_delta"],
                "original_perplexity": result["original_perplexity"],
                "orka_perplexity": result["orka_perplexity"],
                "perplexity_ratio": result["perplexity_ratio"],
            },
            indent=2,
        )
    )
    return 0


def cmd_eval_sweep(args: argparse.Namespace) -> int:
    try:
        result = eval_sweep(
            sweep_path=Path(args.sweep),
            prompts_path=Path(args.prompts),
            out_path=Path(args.out),
            model_dir=Path(args.model_dir) if args.model_dir else None,
            max_prompts=args.max_prompts,
            max_length=args.max_length,
            device=args.device,
            local_files_only=not args.allow_download,
            max_runs=args.max_runs,
            reconstructed_model_root=(
                Path(args.reconstructed_model_root)
                if args.reconstructed_model_root
                else None
            ),
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=os.sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "out": args.out,
                "eval_root": result["eval_root"],
                "run_count": result["run_count"],
                "best_by_loss_delta": result["best_by_loss_delta"],
                "best_by_perplexity_ratio": result["best_by_perplexity_ratio"],
                "best_by_artifact_bytes": result["best_by_artifact_bytes"],
            },
            indent=2,
        )
    )
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orka model compiler prototype")
    sub = parser.add_subparsers(dest="command", required=True)

    calc = sub.add_parser("calc", help="estimate Orka payload size")
    calc.add_argument(
        "--params", required=True, help="parameter count, for example 8.03b"
    )
    calc.add_argument("--group-size", type=int, default=8)
    calc.add_argument("--codebook-size", type=int, default=256)
    calc.add_argument("--scale-block-vectors", type=int, default=64)
    calc.add_argument("--scale-bits", type=int, default=16)
    calc.set_defaults(func=cmd_calc)

    inspect = sub.add_parser(
        "inspect", help="inspect a safetensors or PyTorch checkpoint"
    )
    inspect.add_argument("source")
    inspect.set_defaults(func=cmd_inspect)

    def add_pack_args(p):
        p.add_argument("--group-size", type=int, default=8)
        p.add_argument("--codebook-size", type=int, default=256)
        p.add_argument(
            "--codebook-sizes",
            type=int,
            nargs="+",
            default=None,
            help="explicit per-stage codebook sizes (overrides --codebook-size and --quant-mode)",
        )
        p.add_argument(
            "--quant-mode",
            default=None,
            help="compositional spec like vq-8 or vq-16-8 (per-stage bits, 1..16, total ≤ 64)",
        )
        p.add_argument(
            "--codebook-mode",
            choices=["per-tensor", "global", "family"],
            default="per-tensor",
        )
        p.add_argument(
            "--backend", choices=["auto", "numpy", "torch"], default="auto"
        )
        p.add_argument(
            "--device",
            default="cpu",
            help="torch backend device, for example cpu, cuda, cuda:0, or auto",
        )
        p.add_argument(
            "--normalization",
            choices=["none", "row-l2", "col-l2", "block-max", "awq", "awq-block-max", "slrq-block"],
            default="none",
        )
        p.add_argument(
            "--block-scale-size",
            type=int,
            default=32,
            help="elements per block when --normalization block-max (typical 16 or 32)",
        )
        p.add_argument(
            "--rotation",
            choices=["none", "orthogonal", "hadamard"],
            default="none",
            help="rotation along inner axis before VQ. orthogonal: per-tensor seeded random orthogonal (any size). hadamard: deterministic FWHT (requires power-of-2 last dim).",
        )
        p.add_argument(
            "--rotation-seed",
            type=int,
            default=None,
            help="seed for orthogonal rotation (deterministic)",
        )
        p.add_argument("--sample-vectors", type=int, default=None)
        p.add_argument("--iterations", type=int, default=12)
        p.add_argument("--max-values-per-tensor", type=int, default=None)
        p.add_argument(
            "--max-gpu-mem-gb",
            type=float,
            default=None,
            help="strict cap on per-process GPU memory (GB)",
        )
        p.add_argument(
            "--outlier-frac",
            type=float,
            default=0.0,
            help="fraction of top-magnitude weights kept as fp16 sidecar (e.g. 0.001 = 0.1%%)",
        )
        p.add_argument(
            "--awq-calibration",
            default=None,
            help="prompts file for AWQ calibration; enables activation-aware VQ",
        )
        p.add_argument(
            "--awq-model-dir",
            default=None,
            help="HF model dir for AWQ activation collection",
        )
        p.add_argument(
            "--awq-alpha",
            type=float,
            default=0.5,
            help="activation magnitude scaling power (default 0.5)",
        )
        p.add_argument("--calibration-max-prompts", type=int, default=32)
        p.add_argument("--calibration-max-length", type=int, default=256)
        p.add_argument(
            "--calibration-max-samples",
            type=int,
            default=4096,
            help="max activation samples retained per layer for AWQ calibration",
        )
        p.add_argument("--progress-file", help="file to write real-time progress status")
        p.add_argument(
            "--sensitivity-map",
            help="JSON file from sensitivity.py to enable mixed-precision",
        )
        p.add_argument(
            "--max-tensors",
            type=int,
            default=None,
            help="limit pack to first N tensors (for fail-fast iteration)",
        )
        p.add_argument(
            "--codebook-cache",
            default=None,
            help="dir to cache stage-0 codebooks (zero-loss reuse on identical configs)",
        )

    pack = sub.add_parser(
        "pack", help="pack candidate weight tensors into an .orka directory"
    )
    pack.add_argument("source")
    pack.add_argument("--out", required=True)
    add_pack_args(pack)
    pack.set_defaults(func=cmd_pack)

    kp = sub.add_parser(
        "kaggle-pack", help="Download from HF, pack on Kaggle, and upload back to HF"
    )
    kp.add_argument("--repo-id", required=True, help="HF model repo to download")
    kp.add_argument(
        "--out",
        default=None,
        help="output .orka directory (default on Kaggle: /kaggle/working/<slug>.orka)",
    )
    kp.add_argument("--upload-repo", help="HF repo to upload the result to")
    add_pack_args(kp)
    kp.add_argument("--run-eval", action="store_true",
                    help="run perplexity eval after packing")
    kp.add_argument("--eval-prompts", default=None,
                    help="prompts file for perplexity eval (defaults to AWQ calibration file)")
    kp.add_argument("--eval-max-prompts", type=int, default=16)
    kp.add_argument("--eval-max-length", type=int, default=128)
    kp.set_defaults(func=cmd_kaggle_pack)

    report = sub.add_parser("report", help="summarize an .orka artifact")
    report.add_argument("artifact")
    report.set_defaults(func=cmd_report)

    verify = sub.add_parser(
        "verify", help="decode an .orka artifact and recompute source MSE"
    )
    verify.add_argument("artifact")
    verify.set_defaults(func=cmd_verify)

    reconstruct = sub.add_parser(
        "reconstruct", help="decode an .orka artifact to JSON tensors"
    )
    reconstruct.add_argument("artifact")
    reconstruct.add_argument("--out", required=True)
    reconstruct.add_argument(
        "--format", choices=["json", "safetensors"], default="json"
    )
    reconstruct.set_defaults(func=cmd_reconstruct)

    sweep = sub.add_parser(
        "sweep", help="run a pack/report matrix and write a JSON summary"
    )
    sweep.add_argument("source")
    sweep.add_argument("--out", required=True)
    sweep.add_argument("--group-sizes", type=int, nargs="+", default=[8])
    sweep.add_argument(
        "--codebook-sizes",
        type=int,
        nargs="+",
        default=None,
        help="single-stage codebook sizes to sweep",
    )
    sweep.add_argument(
        "--quant-modes",
        nargs="+",
        default=None,
        help="compositional specs (e.g. vq-8 vq-16 vq-16-8 vq-16-16-16-16)",
    )
    sweep.add_argument(
        "--codebook-modes",
        choices=["per-tensor", "global", "family"],
        nargs="+",
        default=["global"],
    )
    sweep.add_argument(
        "--normalizations",
        choices=["none", "row-l2", "col-l2", "block-max", "awq", "awq-block-max", "slrq-block"],
        nargs="+",
        default=["none", "row-l2"],
    )
    sweep.add_argument(
        "--rotation", choices=["none", "orthogonal", "hadamard"], default="none"
    )
    sweep.add_argument("--rotation-seed", type=int, default=None)
    sweep.add_argument(
        "--backend", choices=["auto", "numpy", "torch"], default="auto"
    )
    sweep.add_argument(
        "--device",
        default="cpu",
        help="torch backend device, for example cpu, cuda, cuda:0, or auto",
    )
    sweep.add_argument("--sample-vectors", type=int, default=None)
    sweep.add_argument("--iterations", type=int, default=12)
    sweep.add_argument("--max-values-per-tensor", type=int, default=None)
    sweep.add_argument(
        "--verify",
        action="store_true",
        help="verify every sweep artifact after packing",
    )
    sweep.add_argument(
        "--max-gpu-mem-gb",
        type=float,
        default=None,
        help="strict cap on per-process GPU memory (GB)",
    )
    sweep.add_argument(
        "--progress-file", help="file to write real-time progress status"
    )
    sweep.add_argument(
        "--max-tensors", type=int, default=None, help="limit sweep to first N tensors"
    )
    sweep.add_argument(
        "--outlier-frac",
        type=float,
        default=0.0,
        help="fraction of top-magnitude weights kept as fp16 sidecar",
    )
    sweep.add_argument(
        "--awq-calibration",
        default=None,
        help="prompts file for AWQ calibration; enables activation-aware VQ",
    )
    sweep.add_argument(
        "--awq-model-dir",
        default=None,
        help="HF model dir for AWQ activation collection",
    )
    sweep.add_argument(
        "--awq-alpha",
        type=float,
        default=0.5,
        help="activation magnitude scaling power (default 0.5)",
    )
    sweep.add_argument(
        "--awq-alphas",
        type=float,
        nargs="+",
        default=None,
        help="sweep multiple AWQ alphas in one run; overrides --awq-alpha when set",
    )
    sweep.add_argument("--calibration-max-prompts", type=int, default=32)
    sweep.add_argument("--calibration-max-length", type=int, default=256)
    sweep.add_argument("--calibration-max-samples", type=int, default=4096)
    sweep.set_defaults(func=cmd_sweep)

    eval_cmd = sub.add_parser(
        "eval", help="evaluate an .orka artifact with Hugging Face prompt loss"
    )
    eval_cmd.add_argument("artifact")
    eval_cmd.add_argument(
        "--prompts", required=True, help="text file with one prompt per non-empty line"
    )
    eval_cmd.add_argument("--out", required=True)
    eval_cmd.add_argument(
        "--model-dir", default=None, help="override Hugging Face model directory"
    )
    eval_cmd.add_argument("--max-prompts", type=int, default=None)
    eval_cmd.add_argument("--max-length", type=int, default=512)
    eval_cmd.add_argument("--device", default="cpu")
    eval_cmd.add_argument("--reconstructed-model-dir", default=None)
    eval_cmd.add_argument(
        "--allow-download",
        action="store_true",
        help="allow transformers to download missing files",
    )
    eval_cmd.set_defaults(func=cmd_eval)

    eval_sweep_cmd = sub.add_parser(
        "eval-sweep", help="evaluate every artifact recorded in a sweep JSON"
    )
    eval_sweep_cmd.add_argument("sweep")
    eval_sweep_cmd.add_argument(
        "--prompts", required=True, help="text file with one prompt per non-empty line"
    )
    eval_sweep_cmd.add_argument("--out", required=True)
    eval_sweep_cmd.add_argument(
        "--model-dir", default=None, help="override Hugging Face model directory"
    )
    eval_sweep_cmd.add_argument("--max-prompts", type=int, default=None)
    eval_sweep_cmd.add_argument("--max-length", type=int, default=512)
    eval_sweep_cmd.add_argument("--device", default="cpu")
    eval_sweep_cmd.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="evaluate only the first N sweep runs",
    )
    eval_sweep_cmd.add_argument(
        "--reconstructed-model-root",
        default=None,
        help="keep reconstructed model directories under this root",
    )
    eval_sweep_cmd.add_argument(
        "--allow-download",
        action="store_true",
        help="allow transformers to download missing files",
    )
    eval_sweep_cmd.set_defaults(func=cmd_eval_sweep)

    def _run_tests(_args):
        import unittest

        loader = unittest.defaultTestLoader
        suite = loader.discover("tests", top_level_dir=".")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    slrq = sub.add_parser("slrq-eval", help="Test SLRQ hypothesis directly on a HuggingFace model in memory")
    slrq.add_argument("--model-id", required=True, help="HF model ID or path")
    slrq.add_argument("--prompts", default=None, help="Optional text file of prompts")
    slrq.add_argument("--max-prompts", type=int, default=16)
    slrq.add_argument("--block-size", type=int, default=16)
    slrq.add_argument("--bits", type=int, default=4)
    slrq.set_defaults(func=cmd_slrq_eval)

    selftest = sub.add_parser("selftest", help="run built-in tests")
    selftest.set_defaults(func=_run_tests)
    return parser

def main() -> int:
    """Programmatic entry point. Auto-bootstraps Kaggle config when invoked
    on Kaggle with no CLI args, otherwise just parses argv and dispatches."""
    import sys as _sys

    if Path("/kaggle/working").exists():
        # ── KAGGLE CONFIG ─────────────────────────────────────────────────────
        # Edit these values before pushing to Kaggle.
        # Defaults match the best-loss config (smollm2-135m-ultimate):
        #   awq-block-max + family + orthogonal rotation + sensitivity skip.
        _KAGGLE_CONFIG = {
            "repo_id":         "Qwen/Qwen3-0.6B",
            "upload_repo":     None,
            "quant_mode":      "rvq-16-8-8",       # 3 stages: [65536, 256, 256] = 4 bits/weight
            "codebook_mode":   "per-tensor",      # each tensor gets own codebook (gate/up_proj need this)
            "normalization":   "awq-block-max",   # AWQ scaling + block-max with real Wikitext calib
            "rotation":        "orthogonal",      # smear outliers
            "rotation_seed":   42,
            "backend":         "torch",
            "device":          "cuda",
            "max_gpu_mem_gb":  14.0,
            "sample_vectors":  1000000,
            "iterations":      12,
            "outlier_frac":    0.001,             # top 0.1% values escape as fp16 sidecar
            "group_size":      8,
            "codebook_size":   256,
            "awq_calibration": True,              # ON - use real Wikitext for activations
            "awq_alpha":       0.5,
            "calibration_max_prompts": 128,
            "calibration_max_length":  512,
            "skip_sensitive":  True,              # skip lm_head + embed_tokens (FP16 passthrough)
            "run_eval":        True,              # run perplexity eval after pack
            "eval_max_prompts": 64,
            "eval_max_length":  256,
        }
        # ── END CONFIG ────────────────────────────────────────────────────────

        if len(_sys.argv) == 1:
            print("Kaggle: building args from _KAGGLE_CONFIG", flush=True)
            cfg = _KAGGLE_CONFIG

            # Download Wikitext-2 samples for AWQ calibration + perplexity eval.
            calib_path = Path("/tmp/orka_calib_prompts.txt")
            if cfg.get("awq_calibration") or cfg.get("run_eval"):
                try:
                    from datasets import load_dataset
                    print("Kaggle: loading Wikitext-2-raw test split ...", flush=True)
                    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
                    samples = []
                    target = max(int(cfg.get("calibration_max_prompts", 128)),
                                 int(cfg.get("eval_max_prompts", 64))) + 32
                    for row in ds:
                        text = (row.get("text") or "").strip()
                        if len(text) >= 200:  # skip headers / short fragments
                            samples.append(text)
                            if len(samples) >= target:
                                break
                    if not samples:
                        raise RuntimeError("no usable Wikitext samples")
                    calib_path.write_text("\n".join(samples))
                    print(f"Kaggle: wrote {len(samples)} Wikitext samples to {calib_path}", flush=True)
                except Exception as exc:
                    print(f"Kaggle: Wikitext fetch failed ({exc}); falling back to inline prompts", flush=True)
                    calib_path.write_text("\n".join([
                        "The history of artificial intelligence began in antiquity.",
                        "Quantum mechanics describes physical properties of nature.",
                        "Climate change refers to long-term shifts in temperatures.",
                        "Machine learning algorithms build a model from data.",
                        "The theory of relativity is a theory of gravitation.",
                        "Photosynthesis is the process by which green plants synthesize foods.",
                        "DNA carries genetic instructions for the development of organisms.",
                        "Black holes are regions of spacetime where gravity is strong.",
                        "Neural networks are inspired by biological neural networks.",
                        "Cellular respiration converts biochemical energy from nutrients.",
                        "The water cycle describes movement of water on Earth.",
                        "Stars are luminous spheres of plasma held together by gravity.",
                        "Programming languages produce various kinds of output.",
                        "Mathematics is the abstract science of number, quantity, and space.",
                        "The Milky Way galaxy contains our Solar System.",
                        "Vaccines stimulate the immune system to combat pathogens.",
                    ]))

            # Stub sensitivity map: skip lm_head + embed_tokens (orka skips loss_delta>1.5 OR embed/lm_head substring).
            smap_path = Path("/tmp/orka_sensitivity_map.json")
            if cfg.get("skip_sensitive"):
                import json as _json
                smap_path.write_text(_json.dumps({
                    "base_loss": 0.0,
                    "layers": [
                        {"layer": "lm_head", "loss_delta": 999.0, "sensitivity": "high"},
                        {"layer": "model.embed_tokens", "loss_delta": 999.0, "sensitivity": "high"},
                    ],
                }))
                print(f"Kaggle: wrote sensitivity stub to {smap_path}", flush=True)

            _sys.argv += [
                "kaggle-pack",
                "--repo-id",        cfg["repo_id"],
                "--quant-mode",     cfg["quant_mode"],
                "--codebook-mode",  cfg["codebook_mode"],
                "--normalization",  cfg["normalization"],
                "--rotation",       cfg["rotation"],
                "--backend",        cfg["backend"],
                "--device",         cfg["device"],
                *(["--max-gpu-mem-gb", str(cfg["max_gpu_mem_gb"])] if cfg.get("max_gpu_mem_gb") is not None else []),
                *(["--rotation-seed", str(cfg["rotation_seed"])] if cfg.get("rotation_seed") is not None else []),
                "--sample-vectors", str(cfg["sample_vectors"]),
                "--iterations",     str(cfg["iterations"]),
                "--outlier-frac",   str(cfg["outlier_frac"]),
                "--group-size",     str(cfg["group_size"]),
                "--codebook-size",  str(cfg["codebook_size"]),
            ]
            if cfg.get("awq_calibration"):
                _sys.argv += [
                    "--awq-calibration", str(calib_path),
                    "--awq-alpha",       str(cfg["awq_alpha"]),
                    "--calibration-max-prompts", str(cfg.get("calibration_max_prompts", 32)),
                    "--calibration-max-length",  str(cfg.get("calibration_max_length", 256)),
                ]
            if cfg.get("skip_sensitive"):
                _sys.argv += ["--sensitivity-map", str(smap_path)]
            if cfg.get("run_eval"):
                # Always write a small prompts file for eval (even if AWQ off, we still need prompts).
                if not calib_path.exists():
                    calib_path.write_text("\n".join([
                        "The history of artificial intelligence began in antiquity.",
                        "Quantum mechanics describes physical properties of nature.",
                        "Climate change refers to long-term shifts in temperatures.",
                        "Machine learning algorithms build a model from data.",
                        "The theory of relativity is a theory of gravitation.",
                        "Photosynthesis is the process by which green plants synthesize foods.",
                        "DNA carries genetic instructions for the development of organisms.",
                        "Black holes are regions of spacetime where gravity is strong.",
                        "Neural networks are inspired by biological neural networks.",
                        "Cellular respiration converts biochemical energy from nutrients.",
                        "The water cycle describes movement of water on Earth.",
                        "Stars are luminous spheres of plasma held together by gravity.",
                        "Programming languages produce various kinds of output.",
                        "Mathematics is the abstract science of number, quantity, and space.",
                        "The Milky Way galaxy contains our Solar System.",
                        "Vaccines stimulate the immune system to combat pathogens.",
                    ]))
                _sys.argv += [
                    "--run-eval",
                    "--eval-prompts",     str(calib_path),
                    "--eval-max-prompts", str(cfg["eval_max_prompts"]),
                    "--eval-max-length",  str(cfg["eval_max_length"]),
                ]
            if cfg.get("upload_repo"):
                _sys.argv += ["--upload-repo", cfg["upload_repo"]]

    cli_args = build_parser().parse_args()
    return int(cli_args.func(cli_args))

