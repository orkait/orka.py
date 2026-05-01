"""All cmd_* dispatchers. Each takes argparse.Namespace, returns int exit code."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from orka._runtime import (
    _apply_cpu_cap,
    _apply_gpu_memory_cap,
    _apply_system_ram_cap,
    _stop_ram_monitor,
    _wrap_capped_oom,
)
from orka._util import _human_bytes, _parse_params
from orka.activations import _load_awq_activations
from orka.deploy.kaggle import cmd_kaggle_pack
from orka.eval import eval_artifact, eval_sweep
from orka.pipeline.pack import pack_checkpoint
from orka.quant import (
    _resolve_quant_stages,
    estimate_payload,
    is_rvq_mixed_spec,
    rvq_mixed_family_stages,
)
from orka.reconstruct import reconstruct_artifact
from orka.report import report_artifact
from orka.sweep import sweep_checkpoint
from orka.verify import verify_artifact
from orka._checkpoint import inspect_checkpoint


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
    _apply_system_ram_cap(args.max_system_ram_gb)
    _apply_cpu_cap(args.max_cpu_threads)
    try:
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
    finally:
        _stop_ram_monitor()


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
    _apply_system_ram_cap(args.max_system_ram_gb)
    _apply_cpu_cap(args.max_cpu_threads)
    try:
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
    finally:
        _stop_ram_monitor()


def cmd_eval(args: argparse.Namespace) -> int:
    _apply_gpu_memory_cap("torch", args.device, getattr(args, "max_gpu_mem_gb", None))
    _apply_system_ram_cap(getattr(args, "max_system_ram_gb", None))
    _apply_cpu_cap(getattr(args, "max_cpu_threads", None))
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
    finally:
        _stop_ram_monitor()
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
    _apply_gpu_memory_cap("torch", args.device, getattr(args, "max_gpu_mem_gb", None))
    _apply_system_ram_cap(getattr(args, "max_system_ram_gb", None))
    _apply_cpu_cap(getattr(args, "max_cpu_threads", None))
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
    finally:
        _stop_ram_monitor()
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
