"""Eval-family CLI command handlers (extracted from cli.commands)."""
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
    _resolve_auto_backend,
    _stop_ram_monitor,
    _wrap_capped_oom,
)
from orka.core._util import _human_bytes, _parse_params
from orka.quant.activations import _load_awq_activations
from orka.deploy.kaggle import cmd_kaggle_pack
from orka.eval import eval_artifact, eval_sweep, pulse_check_artifact
from orka.pipeline.pack import pack_checkpoint
from orka.quant.spec import (
    _resolve_quant_stages,
    estimate_payload,
    is_rvq_mixed_spec,
    rvq_mixed_family_stages,
)
from orka.quant.semantic import (
    cmd_sem_analyze,
    cmd_sem_map,
)
from orka.artifact.reconstruct import reconstruct_artifact
from orka.eval.report import report_artifact
from orka.eval.sweep import sweep_checkpoint
from orka.eval.verify import verify_artifact
from orka.core._checkpoint import inspect_checkpoint
from orka.artifact.merge import merge_orka_artifacts


def cmd_sweep(args: argparse.Namespace) -> int:
    args.backend = _resolve_auto_backend(args.backend)
    _apply_gpu_memory_cap(args.backend, args.device, args.max_gpu_mem_gb)
    _apply_system_ram_cap(args.max_system_ram_gb, getattr(args, "workload_budget_gb", None))
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
            em_aq_passes=getattr(args, "em_aq_passes", 3),
            codebook_cache_dir=Path(args.codebook_cache).expanduser()
            if getattr(args, "codebook_cache", None)
            else None,
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
    _apply_system_ram_cap(
        getattr(args, "max_system_ram_gb", None),
        getattr(args, "workload_budget_gb", None),
    )
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


def cmd_pulse_check(args: argparse.Namespace) -> int:
    _apply_gpu_memory_cap("torch", args.device, getattr(args, "max_gpu_mem_gb", None))
    _apply_system_ram_cap(
        getattr(args, "max_system_ram_gb", None),
        getattr(args, "workload_budget_gb", None),
    )
    _apply_cpu_cap(getattr(args, "max_cpu_threads", None))
    try:
        result = pulse_check_artifact(
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
                "kl_divergence": result["kl_divergence"],
                "top1_agreement": result["top1_agreement"],
                "total_tokens": result["total_tokens"],
            },
            indent=2,
        )
    )
    return 0


def cmd_eval_sweep(args: argparse.Namespace) -> int:
    _apply_gpu_memory_cap("torch", args.device, getattr(args, "max_gpu_mem_gb", None))
    _apply_system_ram_cap(
        getattr(args, "max_system_ram_gb", None),
        getattr(args, "workload_budget_gb", None),
    )
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
