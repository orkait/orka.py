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
from orka.reconstruct import reconstruct_artifact
from orka.report import report_artifact
from orka.sweep import sweep_checkpoint
from orka.verify import verify_artifact
from orka._checkpoint import inspect_checkpoint
from orka.merge import merge_orka_artifacts


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
    source_input = args.source
    source_file = Path(source_input)
    if not source_file.exists():
        print(f"--- Resolving {source_input} from HF Hub ---", flush=True)
        try:
            from huggingface_hub import snapshot_download
            model_dir = Path(snapshot_download(source_input))
            candidates = sorted(model_dir.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(f"no .safetensors found in {model_dir}")
            # Sharded checkpoints: pass the directory so _load_tensors walks all shards.
            source_file = candidates[0] if len(candidates) == 1 else model_dir
            print(f"  Using source: {source_file.name} ({len(candidates)} shard(s))", flush=True)
        except Exception as exc:
            print(f"Error resolving source: {exc}")
            return 1

    _apply_gpu_memory_cap(args.backend, args.device, args.max_gpu_mem_gb)
    _apply_system_ram_cap(args.max_system_ram_gb, getattr(args, "workload_budget_gb", None))
    _apply_cpu_cap(args.max_cpu_threads)
    try:
        if getattr(args, "sequential_calibration", False):
            return _run_sequential_pack(args, source_file)

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
        tensor_map = _load_allocation_map(args)
        manifest = _wrap_capped_oom(
            args.max_gpu_mem_gb,
            pack_checkpoint,
            source=source_file,
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
            tensor_stages_map=tensor_map,
            outlier_frac=args.outlier_frac,
            rotation=args.rotation,
            rotation_seed=args.rotation_seed,
            awq_activations=awq_activations,
            awq_alpha=args.awq_alpha,
            max_tensors=args.max_tensors,
            only_tensors=args.only_tensors,
            sensitivity_map=smap,
            progress_file=Path(args.progress_file) if args.progress_file else None,
            codebook_cache_dir=Path(args.codebook_cache).expanduser()
            if args.codebook_cache
            else None,
            block_scale_size=args.block_scale_size,
            codebook_dtype=getattr(args, "codebook_dtype", "float16"),
            em_aq_passes=getattr(args, "em_aq_passes", 3),
            slrq_salient=getattr(args, "slrq_salient", True),
            tensor_partition_count=args.tensor_partition_count,
            tensor_partition_index=args.tensor_partition_index,
            error_compensation=getattr(args, "error_compensation", False),
            mse_scale=getattr(args, "mse_scale", False),
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


def _load_allocation_map(args: argparse.Namespace):
    if not getattr(args, "allocation_map", None):
        return None
    from orka.allocate import allocation_tensor_stages

    with open(args.allocation_map, "r") as f:
        allocation = json.load(f)
    return allocation_tensor_stages(allocation)


def _run_sequential_pack(args: argparse.Namespace, source_file: Path) -> int:
    from orka.pipeline.sequential import pack_checkpoint_sequential

    if not args.awq_model_dir or not args.awq_calibration:
        print(
            "Error: --sequential-calibration requires --awq-model-dir and "
            "--awq-calibration.",
            file=os.sys.stderr,
        )
        return 1
    if args.codebook_mode != "per-tensor":
        print(
            "Error: --sequential-calibration requires --codebook-mode per-tensor.",
            file=os.sys.stderr,
        )
        return 1
    if is_rvq_mixed_spec(args.quant_mode):
        print(
            "Error: --sequential-calibration does not support rvq-mixed yet; "
            "use an explicit spec like rvq-16-8.",
            file=os.sys.stderr,
        )
        return 1

    sizes = _resolve_quant_stages(
        args.quant_mode, args.codebook_sizes, args.codebook_size
    )
    manifest = _wrap_capped_oom(
        args.max_gpu_mem_gb,
        pack_checkpoint_sequential,
        source=source_file,
        out_dir=Path(args.out),
        model_dir=Path(args.awq_model_dir),
        prompts_path=Path(args.awq_calibration),
        model_device=args.device if args.backend == "torch" else "cpu",
        calibration_max_prompts=args.calibration_max_prompts,
        calibration_max_length=args.calibration_max_length,
        calibration_max_samples=args.calibration_max_samples,
        progress_file=Path(args.progress_file) if args.progress_file else None,
        group_size=args.group_size,
        codebook_size=sizes[0],
        codebook_sizes=sizes,
        tensor_stages_map=_load_allocation_map(args),
        iterations=args.iterations,
        max_values_per_tensor=args.max_values_per_tensor,
        codebook_mode=args.codebook_mode,
        sample_vectors=args.sample_vectors,
        backend=args.backend,
        normalization=args.normalization,
        device=args.device,
        outlier_frac=args.outlier_frac,
        rotation=args.rotation,
        rotation_seed=args.rotation_seed,
        block_scale_size=args.block_scale_size,
        codebook_dtype=getattr(args, "codebook_dtype", "float16"),
        em_aq_passes=getattr(args, "em_aq_passes", 3),
        slrq_salient=getattr(args, "slrq_salient", True),
        codebook_cache_dir=Path(args.codebook_cache).expanduser()
        if args.codebook_cache
        else None,
    )
    print(
        json.dumps(
            {
                "out": args.out,
                "tensor_count": manifest["tensor_count"],
                "total_index_bytes": manifest["total_index_bytes"],
                "sequential_calibration": True,
            },
            indent=2,
        )
    )
    return 0


def cmd_merge_orka(args: argparse.Namespace) -> int:
    input_artifacts = [Path(path) for path in args.artifacts]
    out_dir = Path(args.out)
    merged = merge_orka_artifacts(input_artifacts=input_artifacts, out_dir=out_dir)
    print(
        json.dumps(
            {
                "out": str(out_dir),
                "tensor_count": merged["tensor_count"],
                "total_index_bytes": merged["total_index_bytes"],
                "partitions": len(input_artifacts),
            },
            indent=2,
        )
    )
    return 0


def cmd_sem_calc(args: argparse.Namespace) -> int:
    """Pre-calculate and save data (like AWQ scales)."""
    awq_activations = _load_awq_activations(args)
    if awq_activations:
        out = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in awq_activations.items()}
        Path(args.out).write_text(json.dumps(out))
        print(f"Calculated and saved data to {args.out}")
        return 0
    print("Nothing to calculate.")
    return 1


def cmd_correct(args: argparse.Namespace) -> int:
    from orka.correct import correct_artifact

    result = correct_artifact(
        Path(args.artifact),
        rank=args.rank,
        device=args.device,
        max_tensors=args.max_tensors,
    )
    print(
        json.dumps(
            {
                "artifact": result["artifact"],
                "tensor_count": result["tensor_count"],
                "improved_count": result["improved_count"],
            },
            indent=2,
        )
    )
    return 0


def cmd_distill(args: argparse.Namespace) -> int:
    from orka.distill import distill_artifact

    activations = None
    if args.activations_file:
        import torch

        path = Path(args.activations_file)
        if not path.exists():
            raise FileNotFoundError(f"activations file not found: {path}")
        try:
            with open(path, "r") as f:
                raw = json.load(f)
            activations = {
                k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items()
            }
        except (UnicodeDecodeError, json.JSONDecodeError):
            activations = torch.load(str(path), map_location="cpu")
    elif args.model_dir and args.prompts:
        from orka.activations import _collect_activations_hf
        from orka.eval.prompts import _read_prompt_file

        prompts = _read_prompt_file(
            Path(args.prompts), max_prompts=args.calibration_max_prompts
        )
        activations = _collect_activations_hf(
            Path(args.model_dir),
            prompts,
            max_length=args.calibration_max_length,
            device=args.device,
            max_samples_per_layer=args.calibration_max_samples,
        )

    result = distill_artifact(
        Path(args.artifact),
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        activations=activations,
        max_tensors=args.max_tensors,
    )
    print(
        json.dumps(
            {
                "artifact": result["artifact"],
                "tensor_count": result["tensor_count"],
                "improved_count": result["improved_count"],
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


def cmd_autoquant(args: argparse.Namespace) -> int:
    import json as _json
    from pathlib import Path
    import numpy as np
    from safetensors import safe_open
    from orka.autoquant.orchestrator import derive_config
    from orka.autoquant.schema import to_allocation_map

    model = Path(args.model)
    sfs = sorted(model.glob("*.safetensors"))
    if not sfs:
        print(f"no safetensors in {model}")
        return 1
    weights: dict[str, np.ndarray] = {}
    for sf in sfs:
        with safe_open(str(sf), "np") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if t.ndim in (1, 2):
                    weights[k] = t.astype("float32")
    llm_fn = None
    if not args.no_llm:
        from orka.autoquant.transport import make_llm_fn, NoLLMBackend
        try:
            llm_fn = make_llm_fn()
        except NoLLMBackend as e:
            print(f"warning: {e}; using deterministic policy only")

    cfg = derive_config(weights, objective=args.objective,
                        use_llm=not args.no_llm, llm_fn=llm_fn)
    Path(args.out).write_text(_json.dumps(to_allocation_map(cfg), indent=2) + "\n")
    n_int8 = sum(1 for c in cfg.values() if c.method == "int8")
    n_rvq = sum(1 for c in cfg.values() if c.method == "rvq")
    n_fp16 = sum(1 for c in cfg.values() if c.method == "fp16")
    n_llm = sum(1 for c in cfg.values() if c.source in ("llm", "cache"))
    print(f"autoquant({args.objective}): {len(cfg)} tensors -> rvq {n_rvq}, int8 {n_int8}, "
          f"fp16 {n_fp16} ({n_llm} via LLM/cache)")
    print(f"wrote {args.out}")
    return 0
