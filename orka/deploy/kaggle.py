"""Kaggle pack pipeline: download from HF, pack, optionally upload back."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from orka._runtime import _apply_gpu_memory_cap, _apply_system_ram_cap, _stop_ram_monitor
from orka._util import _human_bytes
from orka.activations import _load_awq_activations
from orka.eval import eval_artifact
from orka.pipeline.pack import pack_checkpoint
from orka.quant import (
    _resolve_quant_stages,
    is_rvq_mixed_spec,
    rvq_mixed_family_stages,
)
from orka.merge import merge_orka_artifacts
from orka.report import report_artifact


def _load_hf_token() -> str | None:
    for candidate in (
        Path("/kaggle/input/hf-token-private/hf_token.txt"),
        Path("/kaggle/input/hf-token/hf_token.txt"),
    ):
        if candidate.exists():
            tok = candidate.read_text().strip()
            if tok:
                return tok
    if Path("/kaggle/input").exists():
        for name in ("hf_token.txt", "HF_TOKEN", "token"):
            hits = list(Path("/kaggle/input").rglob(name))
            if hits:
                tok = hits[0].read_text().strip()
                if tok:
                    return tok
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        for secret in ("HF_TOKEN", "huggingface_token", "HF_HUB_TOKEN"):
            try:
                tok = client.get_secret(secret)
                if tok:
                    return tok
            except Exception:
                pass
    except ImportError:
        pass
    return os.environ.get("HF_TOKEN")


def _hf_snapshot_with_retry(
    repo_id: str,
    local_dir: Path,
    token: str | None,
    allow_patterns,
    max_retries: int = 3,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub required: pip install huggingface_hub") from exc
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                token=token,
                allow_patterns=allow_patterns,
            )
            return
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = 5 * attempt
                print(f"Download attempt {attempt} failed ({exc}); retry in {delay}s...", flush=True)
                time.sleep(delay)
    raise RuntimeError(f"Download failed after {max_retries} attempts") from last_exc


def _hf_upload_with_retry(
    api,
    folder_path: str,
    repo_id: str,
    max_retries: int = 3,
) -> None:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_folder(folder_path=folder_path, repo_id=repo_id, repo_type="model")
            return
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = 10 * attempt
                print(f"Upload attempt {attempt} failed ({exc}); retry in {delay}s...", flush=True)
                time.sleep(delay)
    raise RuntimeError(f"Upload failed after {max_retries} attempts") from last_exc


def _append_arg(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _visible_cuda_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [item.strip() for item in visible.split(",") if item.strip()]
    try:
        import torch
        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def _build_partition_pack_cmd(
    args: argparse.Namespace,
    source_file: Path,
    part_dir: Path,
    part_index: int,
    part_count: int,
    device: str,
    sensitivity_map: Path | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "orka",
        "pack",
        str(source_file),
        "--out",
        str(part_dir),
        "--group-size",
        str(args.group_size),
        "--codebook-size",
        str(args.codebook_size),
        "--codebook-mode",
        args.codebook_mode,
        "--backend",
        args.backend,
        "--device",
        device,
        "--normalization",
        args.normalization,
        "--block-scale-size",
        str(args.block_scale_size),
        "--rotation",
        args.rotation,
        "--iterations",
        str(args.iterations),
        "--outlier-frac",
        str(args.outlier_frac),
        "--awq-alpha",
        str(args.awq_alpha),
        "--em-aq-passes",
        str(getattr(args, "em_aq_passes", 3)),
        "--tensor-partition-count",
        str(part_count),
        "--tensor-partition-index",
        str(part_index),
    ]
    _append_arg(cmd, "--quant-mode", args.quant_mode)
    if getattr(args, "codebook_sizes", None):
        cmd.append("--codebook-sizes")
        cmd.extend(str(value) for value in args.codebook_sizes)
    _append_arg(cmd, "--rotation-seed", args.rotation_seed)
    _append_arg(cmd, "--sample-vectors", args.sample_vectors)
    _append_arg(cmd, "--max-values-per-tensor", args.max_values_per_tensor)
    _append_arg(cmd, "--max-tensors", args.max_tensors)
    _append_arg(cmd, "--max-gpu-mem-gb", args.max_gpu_mem_gb)
    _append_arg(cmd, "--max-system-ram-gb", getattr(args, "max_system_ram_gb", None))
    _append_arg(cmd, "--workload-budget-gb", getattr(args, "workload_budget_gb", None))
    _append_arg(cmd, "--max-cpu-threads", getattr(args, "max_cpu_threads", None))
    _append_arg(cmd, "--awq-calibration", args.awq_calibration)
    _append_arg(cmd, "--awq-model-dir", args.awq_model_dir)
    _append_arg(cmd, "--awq-activations-file", getattr(args, "awq_activations_file", None))
    _append_arg(cmd, "--calibration-max-prompts", args.calibration_max_prompts)
    _append_arg(cmd, "--calibration-max-length", args.calibration_max_length)
    _append_arg(cmd, "--calibration-max-samples", args.calibration_max_samples)
    if getattr(args, "progress_file", None):
        progress = Path(args.progress_file)
        child_progress = progress.with_name(f"{progress.stem}.part-{part_index}{progress.suffix}")
        cmd.extend(["--progress-file", str(child_progress)])
    if sensitivity_map is not None:
        cmd.extend(["--sensitivity-map", str(sensitivity_map)])
    if getattr(args, "only_tensors", None):
        cmd.append("--only-tensors")
        cmd.extend(args.only_tensors)
    if getattr(args, "codebook_cache", None):
        cache_dir = Path(args.codebook_cache) / f"part-{part_index}"
        cmd.extend(["--codebook-cache", str(cache_dir)])
    if not getattr(args, "slrq_salient", True):
        cmd.append("--no-slrq-salient")
    return cmd


def _run_partitioned_pack(
    args: argparse.Namespace,
    source_file: Path,
    out_dir: Path,
    sensitivity_map_data: dict | None,
) -> dict:
    part_count = int(args.tensor_partition_count)
    if part_count < 2:
        raise ValueError("partitioned Kaggle pack requires at least two partitions")
    if args.backend != "torch" or not str(args.device).startswith("cuda"):
        raise ValueError("automatic partition workers require --backend torch --device cuda")

    worker_count = int(getattr(args, "partition_worker_count", 1) or 1)
    if worker_count < 1:
        raise ValueError("partition_worker_count must be >= 1")
    worker_count = min(worker_count, part_count)
    if worker_count > 1 and getattr(args, "max_system_ram_gb", None) is not None:
        print(
            "WARNING: --max-system-ram-gb is per child process, not a notebook-wide RAM cap. "
            "Use --partition-worker-count 1 for memory-first Kaggle runs.",
            flush=True,
        )

    cuda_ids = _visible_cuda_ids()
    if len(cuda_ids) < part_count:
        raise RuntimeError(
            f"requested {part_count} partitions but only {len(cuda_ids)} CUDA devices are visible"
        )

    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"output directory already exists with content: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    sensitivity_map = None
    if getattr(args, "sensitivity_map", None):
        sensitivity_map = Path(args.sensitivity_map)
    elif sensitivity_map_data is not None:
        sensitivity_map = out_dir.parent / "orka_auto_sensitivity_map.json"
        sensitivity_map.write_text(json.dumps(sensitivity_map_data, indent=2) + "\n")

    part_dirs = [out_dir.parent / f"{out_dir.name}.part-{i}" for i in range(part_count)]
    for part_dir in part_dirs:
        if part_dir.exists():
            shutil.rmtree(part_dir)

    failed: list[int] = []
    print(
        f"--- Launching {part_count} Kaggle partitions with {worker_count} concurrent worker(s) ---",
        flush=True,
    )
    # Child workers run `python -m orka pack` in a fresh interpreter. When orka
    # is loaded from a sys.path entry (e.g. a Kaggle dataset mount) rather than
    # pip-installed, the child cannot import it. Propagate orka's parent dir via
    # PYTHONPATH so the partition subprocesses resolve the package.
    import orka as _orka_pkg

    orka_parent = str(Path(_orka_pkg.__file__).resolve().parent.parent)

    for start in range(0, part_count, worker_count):
        batch = list(range(start, min(start + worker_count, part_count)))
        procs: list[tuple[int, subprocess.Popen]] = []
        for offset, i in enumerate(batch):
            part_dir = part_dirs[i]
            env = os.environ.copy()
            gpu_id = cuda_ids[i % len(cuda_ids)]
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["PYTHONPATH"] = orka_parent + os.pathsep + env.get("PYTHONPATH", "")
            cmd = _build_partition_pack_cmd(
                args=args,
                source_file=source_file,
                part_dir=part_dir,
                part_index=i,
                part_count=part_count,
                device="cuda:0",
                sensitivity_map=sensitivity_map,
            )
            print(
                f"Partition {i}/{part_count - 1}: physical GPU {gpu_id} -> logical cuda:0",
                flush=True,
            )
            procs.append((i, subprocess.Popen(cmd, env=env)))

        for i, proc in procs:
            code = proc.wait()
            if code != 0:
                failed.append(i)
        if failed:
            break
    if failed:
        raise RuntimeError(f"partition workers failed: {failed}")

    print("--- Merging partition artifacts ---", flush=True)
    return merge_orka_artifacts(part_dirs, out_dir)


def _write_artifact_tarball(out_dir: Path, on_kaggle: bool) -> Path:
    tar_path = (
        Path("/kaggle/working") / f"{out_dir.name}.tar.gz"
        if on_kaggle
        else out_dir.parent / f"{out_dir.name}.tar.gz"
    )
    if tar_path.exists():
        tar_path.unlink()
    print(f"--- Writing artifact tarball: {tar_path} ---", flush=True)
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    return tar_path


def cmd_kaggle_pack(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub required. Run: pip install huggingface_hub", file=os.sys.stderr)
        return 1

    token = _load_hf_token()
    if not token:
        print(
            "Error: HF token not found. Attach hf-token-private Kaggle dataset, "
            "add a Kaggle Secret named HF_TOKEN, or set the HF_TOKEN env var.",
            file=os.sys.stderr,
        )
        return 1

    on_kaggle = Path("/kaggle/working").exists()

    if args.out:
        out_dir = Path(args.out)
    elif on_kaggle:
        slug = args.repo_id.split("/")[-1]
        out_dir = Path("/kaggle/working") / f"{slug}.orka"
    else:
        print("Error: --out required when not running on Kaggle.", file=os.sys.stderr)
        return 1

    src_dir = (Path("/kaggle/tmp") / "orka_src_model") if on_kaggle else (
        Path(tempfile.mkdtemp()) / "orka_src_model"
    )
    src_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"--- Downloading {args.repo_id} ---", flush=True)
        _hf_snapshot_with_retry(
            repo_id=args.repo_id,
            local_dir=src_dir,
            token=token,
            allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"],
        )

        source_file = next(src_dir.glob("*.safetensors"), None)
        if not source_file:
            print(f"Error: no .safetensors found in {args.repo_id}", file=os.sys.stderr)
            return 1

        print(f"--- Packing {source_file.name} ---", flush=True)

        if is_rvq_mixed_spec(args.quant_mode):
            _kp_family_map = rvq_mixed_family_stages()
            _kp_sizes = [_kp_family_map["other"][0]]
            _kp_codebook_mode = "per-tensor"
        else:
            _kp_family_map = None
            _kp_sizes = _resolve_quant_stages(
                args.quant_mode,
                getattr(args, "codebook_sizes", None),
                args.codebook_size,
            )
            _kp_codebook_mode = args.codebook_mode

        if args.awq_calibration:
            args.awq_model_dir = str(src_dir)
        partition_parent = (
            args.tensor_partition_count is not None
            and args.tensor_partition_count > 1
            and args.tensor_partition_index is None
        )
        _kp_awq = None if partition_parent else _load_awq_activations(args)

        _kp_smap = None
        if getattr(args, "sensitivity_map", None):
            with open(args.sensitivity_map) as f:
                _kp_smap = json.load(f)
        elif not getattr(args, "skip_sensitive", False) and on_kaggle:
            try:
                print("--- Auto-generating Pillar Map (Frequency + Magnitude) ---", flush=True)
                import torch
                from transformers import AutoTokenizer
                import numpy as np
                from scipy.stats import rankdata
                from collections import Counter
                from safetensors import safe_open
                
                tok = AutoTokenizer.from_pretrained(src_dir, trust_remote_code=True)
                calib_path = (
                    Path(args.eval_prompts)
                    if getattr(args, "eval_prompts", None)
                    else (Path(args.awq_calibration) if args.awq_calibration else None)
                )
                if calib_path is None or not calib_path.exists():
                    raise RuntimeError("no calibration/eval prompts available")
                with open(calib_path) as f:
                    text = f.read()
                counts = Counter(tok.encode(text))
                
                emb = None
                # framework="pt" so bf16/fp16 embeddings load (numpy cannot represent bfloat16);
                # cast to float32 numpy for the norm computation below.
                with safe_open(str(source_file), framework="pt") as f:
                    for k in f.keys():
                        if "embed_tokens" in k or "wte" in k or "word_embeddings" in k:
                            emb = f.get_tensor(k).to(torch.float32).cpu().numpy()
                            break
                if emb is None:
                    raise RuntimeError("Could not find embedding tensor in safetensors file")
                    
                actual_vocab = emb.shape[0]
                norms = np.linalg.norm(emb, axis=1)
                
                f_arr = np.zeros(actual_vocab)
                for tid, c in counts.items():
                    if tid < actual_vocab: f_arr[tid] = c
                
                f_rank = rankdata(f_arr) / actual_vocab
                n_rank = rankdata(norms) / actual_vocab
                score = (f_rank * 0.5) + (n_rank * 0.5)
                
                top_count = int(actual_vocab * 0.10)
                top_ids = np.argsort(score)[::-1][:top_count].tolist()
                _kp_smap = {"top_tokens": top_ids, "layers": []}
                print(f"Kaggle: protected {len(top_ids)} pillars", flush=True)
                
                del emb
                import gc
                gc.collect()
            except Exception as exc:
                print(f"WARNING: Auto-pillar failed ({exc}); proceeding without pillars", flush=True)

        if partition_parent:
            manifest = _run_partitioned_pack(args, source_file, out_dir, _kp_smap)
        else:
            _apply_gpu_memory_cap(args.backend, args.device, args.max_gpu_mem_gb)
            _apply_system_ram_cap(
                getattr(args, "max_system_ram_gb", None),
                getattr(args, "workload_budget_gb", None),
            )

            manifest = pack_checkpoint(
                source=source_file,
                out_dir=out_dir,
                group_size=args.group_size,
                codebook_size=_kp_sizes[0],
                codebook_sizes=_kp_sizes if _kp_family_map is None else None,
                family_stages_map=_kp_family_map,
                codebook_mode=_kp_codebook_mode,
                backend=args.backend,
                device=args.device,
                normalization=args.normalization,
                block_scale_size=args.block_scale_size,
                rotation=args.rotation,
                rotation_seed=args.rotation_seed,
                sample_vectors=args.sample_vectors,
                iterations=args.iterations,
                max_values_per_tensor=args.max_values_per_tensor,
                outlier_frac=args.outlier_frac,
                awq_activations=_kp_awq,
                awq_alpha=args.awq_alpha,
                progress_file=Path(args.progress_file) if args.progress_file else None,
                sensitivity_map=_kp_smap,
                max_tensors=args.max_tensors,
                em_aq_passes=getattr(args, "em_aq_passes", 3),
                slrq_salient=getattr(args, "slrq_salient", True),
                codebook_cache_dir=Path(args.codebook_cache) if getattr(args, "codebook_cache", None) else None,
                tensor_partition_count=args.tensor_partition_count,
                tensor_partition_index=args.tensor_partition_index,
            )

        artifact_report = report_artifact(out_dir)
        pack_report = {
            "source_repo": args.repo_id,
            "upload_repo": args.upload_repo,
            "artifact": str(out_dir),
            "tensor_count": manifest["tensor_count"],
            "group_size": args.group_size,
            "codebook_mode": _kp_codebook_mode,
            "normalization": args.normalization,
            "artifact_bytes": artifact_report["artifact_bytes"],
            "artifact_size": _human_bytes(artifact_report["artifact_bytes"]),
            "original_fp16_bytes": artifact_report["original_fp16_bytes"],
            "compression_ratio_fp16_to_artifact": artifact_report[
                "compression_ratio_fp16_to_artifact"
            ],
            "weighted_mse": artifact_report["weighted_mse"],
            "relative_rmse": artifact_report["relative_rmse"],
            "cosine_similarity": artifact_report["cosine_similarity"],
        }
        report_path = (
            Path("/kaggle/working/pack_report.json") if on_kaggle
            else out_dir.parent / "pack_report.json"
        )
        report_path.write_text(json.dumps(pack_report, indent=2) + "\n")
        print(f"Pack report written to {report_path}", flush=True)

        tar_path = _write_artifact_tarball(out_dir, on_kaggle)
        pack_report["artifact_tarball"] = str(tar_path)
        pack_report["artifact_tarball_bytes"] = tar_path.stat().st_size
        pack_report["artifact_tarball_size"] = _human_bytes(tar_path.stat().st_size)
        report_path.write_text(json.dumps(pack_report, indent=2) + "\n")

        if getattr(args, "run_eval", False):
            print("--- Running perplexity eval ---", flush=True)
            eval_prompts = (
                Path(args.eval_prompts) if args.eval_prompts
                else (Path(args.awq_calibration) if args.awq_calibration else None)
            )
            if eval_prompts is None or not eval_prompts.exists():
                print("WARNING: no eval prompts file; skipping eval", flush=True)
            else:
                eval_out = (
                    Path("/kaggle/working/eval_report.json") if on_kaggle
                    else out_dir.parent / "eval_report.json"
                )
                try:
                    eval_result = eval_artifact(
                        artifact_dir=out_dir,
                        prompts_path=eval_prompts,
                        out_path=eval_out,
                        model_dir=src_dir,
                        max_prompts=args.eval_max_prompts,
                        max_length=args.eval_max_length,
                        device=args.device if args.backend == "torch" else "cpu",
                        local_files_only=True,
                    )
                    pack_report["eval"] = {
                        "prompt_count": eval_result["prompt_count"],
                        "token_count": eval_result["token_count"],
                        "original_loss": eval_result["original_loss"],
                        "orka_loss": eval_result["orka_loss"],
                        "loss_delta": eval_result["loss_delta"],
                        "original_perplexity": eval_result["original_perplexity"],
                        "orka_perplexity": eval_result["orka_perplexity"],
                        "perplexity_ratio": eval_result["perplexity_ratio"],
                    }
                    report_path.write_text(json.dumps(pack_report, indent=2) + "\n")
                    print(f"Eval report written to {eval_out}", flush=True)
                except Exception as exc:
                    print(f"Eval failed: {exc}", flush=True)
                    pack_report["eval_error"] = str(exc)
                    report_path.write_text(json.dumps(pack_report, indent=2) + "\n")

        print("--- Cleaning up source model to free disk space ---", flush=True)
        shutil.rmtree(str(src_dir), ignore_errors=True)

        if args.upload_repo:
            print(f"--- Uploading to {args.upload_repo} ---", flush=True)
            api = HfApi(token=token)
            api.create_repo(args.upload_repo, repo_type="model", exist_ok=True)
            _hf_upload_with_retry(api, str(out_dir), args.upload_repo)
            print(f"Uploaded to {args.upload_repo}", flush=True)

        print(json.dumps(pack_report, indent=2))
        return 0

    finally:
        _stop_ram_monitor()
        if src_dir.exists():
            shutil.rmtree(str(src_dir), ignore_errors=True)


_KAGGLE_CONFIG = {
    "repo_id":         "MerlinSafety/HybridIntelligence-0.5B",
    "upload_repo":     None,
    "quant_mode":      "rvq-mixed",
    "codebook_mode":   "per-tensor",
    "normalization":   "slrq-block",
    "rotation":        "orthogonal",
    "rotation_seed":   42,
    "backend":         "torch",
    "device":          "cuda",
    # 0.5B model (1 GB) fits on a single T4 - no partitioning needed.
    "max_gpu_mem_gb":  14.0,
    "max_system_ram_gb": 28.0,
    "workload_budget_gb": 20.0,
    "max_cpu_threads": 2,
    "sample_vectors":  65536,
    "iterations":      8,
    "outlier_frac":    0.005,
    "group_size":      8,
    "codebook_size":   256,
    "awq_calibration": False,
    "awq_alpha":       0.5,
    "calibration_max_prompts": 32,
    "calibration_max_length":  256,
    "skip_sensitive":  False,
    "run_eval":        True,
    "eval_max_prompts": 50,
    "eval_max_length":  128,
    "em_aq_passes":    3,
    "tensor_partition_count": 1,
    "tensor_partition_index": None,
    "partition_worker_count": 1,
}


_FALLBACK_PROMPTS = [
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
]


def bootstrap_argv(argv: list) -> None:
    """Inject kaggle-pack args from _KAGGLE_CONFIG when invoked on Kaggle with no args.

    Mutates argv in place. No-op if argv already has subcommand.
    """
    if len(argv) != 1:
        return
    print("Kaggle: building args from _KAGGLE_CONFIG", flush=True)
    cfg = _KAGGLE_CONFIG

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
                if len(text) >= 200:
                    samples.append(text)
                    if len(samples) >= target:
                        break
            if not samples:
                raise RuntimeError("no usable Wikitext samples")
            calib_path.write_text("\n".join(samples))
            print(f"Kaggle: wrote {len(samples)} Wikitext samples to {calib_path}", flush=True)
        except Exception as exc:
            print(f"Kaggle: Wikitext fetch failed ({exc}); falling back to inline prompts", flush=True)
            calib_path.write_text("\n".join(_FALLBACK_PROMPTS))

    smap_path = Path("/tmp/orka_sensitivity_map.json")
    if cfg.get("skip_sensitive"):
        smap_path.write_text(json.dumps({
            "base_loss": 0.0,
            "layers": [
                {"layer": "lm_head", "loss_delta": 999.0, "sensitivity": "high"},
                {"layer": "model.embed_tokens", "loss_delta": 999.0, "sensitivity": "high"},
            ],
        }))
        print(f"Kaggle: wrote sensitivity stub to {smap_path}", flush=True)

    argv += [
        "kaggle-pack",
        "--repo-id",        cfg["repo_id"],
        "--quant-mode",     cfg["quant_mode"],
        "--codebook-mode",  cfg["codebook_mode"],
        "--normalization",  cfg["normalization"],
        "--rotation",       cfg["rotation"],
        "--backend",        cfg["backend"],
        "--device",         cfg["device"],
        *(["--max-gpu-mem-gb", str(cfg["max_gpu_mem_gb"])] if cfg.get("max_gpu_mem_gb") is not None else []),
        *(["--max-system-ram-gb", str(cfg["max_system_ram_gb"])] if cfg.get("max_system_ram_gb") is not None else []),
        *(["--workload-budget-gb", str(cfg["workload_budget_gb"])] if cfg.get("workload_budget_gb") is not None else []),
        *(["--max-cpu-threads", str(cfg["max_cpu_threads"])] if cfg.get("max_cpu_threads") is not None else []),
        *(["--rotation-seed", str(cfg["rotation_seed"])] if cfg.get("rotation_seed") is not None else []),
        "--sample-vectors", str(cfg["sample_vectors"]),
        "--iterations",     str(cfg["iterations"]),
        "--outlier-frac",   str(cfg["outlier_frac"]),
        "--group-size",     str(cfg["group_size"]),
        "--codebook-size",  str(cfg["codebook_size"]),
        "--em-aq-passes",   str(cfg.get("em_aq_passes", 3)),
    ]
    if cfg.get("tensor_partition_count") is not None:
        argv += [
            "--tensor-partition-count",
            str(int(cfg["tensor_partition_count"])),
        ]
        if cfg.get("partition_worker_count") is not None:
            argv += [
                "--partition-worker-count",
                str(int(cfg["partition_worker_count"])),
            ]
        if cfg.get("tensor_partition_index") is not None:
            argv += [
                "--tensor-partition-index",
                str(int(cfg["tensor_partition_index"])),
            ]
    if cfg.get("awq_calibration"):
        argv += [
            "--awq-calibration", str(calib_path),
            "--awq-alpha",       str(cfg["awq_alpha"]),
            "--calibration-max-prompts", str(cfg.get("calibration_max_prompts", 32)),
            "--calibration-max-length",  str(cfg.get("calibration_max_length", 256)),
        ]
    if cfg.get("skip_sensitive"):
        argv += ["--sensitivity-map", str(smap_path)]
    if cfg.get("run_eval"):
        if not calib_path.exists():
            calib_path.write_text("\n".join(_FALLBACK_PROMPTS))
        argv += [
            "--run-eval",
            "--eval-prompts",     str(calib_path),
            "--eval-max-prompts", str(cfg["eval_max_prompts"]),
            "--eval-max-length",  str(cfg["eval_max_length"]),
        ]
    if cfg.get("upload_repo"):
        argv += ["--upload-repo", cfg["upload_repo"]]
