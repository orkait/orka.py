"""Kaggle pack pipeline: download from HF, pack, optionally upload back."""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
        _kp_awq = _load_awq_activations(args)

        _kp_smap = None
        if getattr(args, "sensitivity_map", None):
            with open(args.sensitivity_map) as f:
                _kp_smap = json.load(f)

        _apply_gpu_memory_cap(args.backend, args.device, args.max_gpu_mem_gb)
        _apply_system_ram_cap(getattr(args, "max_system_ram_gb", None))

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
    "repo_id":         "Qwen/Qwen3-0.6B",
    "upload_repo":     None,
    "quant_mode":      "rvq-16-8-8",
    "codebook_mode":   "per-tensor",
    "normalization":   "awq-block-max",
    "rotation":        "orthogonal",
    "rotation_seed":   42,
    "backend":         "torch",
    "device":          "cuda",
    "max_gpu_mem_gb":  14.0,
    "sample_vectors":  1000000,
    "iterations":      12,
    "outlier_frac":    0.001,
    "group_size":      8,
    "codebook_size":   256,
    "awq_calibration": True,
    "awq_alpha":       0.5,
    "calibration_max_prompts": 128,
    "calibration_max_length":  512,
    "skip_sensitive":  True,
    "run_eval":        True,
    "eval_max_prompts": 64,
    "eval_max_length":  256,
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
        *(["--rotation-seed", str(cfg["rotation_seed"])] if cfg.get("rotation_seed") is not None else []),
        "--sample-vectors", str(cfg["sample_vectors"]),
        "--iterations",     str(cfg["iterations"]),
        "--outlier-frac",   str(cfg["outlier_frac"]),
        "--group-size",     str(cfg["group_size"]),
        "--codebook-size",  str(cfg["codebook_size"]),
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
