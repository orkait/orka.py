"""HF prompt-loss / perplexity evaluation."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Callable, Sequence

from orka._format import ORKA_VERSION
from orka._util import _best_run, _require_non_empty, _safe_exp, _safe_tensor_name
from orka.eval.hf import (
    _combine_eval_losses,
    _hf_prompt_losses,
    _load_hf_eval_dependencies,
    _prepare_reconstructed_hf_dir,
    _resolve_eval_model_dir,
)
from orka.eval.prompts import _read_prompt_file


def _summarize_eval_rows(rows: Sequence[dict]) -> dict:
    _require_non_empty("eval rows", rows)
    token_count = sum(int(row["token_count"]) for row in rows)
    if token_count <= 0:
        raise ValueError("eval rows must contain at least one scored token")
    original_loss = (
        sum(float(row["original_loss"]) * int(row["token_count"]) for row in rows)
        / token_count
    )
    orka_loss = (
        sum(float(row["orka_loss"]) * int(row["token_count"]) for row in rows)
        / token_count
    )
    original_perplexity = _safe_exp(original_loss)
    orka_perplexity = _safe_exp(orka_loss)
    if original_perplexity and math.isfinite(original_perplexity):
        perplexity_ratio = orka_perplexity / original_perplexity
    else:
        perplexity_ratio = float("inf")
    return {
        "prompt_count": len(rows),
        "token_count": token_count,
        "original_loss": original_loss,
        "orka_loss": orka_loss,
        "loss_delta": orka_loss - original_loss,
        "original_perplexity": original_perplexity,
        "orka_perplexity": orka_perplexity,
        "perplexity_ratio": perplexity_ratio,
    }

def eval_artifact(
    artifact_dir: Path,
    prompts_path: Path,
    out_path: Path,
    model_dir: Path | None = None,
    max_prompts: int | None = None,
    max_length: int = 512,
    device: str = "cpu",
    reconstructed_model_dir: Path | None = None,
    local_files_only: bool = True,
) -> dict:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Orka manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest["source"])
    original_model_dir = _resolve_eval_model_dir(source, model_dir)
    prompts = _read_prompt_file(prompts_path, max_prompts=max_prompts)
    _load_hf_eval_dependencies()

    def run_with_reconstructed_dir(target_dir: Path) -> dict:
        prepared = _prepare_reconstructed_hf_dir(
            artifact_dir, original_model_dir, target_dir, device=device
        )
        original_rows = _hf_prompt_losses(
            original_model_dir,
            prompts,
            max_length=max_length,
            device=device,
            local_files_only=local_files_only,
        )
        orka_rows = _hf_prompt_losses(
            target_dir,
            prompts,
            max_length=max_length,
            device=device,
            local_files_only=local_files_only,
        )
        rows = _combine_eval_losses(original_rows, orka_rows)
        summary = _summarize_eval_rows(rows)
        result = {
            "format": "orka-eval",
            "version": ORKA_VERSION,
            "artifact": str(artifact_dir),
            "source": str(source),
            "model_dir": str(original_model_dir),
            "prompts": str(prompts_path),
            "max_length": max_length,
            "device": device,
            "local_files_only": local_files_only,
            "reconstructed_model_dir": str(target_dir),
            "prepared": prepared,
            **summary,
            "rows": rows,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")

        return result

    if reconstructed_model_dir is not None:
        reconstructed_model_dir.mkdir(parents=True, exist_ok=True)
        return run_with_reconstructed_dir(reconstructed_model_dir)

    with tempfile.TemporaryDirectory() as tmp:
        return run_with_reconstructed_dir(Path(tmp) / "reconstructed-model")


def _eval_sweep_output_root(out_path: Path) -> Path:
    if out_path.suffix:
        return out_path.with_name(f"{out_path.stem}.evals")
    return out_path.parent / f"{out_path.name}.evals"


def _eval_run_summary(result: dict) -> dict:
    keys = [
        "prompt_count",
        "token_count",
        "original_loss",
        "orka_loss",
        "loss_delta",
        "original_perplexity",
        "orka_perplexity",
        "perplexity_ratio",
    ]
    return {key: result[key] for key in keys if key in result}


def eval_sweep(
    sweep_path: Path,
    prompts_path: Path,
    out_path: Path,
    model_dir: Path | None = None,
    max_prompts: int | None = None,
    max_length: int = 512,
    device: str = "cpu",
    local_files_only: bool = True,
    max_runs: int | None = None,
    reconstructed_model_root: Path | None = None,
    evaluator: Callable[..., dict] = eval_artifact,
) -> dict:
    if max_runs is not None and max_runs <= 0:
        raise ValueError("max_runs must be positive")

    sweep = json.loads(sweep_path.read_text())
    runs = list(sweep.get("runs", []))
    _require_non_empty("sweep runs", runs)
    if max_runs is not None:
        runs = runs[:max_runs]

    eval_root = _eval_sweep_output_root(out_path)
    eval_root.mkdir(parents=True, exist_ok=True)
    if reconstructed_model_root is not None:
        reconstructed_model_root.mkdir(parents=True, exist_ok=True)

    evaluated_runs = []
    for run_i, run in enumerate(runs):
        artifact_dir = Path(run["artifact"])
        run_name = f"{run_i:04d}-{_safe_tensor_name(artifact_dir.name)}"
        eval_path = eval_root / f"{run_name}.eval.json"
        reconstructed_model_dir = (
            reconstructed_model_root / run_name
            if reconstructed_model_root is not None
            else None
        )
        eval_result = evaluator(
            artifact_dir=artifact_dir,
            prompts_path=prompts_path,
            out_path=eval_path,
            model_dir=model_dir,
            max_prompts=max_prompts,
            max_length=max_length,
            device=device,
            reconstructed_model_dir=reconstructed_model_dir,
            local_files_only=local_files_only,
        )
        eval_summary = _eval_run_summary(eval_result)
        combined = dict(run)
        combined["eval_path"] = str(eval_path)
        combined["eval"] = eval_summary
        combined.update(eval_summary)
        evaluated_runs.append(combined)

    summary = {
        "format": "orka-eval-sweep",
        "version": ORKA_VERSION,
        "source_sweep": str(sweep_path),
        "sweep_source": sweep.get("source"),
        "prompts": str(prompts_path),
        "model_dir": str(model_dir) if model_dir is not None else None,
        "max_prompts": max_prompts,
        "max_length": max_length,
        "device": device,
        "local_files_only": local_files_only,
        "input_run_count": len(sweep.get("runs", [])),
        "run_count": len(evaluated_runs),
        "eval_root": str(eval_root),
        "reconstructed_model_root": (
            str(reconstructed_model_root)
            if reconstructed_model_root is not None
            else None
        ),
        "best_by_loss_delta": _best_run(evaluated_runs, "loss_delta", reverse=False),
        "best_by_perplexity_ratio": _best_run(
            evaluated_runs, "perplexity_ratio", reverse=False
        ),
        "best_by_artifact_bytes": _best_run(
            evaluated_runs, "artifact_bytes", reverse=False
        ),
        "runs": evaluated_runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    return summary
