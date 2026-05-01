"""HF model loading + per-prompt loss computation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Sequence

from orka.reconstruct import _write_complete_safetensors_reconstruction


def _resolve_eval_model_dir(source: Path, model_dir: Path | None) -> Path:
    candidate = (
        model_dir
        if model_dir is not None
        else (source if source.is_dir() else source.parent)
    )
    if not (candidate / "config.json").exists():
        raise FileNotFoundError(
            f"eval requires a Hugging Face model directory with config.json: {candidate}"
        )
    return candidate

def _is_model_weight_sidecar(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".safetensors")
        or name.endswith(".bin")
        or name.endswith(".pt")
        or name.endswith(".pth")
        or name.endswith(".onnx")
        or name.endswith(".gguf")
        or name.endswith(".safetensors.index.json")
        or name.endswith(".bin.index.json")
    )


def _copy_hf_sidecars(source_dir: Path, target_dir: Path) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for child in sorted(source_dir.iterdir()):
        if not child.is_file() or _is_model_weight_sidecar(child):
            continue
        if child.suffix.lower() not in {".json", ".txt", ".model"}:
            continue
        shutil.copy2(child, target_dir / child.name)
        copied.append(child.name)
    if "config.json" not in copied:
        raise FileNotFoundError(f"missing config.json in model directory: {source_dir}")
    return copied


def _prepare_reconstructed_hf_dir(
    artifact_dir: Path, original_model_dir: Path, target_dir: Path,
    device: str | None = None,
) -> dict:
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(
            f"reconstructed model directory must be empty: {target_dir}"
        )
    copied = _copy_hf_sidecars(original_model_dir, target_dir)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Orka manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    reconstructed = _write_complete_safetensors_reconstruction(
        artifact_dir,
        target_dir / "model.safetensors",
        manifest,
        device=device,
    )
    return {
        "model_dir": str(target_dir),
        "copied_files": copied,
        "reconstructed": reconstructed,
    }


def _load_hf_eval_dependencies():
    try:
        import numpy  # noqa: F401
        import safetensors  # noqa: F401
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "eval requires optional dependencies: torch, transformers, numpy, and safetensors"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _hf_prompt_losses(
    model_dir: Path,
    prompts: Sequence[str],
    max_length: int,
    device: str,
    local_files_only: bool,
) -> list[dict]:
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    torch, AutoModelForCausalLM, AutoTokenizer = _load_hf_eval_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()

    rows = []
    try:
        with torch.no_grad():
            for prompt in prompts:
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                input_ids = encoded["input_ids"]
                if int(input_ids.shape[-1]) < 2:
                    continue
                model_inputs = {
                    key: value.to(device)
                    for key, value in encoded.items()
                    if key in {"input_ids", "attention_mask"}
                }
                outputs = model(**model_inputs, labels=model_inputs["input_ids"])
                rows.append(
                    {
                        "prompt": prompt,
                        "token_count": int(input_ids.shape[-1]) - 1,
                        "loss": float(outputs.loss.detach().cpu().item()),
                    }
                )
    finally:
        # Release model + tokenizer + cache to free GPU memory before next call.
        try:
            model.to("cpu")
        except Exception:
            pass
        del model, tokenizer
        if device != "cpu":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if not rows:
        raise ValueError("eval prompts produced no scored tokens")
    return rows


def _combine_eval_losses(
    original_rows: Sequence[dict], orka_rows: Sequence[dict]
) -> list[dict]:
    if len(original_rows) != len(orka_rows):
        raise ValueError("original and Orka eval row counts differ")
    rows = []
    for original, orka in zip(original_rows, orka_rows):
        if original["prompt"] != orka["prompt"]:
            raise ValueError("original and Orka prompt order differs")
        rows.append(
            {
                "prompt": original["prompt"],
                "token_count": int(original["token_count"]),
                "original_loss": float(original["loss"]),
                "orka_loss": float(orka["loss"]),
                "loss_delta": float(orka["loss"]) - float(original["loss"]),
            }
        )
    return rows


def _hf_pulse_check(
    original_model_dir: Path,
    reconstructed_model_dir: Path,
    prompts: Sequence[str],
    max_length: int,
    device: str,
    local_files_only: bool,
) -> dict:
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    torch, AutoModelForCausalLM, AutoTokenizer = _load_hf_eval_dependencies()
    import torch.nn.functional as F

    tokenizer = AutoTokenizer.from_pretrained(
        str(original_model_dir),
        local_files_only=local_files_only,
    )

    # 1. Run Original Model
    model = AutoModelForCausalLM.from_pretrained(
        str(original_model_dir),
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()

    orig_logits = []
    try:
        with torch.no_grad():
            for prompt in prompts:
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                input_ids = encoded["input_ids"]
                if int(input_ids.shape[-1]) < 2:
                    orig_logits.append(None)
                    continue
                model_inputs = {
                    key: value.to(device)
                    for key, value in encoded.items()
                    if key in {"input_ids", "attention_mask"}
                }
                outputs = model(**model_inputs)
                orig_logits.append(outputs.logits.detach().cpu())
    finally:
        try:
            model.to("cpu")
        except Exception:
            pass
        del model
        if device != "cpu":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # 2. Run Reconstructed Model
    model = AutoModelForCausalLM.from_pretrained(
        str(reconstructed_model_dir),
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()

    total_kl = 0.0
    total_tokens = 0
    top1_matches = 0

    try:
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                orig_l = orig_logits[i]
                if orig_l is None:
                    continue
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                model_inputs = {
                    key: value.to(device)
                    for key, value in encoded.items()
                    if key in {"input_ids", "attention_mask"}
                }
                outputs = model(**model_inputs)
                orka_l = outputs.logits.detach().cpu()

                # Compare distributions (KL Divergence)
                p = F.log_softmax(orka_l, dim=-1)
                q = F.softmax(orig_l, dim=-1)
                kl = F.kl_div(p, q, reduction="batchmean", log_target=False).item()

                # Compare Top-1 Agreement
                orig_top1 = orig_l.argmax(dim=-1)
                orka_top1 = orka_l.argmax(dim=-1)
                matches = (orig_top1 == orka_top1).sum().item()
                tokens = orig_top1.numel()

                total_kl += kl * tokens
                top1_matches += matches
                total_tokens += tokens
    finally:
        try:
            model.to("cpu")
        except Exception:
            pass
        del model, tokenizer
        if device != "cpu":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if total_tokens == 0:
        raise ValueError("pulse check prompts produced no scored tokens")

    return {
        "kl_divergence": total_kl / total_tokens,
        "top1_agreement": top1_matches / total_tokens,
        "total_tokens": total_tokens,
    }

