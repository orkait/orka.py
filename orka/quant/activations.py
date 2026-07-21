"""AWQ-style activation calibration via Hugging Face forward hooks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from orka._runtime import _resolve_torch_device
from orka.core._features import ensure_awq_feature_enabled
from orka.eval.prompts import _read_prompt_file


def _collect_activations_hf(
    model_dir: Path,
    prompts: Sequence[str],
    max_length: int,
    device: str,
    max_samples_per_layer: int = 4096,
) -> dict:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("activation calibration requires torch and transformers") from exc
    # Callers pass the CLI --device verbatim, which defaults to "auto"; torch only
    # accepts concrete device strings.
    device = str(_resolve_torch_device(device))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    model.to(device)
    model.eval()
    activations: dict[str, list] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            captured_name = name

            def hook(_mod, inputs, _outputs, _name=captured_name):
                x = inputs[0]
                if x.dim() > 2:
                    x = x.reshape(-1, x.shape[-1])
                activations.setdefault(_name, []).append(x.detach().cpu())

            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=max_length
            )
            ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            if attn is None:
                attn = torch.ones_like(ids)
            attn = attn.to(device)
            model(input_ids=ids, attention_mask=attn)
    for h in handles:
        h.remove()
    out: dict[str, torch.Tensor] = {}
    for name, xs in activations.items():
        full = xs[0] if len(xs) == 1 else __import__("torch").cat(xs, dim=0)
        if full.shape[0] > max_samples_per_layer:
            import torch as _t

            idx = _t.randperm(full.shape[0])[:max_samples_per_layer]
            full = full[idx]
        out[name + ".weight"] = full.to(dtype=__import__("torch").float32)
    return out





def _load_awq_activations(args: argparse.Namespace):
    # Gate only the legacy AWQ normalization modes. Activations used purely for
    # Hessian-proxy importance weighting are always allowed.
    if (
        getattr(args, "awq_activations_file", None)
        or getattr(args, "awq_calibration", None)
    ) and getattr(args, "normalization", "none") in {"awq", "awq-block-max"}:
        ensure_awq_feature_enabled()

    if getattr(args, "awq_activations_file", None):
        import json

        import torch
        path = Path(args.awq_activations_file)
        if not path.exists():
            raise FileNotFoundError(f"AWQ activations file not found: {path}")
        print(f"Loading pre-calculated AWQ activations from {path}...", flush=True)
        try:
            with open(path) as f:
                raw = json.load(f)
            # JSON format often contains lists; convert back to tensors for normalization module
            return {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items()}
        except Exception:
            # Fallback to torch.load for binary .pt files
            return torch.load(str(path), map_location="cpu")

    if args.awq_calibration:
        prompts = _read_prompt_file(
            Path(args.awq_calibration), max_prompts=args.calibration_max_prompts
        )
        model_dir = (
            Path(args.awq_model_dir) if args.awq_model_dir else Path(args.source).parent
        )
        return _collect_activations_hf(
            model_dir,
            prompts,
            max_length=args.calibration_max_length,
            device=args.device if args.backend == "torch" else "cpu",
            max_samples_per_layer=args.calibration_max_samples,
        )

    # DEFAULT: Hessian-weighting via the bundled calibration corpus, so packs are
    # weighted out of the box. Measured free quality win (same artifact bytes):
    # SmolLM2-135M rvq-12-12 perplexity ratio 2.19x -> 1.63x; helps at 50M/135M/1.1B.
    # Opt out with --no-hessian.
    return _auto_hessian_activations(args)


def _bundled_calibration_path() -> Path:
    return Path(__file__).parent.parent / "data" / "calibration.txt"


def _auto_hessian_activations(args: argparse.Namespace):
    """Collect Hessian-proxy activations from the bundled calibration corpus.

    Falls back to unweighted packing (with a loud warning) when activations
    cannot be collected - non-torch backend, missing model config/tokenizer, or
    a collection failure. This keeps bare ``orka pack file.safetensors`` working
    while making the quality-preserving path the default for real model dirs.
    """
    if getattr(args, "no_hessian", False):
        return None

    src = Path(args.source)
    model_dir = (
        Path(args.awq_model_dir)
        if getattr(args, "awq_model_dir", None)
        else (src if src.is_dir() else src.parent)
    )
    reason = None
    if getattr(args, "backend", "torch") != "torch":
        reason = f"--backend {args.backend} (Hessian-weighting needs torch)"
    elif not (model_dir / "config.json").exists():
        reason = f"no config.json in {model_dir} (cannot load model for calibration)"
    elif not _bundled_calibration_path().exists():
        reason = "bundled calibration corpus missing"

    if reason is None:
        try:
            import torch  # noqa: F401

            n_prompts = getattr(args, "calibration_max_prompts", 32) or 32
            prompts = _read_prompt_file(_bundled_calibration_path(), max_prompts=n_prompts)
            print(
                f"Hessian-weighting: collecting activations from bundled calibration "
                f"({len(prompts)} prompts) - pass --no-hessian to skip.",
                flush=True,
            )
            acts = _collect_activations_hf(
                model_dir,
                prompts,
                max_length=getattr(args, "calibration_max_length", 256),
                device=args.device,
                max_samples_per_layer=getattr(args, "calibration_max_samples", 4096),
            )
            print(
                f"Hessian-weighting: ENABLED (default) over {len(acts)} tensors.",
                flush=True,
            )
            return acts
        except Exception as exc:  # noqa: BLE001 - any failure must degrade gracefully
            reason = f"activation collection failed: {type(exc).__name__}: {exc}"

    print("=" * 72, flush=True)
    print("WARNING: packing UNWEIGHTED (no Hessian-weighting).", flush=True)
    print(f"  reason: {reason}", flush=True)
    print("  Measured cost: ~15-60% larger perplexity gap vs Hessian-weighted", flush=True)
    print("  (SmolLM2-135M rvq-12-12: 2.19x -> 1.63x ratio).", flush=True)
    print("  Enable: pass --awq-model-dir <hf_dir>, or pack from a model dir that", flush=True)
    print("  has config.json + tokenizer. Silence this: --no-hessian.", flush=True)
    print("=" * 72, flush=True)
    return None
