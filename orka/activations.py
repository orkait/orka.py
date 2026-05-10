"""AWQ-style activation calibration via Hugging Face forward hooks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

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
    out: dict[str, "torch.Tensor"] = {}
    for name, xs in activations.items():
        full = xs[0] if len(xs) == 1 else __import__("torch").cat(xs, dim=0)
        if full.shape[0] > max_samples_per_layer:
            import torch as _t

            idx = _t.randperm(full.shape[0])[:max_samples_per_layer]
            full = full[idx]
        out[name + ".weight"] = full.to(dtype=__import__("torch").float32)
    return out


import argparse



def _load_awq_activations(args: argparse.Namespace):
    if not args.awq_calibration:
        return None
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
