"""Sequential calibration-propagated packing (GPTQ-style error propagation).

Blocks are packed in forward order. Before each block, calibration prompts run
through the LIVE model - whose earlier blocks already carry quantized weights -
so the captured activations reflect the inputs the quantized network actually
produces. Later blocks therefore calibrate against (and absorb) the
accumulated quantization error of earlier blocks instead of compounding it.

The original checkpoint stays the pack source; only the calibration
activations come from the partially quantized model.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping, Set
from pathlib import Path

from orka.core._checkpoint import inspect_checkpoint
from orka.core._util import _report_progress
from orka.quant.activations import _align_activation_keys

_LAYER_RE = re.compile(r"\.(\d+)\.")
_FINAL_BLOCK = 1 << 30


def _block_key(name: str) -> int:
    """Forward-order block index for a tensor name.

    Embeddings and unindexed tensors come first (-1); lm_head / embed_out
    consume the final hidden state, so they pack last.
    """
    lowered = name.lower()
    if "lm_head" in lowered or "embed_out" in lowered:
        return _FINAL_BLOCK
    match = _LAYER_RE.search(name)
    if match:
        return int(match.group(1))
    return -1


def _group_tensors_by_block(names: list[str]) -> list[list[str]]:
    blocks: dict[int, list[str]] = {}
    for name in names:
        blocks.setdefault(_block_key(name), []).append(name)
    return [blocks[key] for key in sorted(blocks)]


def _checkpoint_param_names(model, checkpoint_names: Set[str]) -> dict[str, str]:
    """Checkpoint tensor name -> live model parameter name.

    The calibration model can be a text-only class over a wrapped checkpoint, so its
    parameter names differ from the checkpoint's by a prefix.
    """
    params = {name: name for name, _ in model.named_parameters()}
    return _align_activation_keys(params, checkpoint_names)


def _load_packed_into_model(model, param_for: Mapping[str, str], part_dir: Path) -> None:
    """Overwrite the model's weights with one packed part's decoded tensors."""
    import numpy as np
    import torch

    from orka.pipeline.decode import _decode_tensor

    named_params = dict(model.named_parameters())
    part_manifest = json.loads((part_dir / "manifest.json").read_text())
    for tm in part_manifest.get("tensors", []):
        param = named_params.get(param_for.get(tm["name"], ""))
        if param is None:
            continue
        decoded = np.asarray(_decode_tensor(part_dir, tm), dtype=np.float32)
        shaped = torch.from_numpy(decoded).reshape([int(x) for x in tm["shape"]])
        with torch.no_grad():
            param.data.copy_(shaped.to(param.device))


def _collect_block_activations(
    model,
    tokenizer,
    prompts,
    param_for: Mapping[str, str],
    device: str,
    max_length: int,
    max_samples: int,
) -> dict:
    """Capture nn.Linear inputs for the given checkpoint names, keyed by those names."""
    import torch

    activations: dict[str, list] = {}
    handles = []
    checkpoint_for = {param: name for name, param in param_for.items()}
    for mod_name, module in model.named_modules():
        checkpoint_name = checkpoint_for.get(mod_name + ".weight")
        if isinstance(module, torch.nn.Linear) and checkpoint_name is not None:
            captured = checkpoint_name

            def hook(_mod, inputs, _outputs, _name=captured):
                x = inputs[0]
                if x.dim() > 2:
                    x = x.reshape(-1, x.shape[-1])
                activations.setdefault(_name, []).append(x.detach().cpu())

            handles.append(module.register_forward_hook(hook))
    if not handles:
        return {}

    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=max_length
            )
            ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            attn = (attn if attn is not None else torch.ones_like(ids)).to(device)
            model(input_ids=ids, attention_mask=attn)
    for handle in handles:
        handle.remove()

    out = {}
    for name, xs in activations.items():
        full = xs[0] if len(xs) == 1 else torch.cat(xs, dim=0)
        if full.shape[0] > max_samples:
            idx = torch.randperm(full.shape[0])[:max_samples]
            full = full[idx]
        out[name] = full.to(dtype=torch.float32)
    return out


def pack_checkpoint_sequential(
    source: Path,
    out_dir: Path,
    model_dir: Path,
    prompts_path: Path,
    *,
    model_device: str = "cpu",
    local_files_only: bool = True,
    calibration_max_prompts: int = 32,
    calibration_max_length: int = 256,
    calibration_max_samples: int = 4096,
    progress_file: Path | None = None,
    **pack_kwargs,
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from orka._runtime import _resolve_torch_device
    from orka.artifact.merge import merge_orka_artifacts
    from orka.eval.prompts import _read_prompt_file
    from orka.pipeline.pack import pack_checkpoint

    model_device = str(_resolve_torch_device(model_device))

    if pack_kwargs.get("codebook_mode", "per-tensor") != "per-tensor":
        raise ValueError(
            "sequential calibration requires codebook_mode='per-tensor' "
            "(each block packs standalone codebooks)"
        )

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"output directory already exists with content: {out_dir}")

    prompts = _read_prompt_file(prompts_path, max_prompts=calibration_max_prompts)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=local_files_only, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    model.to(model_device)
    model.eval()

    report = inspect_checkpoint(Path(source))
    candidates = [t["name"] for t in report["tensors"] if t["candidate"]]
    if not candidates:
        raise RuntimeError("no quantizable tensors found in source")
    ordered_blocks = _group_tensors_by_block(candidates)
    param_for = _checkpoint_param_names(model, set(candidates))

    part_dirs: list[Path] = []
    any_weighted = False
    try:
        for block_i, block_names in enumerate(ordered_blocks):
            _report_progress(
                progress_file,
                f"--- Sequential block {block_i + 1}/{len(ordered_blocks)} "
                f"({len(block_names)} tensors) ---",
            )
            acts = _collect_block_activations(
                model,
                tokenizer,
                prompts,
                {name: param_for[name] for name in block_names if name in param_for},
                model_device,
                calibration_max_length,
                calibration_max_samples,
            )
            any_weighted = any_weighted or bool(acts)
            part_dir = out_dir.parent / f"{out_dir.name}.seq-part-{block_i}"
            if part_dir.exists():
                shutil.rmtree(part_dir)
            pack_checkpoint(
                source=Path(source),
                out_dir=part_dir,
                only_tensors=block_names,
                only_tensors_passthrough=False,
                awq_activations=acts or None,
                progress_file=progress_file,
                **pack_kwargs,
            )
            part_dirs.append(part_dir)
            _load_packed_into_model(model, param_for, part_dir)
    finally:
        try:
            model.to("cpu")
        except Exception:
            pass
        del model, tokenizer
        if model_device != "cpu":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if len(part_dirs) == 1:
        shutil.move(str(part_dirs[0]), str(out_dir))
        manifest = json.loads((out_dir / "manifest.json").read_text())
    else:
        manifest = merge_orka_artifacts(part_dirs, out_dir)
        for part_dir in part_dirs:
            shutil.rmtree(part_dir, ignore_errors=True)

    manifest["sequential_calibration"] = True
    manifest["hessian_weighted"] = any_weighted
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
