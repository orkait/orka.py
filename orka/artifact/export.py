"""Export an .orka artifact to engine-loadable formats.

``export_vllm`` writes a standard Hugging Face model directory (safetensors
weights + config/tokenizer sidecars) that vLLM, transformers, or any HF
loader serves directly - dequantization happens once at export.

Low-rank correction sidecars (W ~ W_q + A @ B^T) can be split out as a
standard PEFT LoRA adapter instead of being merged into the dense weights:
``lora_B = A [out, r]``, ``lora_A = B^T [r, in]``, ``lora_alpha = r`` so the
adapter delta equals the correction exactly. vLLM then applies it at runtime
through its fused LoRA kernels, and one artifact serves both the plain
quantized model and the corrected one.
"""

from __future__ import annotations

import json
from pathlib import Path

from orka.core._checkpoint import _load_tensors
from orka.eval.hf import _copy_hf_sidecars, _resolve_eval_model_dir
from orka.pipeline.decode import _decode_tensor, _read_lowrank

_EXPORT_DTYPES = {"bfloat16", "float16", "float32"}


def _torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _module_path(tensor_name: str) -> str:
    return tensor_name[: -len(".weight")] if tensor_name.endswith(".weight") else tensor_name


def _leaf_name(tensor_name: str) -> str:
    return _module_path(tensor_name).rsplit(".", 1)[-1]


def export_vllm(
    artifact_dir: Path,
    out_dir: Path,
    *,
    model_dir: Path | None = None,
    dtype: str = "bfloat16",
    correction_adapter: bool = True,
    device: str = "cpu",
) -> dict:
    import numpy as np
    import torch
    from safetensors.torch import save_file

    if dtype not in _EXPORT_DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_EXPORT_DTYPES)}")
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Orka manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest["source"])
    resolved_model_dir = _resolve_eval_model_dir(source, model_dir)

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"export directory must be empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = _copy_hf_sidecars(resolved_model_dir, out_dir)

    target_dtype = _torch_dtype(dtype)
    packed_meta = {t["name"]: t for t in manifest.get("tensors", [])}
    corrections: dict[str, tuple] = {}

    weights: dict[str, torch.Tensor] = {}
    for name, tm in packed_meta.items():
        tm_decode = dict(tm)
        lr_meta = tm_decode.get("lowrank")
        if correction_adapter and lr_meta:
            # Base weights exclude the correction; it ships as the adapter.
            tm_decode.pop("lowrank")
            a, b = _read_lowrank(artifact_dir, lr_meta)
            corrections[name] = (a, b)
        decoded = np.asarray(_decode_tensor(artifact_dir, tm_decode), dtype=np.float32)
        shape = [int(x) for x in tm["shape"]]
        weights[name] = torch.from_numpy(decoded).reshape(shape).to(target_dtype).contiguous()

    # Passthrough (norms, biases, skipped tensors) from the artifact itself;
    # source fallback for anything a partial artifact still misses.
    passthrough_path = artifact_dir / "passthrough.safetensors"
    if passthrough_path.exists():
        for name, tensor in _load_tensors(passthrough_path):
            if name in weights:
                continue
            t = tensor if isinstance(tensor, torch.Tensor) else torch.from_numpy(
                np.asarray(tensor, dtype=np.float32)
            )
            weights[name] = t.to(target_dtype).contiguous()
    if source.exists():
        for name, tensor in _load_tensors(source):
            if name in weights:
                continue
            t = tensor if isinstance(tensor, torch.Tensor) else torch.from_numpy(
                np.asarray(tensor, dtype=np.float32)
            )
            weights[name] = t.to(target_dtype).contiguous()

    save_file(weights, str(out_dir / "model.safetensors"), metadata={"format": "pt"})

    adapter_info = None
    if correction_adapter and corrections:
        adapter_info = _write_peft_adapter(out_dir / "correction-adapter", corrections)

    result = {
        "out": str(out_dir),
        "model_tensors": len(weights),
        "quantized_tensors": len(packed_meta),
        "dtype": dtype,
        "copied_sidecars": copied,
        "correction_adapter": adapter_info,
        "source_artifact": str(artifact_dir),
    }
    (out_dir / "orka_export.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _write_peft_adapter(adapter_dir: Path, corrections: dict) -> dict:
    """corrections: {tensor_name: (A [out, r], B [in, r]) numpy arrays}."""
    import torch
    from safetensors.torch import save_file

    adapter_dir.mkdir(parents=True, exist_ok=True)
    max_rank = max(a.shape[1] for a, _ in corrections.values())
    target_leaves = sorted({_leaf_name(n) for n in corrections})

    tensors: dict[str, torch.Tensor] = {}
    for name, (a, b) in corrections.items():
        module = _module_path(name)
        a_t = torch.from_numpy(a.copy())  # [out, r]
        b_t = torch.from_numpy(b.copy())  # [in, r]
        rank = int(a_t.shape[1])
        if rank < max_rank:
            # Zero-padded columns leave A @ B^T unchanged; uniform rank keeps
            # the adapter loadable without per-module rank_pattern support.
            a_t = torch.nn.functional.pad(a_t, (0, max_rank - rank))
            b_t = torch.nn.functional.pad(b_t, (0, max_rank - rank))
        # PEFT: delta = lora_B @ lora_A * (alpha / r). With alpha = r the
        # scaling is 1 and delta == A @ B^T exactly.
        tensors[f"base_model.model.{module}.lora_A.weight"] = (
            b_t.T.contiguous().to(torch.float16)
        )
        tensors[f"base_model.model.{module}.lora_B.weight"] = (
            a_t.contiguous().to(torch.float16)
        )

    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))
    config = {
        "peft_type": "LORA",
        "r": max_rank,
        "lora_alpha": max_rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "target_modules": target_leaves,
        "task_type": "CAUSAL_LM",
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return {
        "path": str(adapter_dir),
        "rank": max_rank,
        "corrected_tensors": len(corrections),
        "target_modules": target_leaves,
    }


def cmd_export_vllm(args) -> int:
    result = export_vllm(
        Path(args.artifact),
        Path(args.out),
        model_dir=Path(args.model_dir) if args.model_dir else None,
        dtype=args.dtype,
        correction_adapter=not args.merge_correction,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "out": result["out"],
                "model_tensors": result["model_tensors"],
                "quantized_tensors": result["quantized_tensors"],
                "dtype": result["dtype"],
                "correction_adapter": (result["correction_adapter"] or {}).get("path"),
            },
            indent=2,
        )
    )
    return 0
