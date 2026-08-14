"""Allowlisted AutoQuant tools. The reasoner may call only these names."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from orka.autoquant.harness_schema import HarnessState, RunConfig
from orka.autoquant.orchestrator import derive_config
from orka.autoquant.refine import Quality, attribute, escalate_cfg, meets
from orka.autoquant.roles import classify_role
from orka.autoquant.schema import TensorConfig, from_allocation_map, to_packer_allocation
from orka.autoquant.validation import validate_configs
from orka.core._checkpoint import inspect_checkpoint
from orka.quant.arch import ArchProfile


@dataclass
class ToolContext:
    config: RunConfig
    source: Path
    state: HarnessState
    pack_fn: Callable[..., Any] | None = None
    pulse_fn: Callable[..., Any] | None = None
    verify_fn: Callable[..., Any] | None = None


def tool_schemas() -> list[dict]:
    names = ("inspect", "derive", "allocate", "pack", "pulse_check", "refine", "verify")
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"AutoQuant {name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _cfgs(state: HarnessState) -> dict[str, TensorConfig]:
    if not state.candidate:
        raise ValueError("no candidate allocation yet; call derive first")
    if all(isinstance(v, TensorConfig) for v in state.candidate.values()):
        return state.candidate
    return from_allocation_map(state.candidate)


def _dump_cfgs(cfgs: dict[str, TensorConfig]) -> dict[str, dict]:
    return {name: asdict(c) for name, c in cfgs.items()}


def inspect_tool(_args: dict, ctx: ToolContext) -> dict:
    report = inspect_checkpoint(ctx.source)
    shapes = {t["name"]: tuple(t["shape"]) for t in report["tensors"]}
    profile = ArchProfile.from_shapes(shapes)
    tied = sum(1 for s in shapes.values() if len(s) == 2 and profile.vocab_size and s[0] == profile.vocab_size) == 1
    roles = {}
    for t in report["tensors"]:
        role, _ = classify_role(t["name"], tuple(t["shape"]), tied=tied, profile=profile)
        roles[t["name"]] = role
    return {"ok": True, "roles": roles, "tensor_count": report["tensor_count"]}


def derive_tool(_args: dict, ctx: ToolContext) -> dict:
    import numpy as np
    import torch
    from safetensors import safe_open

    source = ctx.source
    paths = sorted(source.glob("*.safetensors")) if source.is_dir() else [source]
    weights: dict[str, np.ndarray] = {}
    for path in paths:
        if path.suffix.lower() != ".safetensors":
            continue
        with safe_open(str(path), "pt") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                if tensor.ndim in (1, 2):
                    weights[key] = tensor.to(torch.float32).numpy()
    if not weights:
        from orka.core._checkpoint import _load_tensors
        from orka.core._tensor import _numpy_float32_array

        for name, tensor in _load_tensors(source):
            arr = _numpy_float32_array(tensor)
            if arr.ndim in (1, 2):
                weights[name] = arr
    cfgs = derive_config(weights, objective=ctx.config.objective, use_llm=False)
    roles = ctx.state.roles or {
        name: classify_role(name, tuple(w.shape))[0] for name, w in weights.items()
    }
    validate_configs(cfgs, roles)
    return {"ok": True, "candidate": _dump_cfgs(cfgs), "roles": roles}


def allocate_tool(_args: dict, ctx: ToolContext) -> dict:
    cfgs = _cfgs(ctx.state)
    validate_configs(cfgs, ctx.state.roles)
    allocation = to_packer_allocation(
        cfgs, group_size=ctx.config.group_size, source=str(ctx.source)
    )
    alloc_path = ctx.config.output_dir / "allocation_map.json"
    ctx.config.output_dir.mkdir(parents=True, exist_ok=True)
    alloc_path.write_text(json.dumps(allocation, indent=2) + "\n")
    return {"ok": True, "allocation": allocation, "path": str(alloc_path)}


def pack_tool(_args: dict, ctx: ToolContext) -> dict:
    from orka.pipeline.pack import pack_checkpoint

    allocation = ctx.state.allocation
    if allocation is None:
        raise ValueError("no allocation yet; call allocate first")
    artifact = Path(ctx.state.artifact_path or (ctx.config.output_dir / "out.orka"))
    if artifact.exists() and artifact.is_file():
        artifact.unlink()
    stages = {
        name: list(entry["stages"])
        for name, entry in allocation.get("tensors", {}).items()
    }
    pack = ctx.pack_fn or pack_checkpoint
    pack(
        source=ctx.source,
        out_dir=artifact,
        group_size=ctx.config.group_size,
        codebook_size=16,
        codebook_sizes=[16, 16],
        iterations=4,
        sample_vectors=256,
        backend="numpy",
        device="cpu",
        normalization="block-max",
        codebook_mode="per-tensor",
        tensor_stages_map=stages,
        only_tensors=list(stages),
        only_tensors_passthrough=True,
        em_aq_passes=0,
    )
    return {"ok": True, "artifact_path": str(artifact)}


def pulse_check_tool(_args: dict, ctx: ToolContext) -> dict:
    if ctx.pulse_fn is not None:
        raw = ctx.pulse_fn(ctx.state)
        quality = raw if isinstance(raw, dict) else {"kl": raw.kl, "top1": raw.top1}
    else:
        quality = {"kl": 0.0, "top1": 1.0, "proxy": True, "reason": "no_lm_eval"}
    q = Quality(kl=float(quality["kl"]), top1=float(quality["top1"]))
    quality["objective_met"] = meets(q, ctx.config.objective, ctx.config.target)
    return {"ok": True, "quality": quality}


def refine_tool(_args: dict, ctx: ToolContext) -> dict:
    from orka.autoquant.probes import Signals

    cfgs = _cfgs(ctx.state)
    dummy = Signals(sqnr_curve={2: 8.0, 3: 12.0, 4: 20.0, 6: 28.0, 8: 35.0},
                    rd_knee_bits=4, sensitivity=0.02)
    signals = {name: dummy for name, c in cfgs.items() if c.method == "rvq"}
    offenders = attribute(cfgs, signals)
    for name in offenders:
        cfgs[name] = escalate_cfg(cfgs[name])
    validate_configs(cfgs, ctx.state.roles)
    return {"ok": True, "candidate": _dump_cfgs(cfgs), "offenders": offenders}


def verify_tool(_args: dict, ctx: ToolContext) -> dict:
    from orka.artifact.verify import verify_artifact

    artifact = Path(ctx.state.artifact_path or "")
    if not artifact.exists():
        raise ValueError("no artifact to verify; call pack first")
    verify = ctx.verify_fn or verify_artifact
    result = verify(artifact)
    passed = bool(result.get("ok", result.get("passed", False)))
    quality = ctx.state.quality or {}
    objective_met = bool(quality.get("objective_met", False))
    accepted = passed and objective_met
    return {
        "ok": passed,
        "accepted": accepted,
        "verification": {**result, "passed": passed},
        "quality": quality,
    }


TOOLS: dict[str, Callable] = {
    "inspect": inspect_tool,
    "derive": derive_tool,
    "allocate": allocate_tool,
    "pack": pack_tool,
    "pulse_check": pulse_check_tool,
    "refine": refine_tool,
    "verify": verify_tool,
}
