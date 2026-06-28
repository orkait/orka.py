"""CLI handlers for the codebook-free E8-lattice compressor (orka.quant.lattice_pack)."""
from __future__ import annotations

import argparse
import json

from orka._runtime import _apply_gpu_memory_cap, _resolve_torch_device


def cmd_lattice_compress(args: argparse.Namespace) -> int:
    from orka.quant.lattice_pack import compress_model

    device = str(_resolve_torch_device(args.device))
    _apply_gpu_memory_cap("torch", device, getattr(args, "max_gpu_mem_gb", None))
    scales = [float(s) for s in args.scales]
    meta = compress_model(args.model, args.out, scales=scales, seed=args.seed, device=device)
    print(json.dumps({
        "out": args.out,
        "scales": scales,
        "avg_bpw_quantized": round(meta["avg_bpw_quantized"], 3),
        "payload_bytes": meta["payload_bytes"],
        "passthrough_bytes": meta["passthrough_bytes"],
    }, indent=2))
    return 0


def cmd_lattice_reconstruct(args: argparse.Namespace) -> int:
    from orka.quant.lattice_pack import reconstruct_to_hf

    device = str(_resolve_torch_device(args.device))
    out = reconstruct_to_hf(args.artifact, args.model, args.out, device=device)
    print(json.dumps({"out": out}))
    return 0
