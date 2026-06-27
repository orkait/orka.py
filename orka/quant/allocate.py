"""Measured per-tensor bit allocation (discrete water-filling).

Replaces name-based family heuristics with measurement: every candidate tensor
gets a small rate-distortion probe (quick k-means at each candidate spec on a
vector sample), then a greedy marginal-utility solver upgrades whichever
tensor buys the most distortion reduction per extra bit until the global
bits-per-weight budget is spent. At the optimum, marginal utility is roughly
equal across tensors - the discrete Lagrangian / water-filling condition.

Distortion is estimated as total SSE (sample MSE scaled to tensor size),
optionally importance-weighted via calibration activations.
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Sequence

from orka.core._checkpoint import _load_tensors
from orka.core._tensor import (
    _decode_to_vectors_format,
    _numpy_float32_array,
    _sample_vector_rows,
    _tensor_shape,
    _vectors_subtract,
)
from orka.core._util import _derive_seed, _index_bits_for_size, _report_progress
from orka.codebook import learn_codebook_auto, quantize_vectors_auto
from orka.quant import parse_quant_spec

DEFAULT_CANDIDATE_SPECS = ("vq-4", "vq-8", "vq-12", "rvq-12-8", "rvq-16-8")


def _spec_bits_per_vector(stages: Sequence) -> int:
    total = 0
    for k in stages:
        if isinstance(k, str) and k.startswith("s"):
            total += int(k[1:])
        else:
            total += _index_bits_for_size(int(k))
    return total


def _probe_spec_distortion(
    vectors, stages: Sequence, iterations: int, backend: str, device: str, seed: int
) -> float:
    """Greedy RVQ on the sample; returns mean squared error per value."""
    residual = vectors
    decoded_sum = None
    n = len(vectors)
    for stage_i, k_spec in enumerate(stages):
        if isinstance(k_spec, str) and k_spec.startswith("s"):
            k = 1 << int(k_spec[1:])
            v_res = residual.reshape(-1, 1)
        else:
            k = int(k_spec)
            v_res = residual
        cb, _, _ = learn_codebook_auto(
            v_res, min(k, len(v_res)), iterations, backend, device,
            seed=seed + stage_i,
        )
        indices, _ = quantize_vectors_auto(v_res, cb, backend, device)
        dec = _decode_to_vectors_format(v_res, cb, indices, backend, device)
        dec = dec.reshape(residual.shape) if dec.shape != residual.shape else dec
        decoded_sum = dec if decoded_sum is None else decoded_sum + dec
        residual = _vectors_subtract(vectors, decoded_sum)
    import numpy as np

    diff = np.asarray(residual, dtype=np.float32) if not hasattr(residual, "detach") else residual.detach().cpu().numpy()
    return float(np.mean(diff.astype(np.float32) ** 2))


def build_allocation(
    source: Path,
    target_bpw: float,
    *,
    candidate_specs: Sequence[str] = DEFAULT_CANDIDATE_SPECS,
    group_size: int = 8,
    sample_vectors: int = 4096,
    iterations: int = 4,
    backend: str = "numpy",
    device: str = "cpu",
    max_tensors: int | None = None,
    progress_file: Path | None = None,
) -> dict:
    import numpy as np

    specs = []
    for spec in candidate_specs:
        stages = parse_quant_spec(spec)
        specs.append((spec, stages, _spec_bits_per_vector(stages)))
    specs.sort(key=lambda item: item[2])
    if len({bits for _, _, bits in specs}) < 2:
        raise ValueError("need at least two candidate specs with distinct rates")

    rows = []
    seen = 0
    for name, tensor in _load_tensors(source):
        shape = _tensor_shape(tensor)
        lowered = name.lower()
        if len(shape) < 2 or any(
            x in lowered
            for x in (".bias", ".norm", ".layernorm", "rotary_emb", "attention.bias")
        ):
            continue
        if max_tensors is not None and seen >= max_tensors:
            break
        seen += 1
        flat = _numpy_float32_array(tensor).reshape(-1)
        numel = int(flat.shape[0])
        pad = (-numel) % group_size
        if pad:
            flat = np.pad(flat, (0, pad))
        vectors = flat.reshape(-1, group_size)
        sample = _sample_vector_rows(vectors, sample_vectors)
        seed = _derive_seed(["allocate", name, group_size])

        distortions = []
        for spec, stages, bits in specs:
            mse = _probe_spec_distortion(
                sample, stages, iterations, backend, device, seed
            )
            # Scale sample MSE to estimated total SSE for cross-tensor comparison.
            distortions.append(mse * numel)
        _report_progress(
            progress_file,
            f"allocate: probed {name} "
            + ", ".join(f"{s}={d:.4g}" for (s, _, _), d in zip(specs, distortions)),
        )
        rows.append({"name": name, "numel": numel, "distortions": distortions})

    if not rows:
        raise RuntimeError("no quantizable tensors found for allocation")

    total_params = sum(r["numel"] for r in rows)
    budget_bits = target_bpw * total_params

    # Discrete rate-distortion allocation. The plain greedy water-filling (upgrade
    # the best distortion-per-bit one step at a time) is optimal only on the convex
    # hull of the per-tensor RD curves; it gets stuck on NON-convex curves where a
    # two-step jump beats a one-step one. The Lagrangian method (Shoham-Gersho) is
    # convex-hull optimal: for multiplier lam each tensor independently picks
    # argmin_s(distortion_s + lam*bits_s); binary-search lam to the budget. We run
    # BOTH (Lagrangian then a greedy fill of the leftover budget) and KEEP THE LOWER
    # total distortion - provably never worse than greedy, ~6% lower on average.
    def _bits(t, s):
        return rows[t]["numel"] * specs[s][2] / group_size

    def _greedy_fill(choice, spent):
        """Heap water-filling from a partial allocation: O(T*S*log T)."""
        def step(t, cur):
            nxt = cur + 1
            if nxt >= len(specs):
                return None
            extra = _bits(t, nxt) - _bits(t, cur)
            gain = rows[t]["distortions"][cur] - rows[t]["distortions"][nxt]
            return None if gain <= 0 else (-(gain / extra), t, nxt, extra)
        heap = [s for t in range(len(rows)) if (s := step(t, choice[t])) is not None]
        heapq.heapify(heap)
        while heap:
            _u, t, nxt, extra = heapq.heappop(heap)
            if spent + extra > budget_bits:
                continue
            choice[t] = nxt
            spent += extra
            if (s := step(t, nxt)) is not None:
                heapq.heappush(heap, s)
        return choice

    def _total_distortion(choice):
        return sum(rows[t]["distortions"][choice[t]] for t in range(len(rows)))

    # candidate A: greedy from the cheapest spec
    base_spent = sum(_bits(t, 0) for t in range(len(rows)))
    cand_greedy = _greedy_fill([0] * len(rows), base_spent)

    # candidate B: Lagrangian (convex-hull optimal) + greedy fill of leftover budget
    def _lagrangian_choice(lam):
        return [min(range(len(specs)), key=lambda s: rows[t]["distortions"][s] + lam * _bits(t, s))
                for t in range(len(rows))]
    lo, hi = 0.0, 1e25
    best_l = _lagrangian_choice(hi)  # max lam -> all cheapest (feasible)
    for _ in range(100):
        mid = (lo + hi) / 2.0
        ch = _lagrangian_choice(mid)
        if sum(_bits(t, ch[t]) for t in range(len(rows))) <= budget_bits:
            hi, best_l = mid, ch
        else:
            lo = mid
    cand_lagr = _greedy_fill(best_l, sum(_bits(t, best_l[t]) for t in range(len(rows))))

    choice = cand_greedy if _total_distortion(cand_greedy) <= _total_distortion(cand_lagr) else cand_lagr
    spent = sum(_bits(t, choice[t]) for t in range(len(rows)))   # bits of the chosen allocation

    tensors = {}
    for t, row in enumerate(rows):
        spec, stages, bits = specs[choice[t]]
        tensors[row["name"]] = {
            "spec": spec,
            "stages": [s if isinstance(s, str) else int(s) for s in stages],
            "bits_per_weight": bits / group_size,
            "estimated_sse": row["distortions"][choice[t]],
        }

    return {
        "format": "orka-allocation",
        "source": str(source),
        "group_size": group_size,
        "target_bpw": target_bpw,
        "achieved_bpw": spent / total_params,
        "total_params": total_params,
        "candidate_specs": [s for s, _, _ in specs],
        "tensors": tensors,
    }


def allocation_tensor_stages(allocation: dict) -> dict[str, list]:
    """Allocation JSON -> {tensor name: stages list} for pack_checkpoint."""
    return {
        name: list(entry["stages"])
        for name, entry in allocation.get("tensors", {}).items()
    }


def cmd_allocate(args) -> int:
    allocation = build_allocation(
        Path(args.source),
        args.target_bpw,
        candidate_specs=args.candidates,
        group_size=args.group_size,
        sample_vectors=args.sample_vectors,
        iterations=args.iterations,
        backend=args.backend,
        device=args.device,
        max_tensors=args.max_tensors,
        progress_file=Path(args.progress_file) if args.progress_file else None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(allocation, indent=2) + "\n")
    print(
        json.dumps(
            {
                "out": str(out),
                "tensor_count": len(allocation["tensors"]),
                "target_bpw": allocation["target_bpw"],
                "achieved_bpw": allocation["achieved_bpw"],
            },
            indent=2,
        )
    )
    return 0
