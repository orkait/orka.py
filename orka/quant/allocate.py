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

# Per-tensor transform search ranks candidates with a CHEAP VQ probe (small sample,
# few iters) at this reference spec - not the scalar-quant proxy, which measured
# anti-correlated with full VQ on real tensors (Spearman -0.44, 0/10 top-1). A
# subsampled VQ probe is faithful by construction (it is VQ): 7/10 top-1, +0.72.
_TRANSFORM_RANK_SPEC = "vq-8"
_TRANSFORM_PROBE_VECTORS = 1024
_TRANSFORM_PROBE_ITERS = 2


def _spec_bits_per_vector(stages: Sequence) -> int:
    total = 0
    for k in stages:
        if isinstance(k, str) and k.startswith("s"):
            total += int(k[1:])
        else:
            total += _index_bits_for_size(int(k))
    return total


def _spec_codebook_bytes(stages: Sequence, group_size: int, dtype_bytes: int = 2) -> int:
    """On-disk codebook bytes for a spec: ``sum_stage K * group_size * dtype_bytes``.

    Scalar ('s') stages carry no learned codebook (uniform dequant), so they cost 0 -
    this is what gives planar specs their headroom on small tensors, where a VQ
    stage's fixed K*group_size codebook dwarfs the index savings.
    """
    total = 0
    for k in stages:
        if isinstance(k, str) and k.startswith("s"):
            continue
        total += int(k) * group_size * dtype_bytes
    return total


# Planar (scalar) candidates appended to the menu under --size-aware so the allocator
# can escape the per-tensor codebook tax where VQ does not earn it.
PLANAR_CANDIDATE_SPECS = ("rvq-s8", "rvq-s8-s8", "rvq-s8-s8-s8")


def _probe_spec_distortion(
    vectors, stages: Sequence, iterations: int, backend: str, device: str, seed: int,
    sample_weights=None,
) -> float:
    """Greedy RVQ on the sample; returns mean squared error per value.

    When ``sample_weights`` (per-vector importance h, length N) is given, the probe
    optimizes the OUTPUT-error proxy instead of raw weight MSE: weighted k-means pulls
    centroids toward high-importance vectors, and the returned distortion is the
    importance-weighted mean squared error. This is what makes the allocation reflect
    each tensor's effect on the model output, not just its weight reconstruction.
    """
    residual = vectors
    decoded_sum = None
    n = len(vectors)
    for stage_i, k_spec in enumerate(stages):
        if isinstance(k_spec, str) and k_spec.startswith("s"):
            k = 1 << int(k_spec[1:])
            v_res = residual.reshape(-1, 1)
            sw = None  # scalar stage reshapes to [N*d, 1]; per-vector weights don't map
        else:
            k = int(k_spec)
            v_res = residual
            sw = sample_weights
        cb, _, _ = learn_codebook_auto(
            v_res, min(k, len(v_res)), iterations, backend, device,
            seed=seed + stage_i, sample_weights=sw,
        )
        indices, _ = quantize_vectors_auto(v_res, cb, backend, device)
        dec = _decode_to_vectors_format(v_res, cb, indices, backend, device)
        dec = dec.reshape(residual.shape) if dec.shape != residual.shape else dec
        decoded_sum = dec if decoded_sum is None else decoded_sum + dec
        residual = _vectors_subtract(vectors, decoded_sum)
    import numpy as np

    diff = np.asarray(residual, dtype=np.float32) if not hasattr(residual, "detach") else residual.detach().cpu().numpy()
    sq = diff.astype(np.float32) ** 2  # [N, d]
    if sample_weights is None:
        return float(np.mean(sq))
    w = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
    return float((w * sq.sum(axis=1)).sum() / (w.sum() * sq.shape[1] + 1e-12))


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
    search_transforms: bool = False,
    transform_block: int = 128,
    size_aware: bool = False,
    codebook_dtype_bytes: int = 2,
    awq_activations: dict | None = None,
) -> dict:
    import numpy as np

    def _per_vector_importance(h, numel: int):
        # h_j = E[x_j^2] per input column -> per-group-vector importance (mean over the
        # group_size columns the vector covers, row-major). Length matches the padded
        # vector count, so it samples in lockstep with the probe vectors.
        hh = np.asarray(h, dtype=np.float32).reshape(-1)
        if hh.size == 0:
            return None
        reps = -(-numel // hh.size)
        hf = np.tile(hh, reps)[:numel]
        pad = (-numel) % group_size
        if pad:
            hf = np.pad(hf, (0, pad))
        return hf.reshape(-1, group_size).mean(axis=1)

    # Size-aware allocation charges each VQ spec its fixed per-tensor codebook tax and
    # offers planar (scalar) specs that pay none - so the allocator picks planar on
    # small tensors and VQ on large ones by true on-disk size, not index bits alone.
    if size_aware:
        candidate_specs = tuple(dict.fromkeys((*candidate_specs, *PLANAR_CANDIDATE_SPECS)))

    specs = []
    for spec in candidate_specs:
        stages = parse_quant_spec(spec)
        specs.append((spec, stages, _spec_bits_per_vector(stages)))
    specs.sort(key=lambda item: item[2])
    if len({bits for _, _, bits in specs}) < 2:
        raise ValueError("need at least two candidate specs with distinct rates")
    spec_cb_bytes = [_spec_codebook_bytes(st, group_size, codebook_dtype_bytes) for _, st, _ in specs]

    # The probe must use >> max-K vectors or k-means trivially fits (K >= N -> one
    # centroid per vector -> 0 distortion), which blinds the allocator to the high-bit
    # specs and makes it under-spend the budget. Scale the sample to OVERSAMPLE x the
    # largest candidate codebook (capped), letting it fall back to the whole tensor.
    _OVERSAMPLE, _PROBE_CAP = 4, 1 << 18
    max_k = max((int(k) for _, st, _ in specs for k in st
                 if not (isinstance(k, str) and k.startswith("s"))), default=0)
    # None = use every vector (already the most accurate); otherwise lift the sample to
    # >> max-K so the probe doesn't saturate.
    probe_sample = None if sample_vectors is None else min(
        max(sample_vectors, _OVERSAMPLE * max_k), _PROBE_CAP
    )

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
        seed = _derive_seed(["allocate", name, group_size])

        # Stage 1: per-tensor transform pick (cheap scalar-quant proxy). The chosen
        # transform is applied before the spec probe; its denorm_factor restores
        # original weight units so distortions stay comparable across tensors that
        # picked different transforms. Default (search off) = raw probe, factor 1.
        transform = None
        denorm_factor = 1.0
        probe_flat = flat
        if search_transforms:
            from orka.quant.transform_search import apply_transform, DEFAULT_TRANSFORM_GRID

            w2d = flat.reshape(shape[0], -1) if len(shape) >= 2 else flat.reshape(1, -1)
            ref_stages = parse_quant_spec(_TRANSFORM_RANK_SPEC)
            best = None  # (orig_mse, norm, rot, transformed_2d, denorm_factor)
            for t_norm, t_rot in DEFAULT_TRANSFORM_GRID:
                try:
                    wt, factor = apply_transform(w2d, t_norm, t_rot, norm_block=transform_block)
                except ValueError:
                    continue  # transform infeasible for this width (e.g. Hadamard)
                ft = np.asarray(wt, dtype=np.float32).reshape(-1)
                p = (-ft.size) % group_size
                if p:
                    ft = np.pad(ft, (0, p))
                st = _sample_vector_rows(ft.reshape(-1, group_size), _TRANSFORM_PROBE_VECTORS)
                mse = _probe_spec_distortion(
                    st, ref_stages, _TRANSFORM_PROBE_ITERS, backend, device, seed
                ) * factor
                if best is None or mse < best[0]:
                    best = (mse, t_norm, t_rot, wt, factor)
            if best is not None:
                _, t_norm, t_rot, wt, denorm_factor = best
                probe_flat = np.asarray(wt, dtype=np.float32).reshape(-1)
                transform = {"normalization": t_norm, "rotation": t_rot}

        pad = (-probe_flat.size) % group_size
        if pad:
            probe_flat = np.pad(probe_flat, (0, pad))
        vectors = probe_flat.reshape(-1, group_size)

        # Importance weighting: optimize output error (Hessian-weighted) not raw weight
        # MSE. Per-vector weights come from the calibration activations of this tensor;
        # exact when columns are preserved (none/block-max), approximate under rotation
        # (column mixing) - the activations are kept pre-transform for v1.
        sw_full = None
        if awq_activations is not None and name in awq_activations:
            act = awq_activations[name]
            xa = act.detach().cpu().numpy() if hasattr(act, "detach") else np.asarray(act)
            if xa.ndim == 2 and xa.shape[1] == int(shape[-1]):
                sw_full = _per_vector_importance((xa.astype(np.float32) ** 2).mean(axis=0), numel)

        n_vec = vectors.shape[0]
        if sw_full is not None and probe_sample is not None and n_vec > probe_sample:
            idx = np.random.default_rng(seed).choice(n_vec, probe_sample, replace=False)
            sample, sw_sample = vectors[idx], sw_full[idx]
        elif sw_full is not None:
            sample, sw_sample = vectors, sw_full
        else:
            sample, sw_sample = _sample_vector_rows(vectors, probe_sample), None

        # Stage 2: spec RD probe on the (transformed) vectors.
        distortions = []
        for spec, stages, bits in specs:
            mse = _probe_spec_distortion(
                sample, stages, iterations, backend, device, seed,
                sample_weights=sw_sample,
            )
            # Sample MSE -> estimated total SSE in original units, for cross-tensor
            # comparison.
            distortions.append(mse * denorm_factor * numel)
        _report_progress(
            progress_file,
            f"allocate: probed {name}"
            + (" [hessian]" if sw_sample is not None else "")
            + (f" [{transform['normalization']}/{transform['rotation']}]" if transform else "")
            + " " + ", ".join(f"{s}={d:.4g}" for (s, _, _), d in zip(specs, distortions)),
        )
        rows.append({
            "name": name, "numel": numel, "distortions": distortions, "transform": transform,
        })

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
        # Index bits, plus (size-aware) the spec's fixed codebook tax for this tensor.
        b = rows[t]["numel"] * specs[s][2] / group_size
        if size_aware:
            b += spec_cb_bytes[s] * 8
        return b

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
        entry = {
            "spec": spec,
            "stages": [s if isinstance(s, str) else int(s) for s in stages],
            "bits_per_weight": bits / group_size,
            "estimated_sse": row["distortions"][choice[t]],
        }
        if row.get("transform"):
            entry.update(row["transform"])  # per-tensor normalization / rotation
        tensors[row["name"]] = entry

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


def allocation_tensor_transforms(allocation: dict) -> dict[str, dict]:
    """Allocation JSON -> {tensor name: {normalization?, rotation?}} for pack_checkpoint.

    Empty when no tensor carries a per-tensor transform override, so callers pass the
    result straight through (pack treats a falsy map as "use the global transforms")."""
    out = {}
    for name, entry in allocation.get("tensors", {}).items():
        over = {k: entry[k] for k in ("normalization", "rotation") if k in entry}
        if over:
            out[name] = over
    return out


def cmd_allocate(args) -> int:
    awq_activations = None
    if getattr(args, "hessian", False):
        from orka.quant.activations import _load_awq_activations

        args.no_hessian = False
        awq_activations = _load_awq_activations(args)
        if awq_activations is None:
            print("WARNING: --hessian requested but activations unavailable; "
                  "falling back to unweighted allocation.")
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
        search_transforms=getattr(args, "search_transforms", False),
        size_aware=getattr(args, "size_aware", False),
        awq_activations=awq_activations,
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
