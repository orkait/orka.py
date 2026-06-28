"""``pack_checkpoint``: source -> normalize -> rotate -> outliers -> RVQ stages
-> joint EM-AQ -> manifest write.

Currently a single ~700 LOC orchestrator. Splitting into named phases is a
follow-up; would not change semantics.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from typing import Sequence

from orka.core._format import (
    ORKA_VERSION,
    _cast_codebook_storage,
    _write_codebook,
    _write_indices,
    _write_passthrough_tensors,
)
from orka._runtime import (
    _BG_WRITER,
    _maybe_fallback_cuda_to_cpu,
    _resolve_auto_backend,
    _resolve_torch_device,
    _check_ram_cap,
)
from orka.core._tensor import (
    _concat_vector_parts,
    _decode_to_vectors_format,
    _is_numpy_array,
    _is_torch_tensor,
    _numpy_float32_array,
    _sample_vector_rows,
    _tensor_shape,
    _torch_f32,
    _vectors_subtract,
)
from orka.core._util import (
    _derive_seed,
    _index_bits_for_size,
    _report_progress,
    _safe_tensor_name,
    _source_signature,
)
from orka.core._checkpoint import _load_tensors
from orka.codebook import (
    _codebook_cache_key,
    _codebook_cache_load,
    _codebook_cache_save,
    learn_codebook_auto,
    quantize_vectors_auto,
)
from orka.quant import classify_tensor_family
from orka.pipeline.pack_helpers import (
    _numpy_vectors_from_tensor,
    _sample_vectors_and_weights,
    _torch_vectors_from_tensor,
    _weights_digest,
)
from orka.pipeline.pack_manifest import (
    _finalize_tensor_manifest_entry,
    _persist_manifest,
)
from orka.pipeline.pack_pipeline import (
    PackCtx,
    _offload,
    _onload,
    process_batched_candidates,
    process_streamed_per_tensor_candidate,
)
from orka.pipeline.strategies import (
    _refine_scales_ls,
    _run_em_aq_refinement,
    maybe_compensate_candidate,
)
from orka.transforms import (
    _apply_normalization,
    _extract_outliers,
    _rotate_tensor_to_2d,
    stores_block_scales,
)


from orka.pipeline.pack_config import (
    EMBEDDING_MAX_GROUP_SIZE,
    IMPORTANCE_WEIGHT_FLOOR,
    PREFETCH_POLL_TIMEOUT_S,
    PREFETCH_QUEUE_DEPTH,
    SENSITIVITY_SKIP_LOSS_DELTA,
    validate_pack_args,
)


# Vector-prep helpers (_weights_digest, _sample_vectors_and_weights,
# _numpy_vectors_from_tensor, _torch_vectors_from_tensor) live in
# orka.pipeline.pack_helpers and are imported above.


# Sidecar persistence + manifest assembly (_persist_tensor_sidecars,
# _build_tensor_manifest_entry, _release_candidate_payload, _finalize_tensor_manifest_entry,
# _persist_manifest) live in orka.pipeline.pack_manifest and are imported above.

# The EM-AQ joint-optimization phase (_run_em_aq_refinement) and _offload_to_cpu
# live in orka.pipeline.strategies.refinement and are imported above.


def _prefetch_worker(
    ctx: PackCtx,
    *,
    source: Path,
    only_tensors: list[str] | None,
    only_tensors_passthrough: bool,
    tensor_partition_count: int | None,
    tensor_partition_index: int | None,
    awq_alpha: float,
    slrq_salient: bool,
    max_values_per_tensor: int | None,
    prefetch_queue: "queue.Queue",
    prefetch_done: threading.Event,
    _prefetch_exc: list,
    _prefetch_state: dict,
    _passthrough: dict,
    awq_fallbacks: list,
) -> None:
    """Producer thread body for pack_checkpoint: iterate _load_tensors(source),
    do shape/candidate detection -> normalization -> rotation -> outlier/salient
    extraction -> vector-prep per tensor, and prefetch_queue.put(...) each candidate.

    Config that already lives on PackCtx (backend, normalization, rotation, etc.) is
    read off ``ctx``; the remaining run knobs and the shared mutable state (queue, the
    prefetch_done Event, _prefetch_exc / _prefetch_state, _passthrough, awq_fallbacks)
    are threaded in explicitly. Pulled out of pack_checkpoint as a module-level function
    so the orchestrator shrinks; behaviour is identical to the former nested closure.
    """
    # Unpack the ctx-resident config so the body below reads exactly as it did inline.
    backend = ctx.backend
    n_stages = ctx.n_stages
    codebook_mode = ctx.codebook_mode
    group_size = ctx.group_size
    resolved_device = ctx.resolved_device
    rotation = ctx.rotation
    rotation_seed = ctx.rotation_seed
    skipped_tensors = ctx.skipped_tensors
    awq_activations = ctx.awq_activations
    normalization = ctx.normalization
    max_tensors = ctx.max_tensors
    block_scale_size = ctx.block_scale_size
    tensor_stages_resolved = ctx.tensor_stages_resolved
    tensor_transforms_resolved = ctx.tensor_transforms_resolved

    try:
        tensors_emitted = 0
        for name, tensor in _load_tensors(source):
            _check_ram_cap()

            shape = _tensor_shape(tensor)
            name_lower = name.lower()
            is_candidate = len(shape) >= 2

            # Exclude biases, norms, and architectural sidecars.
            if any(
                x in name_lower
                for x in (".bias", ".norm", ".layernorm", "rotary_emb", "attention.bias")
            ):
                is_candidate = False

            # Non-candidates (norms, biases) always pass through so any
            # partial artifact stays completable.
            if not is_candidate:
                _passthrough[name] = tensor
                continue

            if only_tensors is not None:
                base_name = name.replace(".weight", "")
                if name not in only_tensors and base_name not in only_tensors:
                    # Unlisted candidates: passthrough by default (legacy
                    # behaviour); skipped entirely for partitioned runs
                    # (sequential packing) so partial artifacts stay small.
                    if only_tensors_passthrough:
                        _passthrough[name] = tensor
                    continue

            if max_tensors is not None and tensors_emitted >= max_tensors:
                break

            # Skipped tensors stay FP16 in the artifact (passthrough), not quantized.
            if name.replace(".weight", "") in skipped_tensors or name in skipped_tensors:
                _passthrough[name] = tensor
                continue

            if tensor_partition_count is not None:
                slot = _prefetch_state["candidate_count"] % tensor_partition_count
                _prefetch_state["candidate_count"] += 1
                if slot != tensor_partition_index:
                    continue
            else:
                _prefetch_state["candidate_count"] += 1

            # Per-tensor transform overrides (allocation map). Shared codebooks need one
            # normalization/rotation across the tensors they cover, so overrides apply in
            # per-tensor mode only. Missing tensor/field -> global ctx value, keeping the
            # default path byte-identical.
            _tf = (
                tensor_transforms_resolved.get(name, {})
                if tensor_transforms_resolved and codebook_mode == "per-tensor"
                else {}
            )
            t_normalization = _tf.get("normalization", normalization)
            t_rotation = _tf.get("rotation", rotation)

            row_scales = None
            source_flat = None
            awq_col_scales = None
            salient_weights = None
            salient_indices = None
            if t_normalization in {
                "block-max",
                "channel-block-max",
                "awq",
                "awq-block-max",
                "slrq-block",
            }:
                (
                    tensor, row_scales, source_flat, awq_col_scales,
                    salient_weights, salient_indices
                ) = _apply_normalization(
                    tensor, name, t_normalization, awq_activations, awq_alpha,
                    block_scale_size, backend, resolved_device, awq_fallbacks,
                    slrq_salient=slrq_salient,
                )

            # --- Post-Normalization Pre-processing ---
            # Capture source_flat if not already set by normalization.
            # This is mandatory for quality metrics verification.
            if source_flat is None:
                if backend == "torch":
                    _, _arr = _torch_f32(tensor, resolved_device)
                    source_flat = _arr.reshape(-1).detach().cpu()
                else:
                    source_flat = _numpy_float32_array(tensor).reshape(-1)

            tensor_seed = None
            tensor_rotation = "none"
            if t_rotation in {"orthogonal", "hadamard"}:
                tensor, tensor_seed = _rotate_tensor_to_2d(
                    tensor, name, t_rotation, rotation_seed, backend, resolved_device
                )
                tensor_rotation = t_rotation

            # --- PER-TENSOR GROUP SIZING ---
            # All families use the base group_size; only embedding caps at 8.
            # Per-family group overrides (the old attn->g4 / mlp->g16 heuristic)
            # were removed: enlarging the group at a fixed codebook size collapses
            # fidelity, since k centroids must tile a higher-dim space - see the
            # inline rationale below.
            # Shared codebooks (global/family) require one vector width across
            # all tensors they cover, so any override is per-tensor only.
            # Measured allocation (tensor_stages_map) plans bits at one uniform
            # group size, so overrides are disabled in that mode too.
            family = classify_tensor_family(name)
            resolved_group_size = group_size
            if codebook_mode == "per-tensor" and tensor_stages_resolved is None:
                if family == "embedding":
                    resolved_group_size = min(group_size, EMBEDDING_MAX_GROUP_SIZE)
                # attention/mlp/expert: keep the base group_size. Enlarging the
                # group at fixed codebook size collapses fidelity - k centroids
                # must tile a higher-dimensional space (measured: mlp 10.6 dB @
                # g16 vs 16.1 dB @ g8). The old heuristic also tightened attention
                # to g4 (35 dB, far past need) while starving mlp, an inverted bit
                # allocation. Uniform base group equalizes SQNR across families.
                # router/other: keep the user-specified group_size

            if backend == "torch":
                packed_values, padded_values, vectors = _torch_vectors_from_tensor(
                    tensor, resolved_group_size, max_values_per_tensor, resolved_device
                )
            else:
                packed_values, padded_values, vectors = _numpy_vectors_from_tensor(
                    tensor, resolved_group_size, max_values_per_tensor
                )

            # --- HESSIAN-PROXY IMPORTANCE WEIGHTS ---
            # h_j = E[x_j^2] per input column from calibration activations.
            # Two weight sets feed weighted k-means:
            #   vector_weights  - within-group dimension pattern (global average),
            #                     scales the distance metric per dimension.
            #   sample_weights  - per-vector scalar = mean importance of the columns
            #                     that vector covers, tiled across rows (row-major
            #                     flatten). Pulls centroids toward high-energy
            #                     column groups in the Lloyd update.
            vw = None
            sw = None
            col_importance = None
            if (awq_activations is not None and name in awq_activations and shape[-1] % resolved_group_size == 0):
                import torch
                H_diag = torch.as_tensor(awq_activations[name], dtype=torch.float32).pow(2).mean(dim=0)
                cols = int(shape[-1])
                # Column importance is in ORIGINAL column space; rotation
                # mixes columns, so salience-guided escape is rotation-off only.
                if tensor_rotation == "none":
                    col_importance = (
                        H_diag if backend == "torch" else H_diag.numpy()
                    )
                groups_per_row = cols // resolved_group_size
                h_groups = H_diag.reshape(groups_per_row, resolved_group_size)
                vw = h_groups.mean(dim=0).clamp(min=IMPORTANCE_WEIGHT_FLOOR).tolist()
                rows_count = padded_values // cols
                if rows_count * cols == padded_values:
                    sw_row = h_groups.mean(dim=1).clamp(min=IMPORTANCE_WEIGHT_FLOOR)
                    sw_row = sw_row / sw_row.mean()
                    sw_full = sw_row.repeat(rows_count)
                    if backend == "torch":
                        sw = sw_full
                    else:
                        sw = sw_full.numpy()

            prefetch_queue.put({
                "name": name, "shape": shape, "source_flat": source_flat,
                "packed_values": packed_values, "padded_values": padded_values,
                "vectors": vectors, "row_scales": row_scales, "awq_col_scales": awq_col_scales,
                "salient_weights": salient_weights, "salient_indices": salient_indices,
                "normalization": t_normalization,
                "block_scale_size": (
                    block_scale_size if stores_block_scales(t_normalization) else None
                ),
                "family": family, "rotation_seed": tensor_seed,
                "rotation": tensor_rotation,
                "group_size": resolved_group_size,
                "vector_weights": vw, "sample_weights": sw,
                "col_importance": col_importance, "stages_data": {},
            })
            tensors_emitted += 1

    except BaseException as exc:
        _prefetch_exc.append(exc)
    finally:
        prefetch_done.set()


def pack_checkpoint(
    source: Path,
    out_dir: Path,
    group_size: int,
    codebook_size: int = 256,
    iterations: int = 12,
    max_values_per_tensor: int | None = None,
    codebook_mode: str = "per-tensor",
    sample_vectors: int | None = None,
    backend: str = "auto",
    normalization: str = "none",
    device: str = "cpu",
    codebook_sizes: Sequence[int] | None = None,
    family_stages_map: dict[str, Sequence[int]] | None = None,
    tensor_stages_map: dict[str, Sequence] | None = None,
    tensor_transforms_map: dict[str, dict] | None = None,
    outlier_frac: float = 0.0,
    rotation: str = "none",
    rotation_seed: int | None = None,
    awq_activations: dict | None = None,
    awq_alpha: float = 0.5,
    max_tensors: int | None = None,
    only_tensors: list[str] | None = None,
    only_tensors_passthrough: bool = True,
    progress_file: Path | None = None,
    sensitivity_map: dict | None = None,
    codebook_cache_dir: Path | None = None,
    block_scale_size: int = 32,
    codebook_dtype: str = "float16",
    em_aq_passes: int = 3,
    slrq_salient: bool = True,
    tensor_partition_count: int | None = None,
    tensor_partition_index: int | None = None,
    error_compensation: bool = False,
    mse_scale: bool = False,
) -> dict:
    validate_pack_args(
        codebook_mode=codebook_mode,
        backend=backend,
        normalization=normalization,
        rotation=rotation,
        awq_activations=awq_activations,
        tensor_partition_count=tensor_partition_count,
        tensor_partition_index=tensor_partition_index,
    )
    backend = _resolve_auto_backend(backend)
    if backend == "torch":
        device = _maybe_fallback_cuda_to_cpu(device, backend)
        resolved_device = str(_resolve_torch_device(device))
    else:
        resolved_device = "cpu"

    # Error compensation only runs with torch backend + no rotation + calibration
    # activations. If a precondition is missing it would silently no-op while the
    # manifest still claimed it ran, so downgrade the flag here (keeps the manifest
    # truthful) and tell the user.
    if error_compensation and (
        backend != "torch" or rotation != "none" or awq_activations is None
    ):
        import sys as _sys

        print(
            "WARNING: --error-compensation needs --backend torch, --rotation none, "
            "and calibration activations; one is missing, so it will NOT be applied.",
            file=_sys.stderr,
        )
        error_compensation = False

    if tensor_partition_count == 1:
        tensor_partition_count = 1
        tensor_partition_index = 0

    if rotation == "orthogonal" and rotation_seed is None:
        rotation_seed = int.from_bytes(os.urandom(8), "little")

    # Mixed-Precision Sensitivity Logic
    skipped_tensors = set()
    if sensitivity_map is not None:
        for entry in sensitivity_map.get("layers", []):
            if (
                entry["loss_delta"] > SENSITIVITY_SKIP_LOSS_DELTA
                or "embed" in entry["layer"]
                or "lm_head" in entry["layer"]
            ):
                skipped_tensors.add(entry["layer"])

    src_sig = _source_signature(source)

    if family_stages_map is not None:
        if codebook_mode != "per-tensor":
            raise ValueError(
                "family_stages_map (mixed mode) requires codebook_mode='per-tensor'"
            )
        family_stages_resolved = {
            fam: [
                int(k) if not (isinstance(k, str) and k.startswith("s")) else k
                for k in stages
            ]
            for fam, stages in family_stages_map.items()
        }
        stages_spec = []
        n_stages = max(len(s) for s in family_stages_resolved.values())
    else:
        family_stages_resolved = None
        stages_spec = list(codebook_sizes) if codebook_sizes else [codebook_size]
        n_stages = len(stages_spec)
        if n_stages < 1:
            raise ValueError("at least one codebook stage is required")

    tensor_stages_resolved = None
    if tensor_stages_map is not None:
        if codebook_mode != "per-tensor":
            raise ValueError(
                "tensor_stages_map (measured allocation) requires codebook_mode='per-tensor'"
            )
        tensor_stages_resolved = {
            name: [
                int(k) if not (isinstance(k, str) and k.startswith("s")) else k
                for k in stages
            ]
            for name, stages in tensor_stages_map.items()
        }
        n_stages = max(
            n_stages, max(len(s) for s in tensor_stages_resolved.values())
        )

    tensor_transforms_resolved = None
    if tensor_transforms_map:
        if codebook_mode != "per-tensor":
            raise ValueError(
                "tensor_transforms_map (per-tensor normalization/rotation) requires "
                "codebook_mode='per-tensor'"
            )
        allowed = {"normalization", "rotation"}
        tensor_transforms_resolved = {
            name: {k: v for k, v in (over or {}).items() if k in allowed}
            for name, over in tensor_transforms_map.items()
        }

    tensor_dir = out_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "orka",
        "version": ORKA_VERSION,
        "source": str(source),
        "group_size": group_size,
        "requested_codebook_size": stages_spec[0]
        if (family_stages_resolved is None and n_stages == 1)
        else None,
        "codebook_sizes": list(stages_spec) if family_stages_resolved is None else None,
        "family_stages_map": family_stages_resolved,
        "tensor_allocation": tensor_stages_map is not None,
        "n_stages": n_stages,
        "codebook_mode": codebook_mode,
        # Per-tensor mode adapts group size by family; per-tensor entries carry
        # the resolved value. Top-level group_size is the requested baseline.
        "dynamic_group_sizing": codebook_mode == "per-tensor",
        "sample_vectors": sample_vectors,
        "backend": backend,
        "device": resolved_device,
        "normalization": normalization,
        "outlier_frac": outlier_frac,
        "rotation": rotation,
        "rotation_seed": rotation_seed,
        "awq_enabled": awq_activations is not None,
        "hessian_weighted": awq_activations is not None,
        "error_compensation": error_compensation,
        "mse_scale": mse_scale,
        "em_aq_passes": em_aq_passes,
        "slrq_salient": slrq_salient,
        "tensor_partition_count": (
            None if tensor_partition_count is None else tensor_partition_count
        ),
        "tensor_partition_index": (
            None if tensor_partition_index is None else tensor_partition_index
        ),
        "tensors": [],
    }

    candidates = []
    awq_fallbacks: list[str] = []
    _passthrough: dict[str, object] = {}

    # Prefetch Queue for Concurrency
    prefetch_queue = queue.Queue(maxsize=PREFETCH_QUEUE_DEPTH)
    prefetch_done = threading.Event()
    _prefetch_exc: list[BaseException] = []
    _prefetch_state = {"candidate_count": 0}

    streamed_tensor_count = 0

    # Per-tensor processing (stage loop + strategies + manifest entry) and the stage-spec
    # resolver live in orka.pipeline.pack_pipeline. Bundle the resolved config into a
    # PackCtx and hand each candidate to process_streamed_per_tensor_candidate below.
    ctx = PackCtx(
        backend=backend,
        n_stages=n_stages,
        codebook_mode=codebook_mode,
        group_size=group_size,
        resolved_device=resolved_device,
        out_dir=out_dir,
        tensor_dir=tensor_dir,
        sample_vectors=sample_vectors,
        progress_file=progress_file,
        iterations=iterations,
        em_aq_passes=em_aq_passes,
        block_scale_size=block_scale_size,
        rotation=rotation,
        rotation_seed=rotation_seed,
        mse_scale=mse_scale,
        src_sig=src_sig,
        skipped_tensors=skipped_tensors,
        codebook_dtype=codebook_dtype,
        codebook_cache_dir=codebook_cache_dir,
        awq_activations=awq_activations,
        outlier_frac=outlier_frac,
        normalization=normalization,
        max_tensors=max_tensors,
        error_compensation=error_compensation,
        stages_spec=stages_spec,
        family_stages_resolved=family_stages_resolved,
        tensor_stages_resolved=tensor_stages_resolved,
        tensor_transforms_resolved=tensor_transforms_resolved,
        manifest=manifest,
    )

    prefetch_thread = threading.Thread(
        target=_prefetch_worker,
        kwargs=dict(
            ctx=ctx,
            source=source,
            only_tensors=only_tensors,
            only_tensors_passthrough=only_tensors_passthrough,
            tensor_partition_count=tensor_partition_count,
            tensor_partition_index=tensor_partition_index,
            awq_alpha=awq_alpha,
            slrq_salient=slrq_salient,
            max_values_per_tensor=max_values_per_tensor,
            prefetch_queue=prefetch_queue,
            prefetch_done=prefetch_done,
            _prefetch_exc=_prefetch_exc,
            _prefetch_state=_prefetch_state,
            _passthrough=_passthrough,
            awq_fallbacks=awq_fallbacks,
        ),
        daemon=True,
    )
    prefetch_thread.start()

    while not prefetch_done.is_set() or not prefetch_queue.empty():
        if _prefetch_exc:
            break
        try:
            c = prefetch_queue.get(timeout=PREFETCH_POLL_TIMEOUT_S)
        except queue.Empty:
            continue
            
        _report_progress(progress_file, f"Prepared {c['name']} {c['shape']} (Ready for Quantization)")

        # --- Frequency-Aware Pillar Protection (SmolLM/Qwen research branch) ---
        is_embedding = c["name"].lower() in (
            "model.embed_tokens.weight", "gpt_neox.embed_in.weight", 
            "embed_out.weight", "lm_head.weight"
        )
        pillar_positions = None
        pillar_values = None

        if is_embedding and sensitivity_map and "top_tokens" in sensitivity_map:
            top_token_ids = sensitivity_map["top_tokens"]
            _report_progress(progress_file, f"    Applying Frequency-Aware Pillar Protection ({len(top_token_ids)} tokens)")

            vocab_size, hidden_dim = c["shape"][0], c["shape"][1]
            p_pos = []
            for tid in top_token_ids:
                if tid < vocab_size:
                    start = tid * hidden_dim
                    p_pos.extend(range(start, start + hidden_dim))

            if p_pos:
                from orka.core._format import _fp16_storage_roundtrip
                if _is_torch_tensor(c["vectors"]):
                    import torch
                    import numpy as np
                    flat = c["vectors"].reshape(-1)
                    pillar_positions = np.array(p_pos, dtype=np.int64)
                    pillar_values = _fp16_storage_roundtrip(
                        flat[pillar_positions].detach().cpu().numpy().astype(np.float32)
                    )
                    mask = torch.ones_like(flat)
                    mask[pillar_positions] = 0
                    c["vectors"] = (flat * mask).reshape(c["vectors"].shape)
                else:
                    import numpy as np
                    flat = c["vectors"].reshape(-1)
                    pillar_positions = np.array(p_pos, dtype=np.int64)
                    pillar_values = _fp16_storage_roundtrip(
                        flat[pillar_positions].astype(np.float32)
                    )
                    flat[pillar_positions] = 0
                    c["vectors"] = flat.reshape(c["vectors"].shape)

        # --- Standard Outlier Extraction ---
        # Salience-guided (h_col * w^2) when calibration importance is present,
        # magnitude otherwise.
        positions, values, new_vectors = _extract_outliers(
            c["vectors"], outlier_frac, c["packed_values"],
            col_importance=c.get("col_importance"),
            cols=int(c["shape"][-1]) if len(c["shape"]) > 1 else None,
        )
        # Round escape values to their fp16 storage grid so the metrics the
        # pipeline reports match what decode reads back from the sidecars.
        from orka.core._format import _fp16_storage_roundtrip
        values = _fp16_storage_roundtrip(values)

        if pillar_positions is not None:
            # Combine freq-aware pillars with magnitude outliers
            import numpy as np
            if positions is not None:
                c["outlier_positions"] = np.concatenate([pillar_positions, positions])
                c["outlier_values"] = np.concatenate([pillar_values, values])
            else:
                c["outlier_positions"] = pillar_positions
                c["outlier_values"] = pillar_values
        else:
            c["outlier_positions"] = positions
            c["outlier_values"] = values

        c["vectors"] = _offload(new_vectors)
        c["vectors_orig"] = c["vectors"]
        c["vectors_residual"] = c["vectors"]
        c["decoded_sum"] = None
        c["stages_meta"] = []
        # MSE/LS scale refinement needs the normalized weights, but they are freed
        # before finalize - stash a clean fp16 CPU copy now (per-tensor mode only).
        if mse_scale and codebook_mode == "per-tensor":
            _vv = c["vectors"]
            if _is_torch_tensor(_vv):
                c["_mse_v"] = _vv.detach().to("cpu").half().clone()
            else:
                import numpy as _np
                c["_mse_v"] = _np.asarray(_vv, dtype=_np.float16).copy()
        if codebook_mode == "per-tensor":
            streamed_tensor_count += 1
            process_streamed_per_tensor_candidate(ctx, c, streamed_tensor_count)
        else:
            candidates.append(c)
        prefetch_queue.task_done()

    prefetch_thread.join()

    if _prefetch_exc:
        exc = _prefetch_exc[0]
        # Preserve the original exception type when it carries semantic meaning
        # (e.g. SystemRAMExceededError, CappedOutOfMemoryError). Plain wrapping
        # to RuntimeError loses that signal for callers/CLI.
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise exc
        from orka._runtime import SystemRAMExceededError, CappedOutOfMemoryError
        if isinstance(exc, (SystemRAMExceededError, CappedOutOfMemoryError)):
            raise type(exc)(f"prefetch worker: {exc}") from exc
        raise RuntimeError(f"prefetch worker failed: {exc}") from exc
    if not candidates and streamed_tensor_count == 0 and tensor_partition_count is None:
        raise RuntimeError(
            "prefetch worker produced 0 candidates - no quantizable tensors found "
            "(check model path, tensor shapes, and device errors above)"
        )
    if not candidates and streamed_tensor_count == 0 and tensor_partition_count is not None:
        if _prefetch_state["candidate_count"] == 0:
            raise RuntimeError(
                "prefetch worker produced 0 candidates - no quantizable tensors found "
                "(check model path, tensor shapes, and device errors above)"
            )

    if _passthrough:
        passthrough_path = out_dir / "passthrough.safetensors"
        _write_passthrough_tensors(passthrough_path, _passthrough)
        manifest["passthrough_count"] = len(_passthrough)

    if codebook_mode == "per-tensor":
        _BG_WRITER.wait()
        manifest["total_index_bytes"] = sum(
            int(t.get("index_bytes", 0)) for t in manifest["tensors"]
        )
        manifest["tensor_count"] = len(manifest["tensors"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    return process_batched_candidates(ctx, candidates)
