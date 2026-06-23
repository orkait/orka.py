"""EM-AQ joint optimization: the post-assignment codebook refinement phase of packing.

For each tensor, decode the current multi-stage additive sum, then iterate
expectation-maximization over the additive quantizer: re-learn each stage's codebook
against the residual target (full sum minus that stage's contribution), re-quantize, and
update the running sum in place. Refined codebooks/indices are streamed to disk via the
background writer. Split out of pack.py so the refinement algorithm reads independently.
"""

from __future__ import annotations

from pathlib import Path

from orka._format import _cast_codebook_storage, _write_codebook, _write_indices
from orka._runtime import _BG_WRITER, _check_ram_cap
from orka._tensor import _decode_to_vectors_format, _is_torch_tensor, _vectors_subtract
from orka._util import _report_progress, _safe_tensor_name
from orka.codebook import learn_codebook_auto, quantize_vectors_auto
from orka.metrics import _stage_quality_metrics
from orka.pipeline.pack_helpers import _sample_vectors_and_weights


def _offload_to_cpu(t):
    """Move torch tensor to CPU; passthrough numpy/list."""
    if _is_torch_tensor(t):
        return t.detach().cpu()
    return t


def _run_em_aq_refinement(
    *,
    candidates: list,
    n_stages: int,
    skipped_tensors: set,
    sample_vectors: int | None,
    backend: str,
    resolved_device: str,
    tensor_dir: Path,
    progress_file: Path | None,
    em_aq_passes: int = 3,
) -> None:
    if em_aq_passes <= 0:
        _report_progress(progress_file, "--- EM-AQ disabled (em_aq_passes=0) ---")
    else:
        _report_progress(progress_file, "--- Starting Joint Optimization (EM-AQ) ---")
    joint_iterations = max(0, int(em_aq_passes))

    def _is_skipped(c: dict) -> bool:
        base = c["name"].replace(".weight", "")
        return base in skipped_tensors or c["name"] in skipped_tensors

    for i, c in enumerate(candidates):
        _check_ram_cap()
        if _is_skipped(c):
            continue

        c_n_stages = len(c["stages_data"])

        # 1. Decode current full_sum
        full_sum = None
        for stage_i in range(c_n_stages):
            sd = c["stages_data"][stage_i]
            s_group_size = sd.get("group_size", c["group_size"])
            t_template = c["vectors_orig"]
            if s_group_size == 1 and c["group_size"] > 1:
                t_template = c["vectors_orig"].reshape(-1, 1)

            dec = _decode_to_vectors_format(
                t_template, sd["cb"], sd["indices"], backend, resolved_device
            )
            if s_group_size == 1 and c["group_size"] > 1:
                dec = dec.reshape(c["vectors_orig"].shape)
            full_sum = dec if full_sum is None else full_sum + dec
            del dec

        current_full_sum = full_sum

        if joint_iterations > 0 and c_n_stages > 1:
            _report_progress(progress_file, f"  Joint Refining {c['name']} ({i+1}/{len(candidates)})...")
            # 2. EM-AQ Loop for this tensor
            for joint_iter in range(joint_iterations):
                for stage_i in range(c_n_stages):
                    _check_ram_cap()
                    sd = c["stages_data"][stage_i]
                    s_group_size = sd.get("group_size", c["group_size"])
                    k = c["stages_meta"][stage_i]["codebook_size"]

                    if sample_vectors is not None and k >= sample_vectors:
                        continue

                    t_template = c["vectors_orig"]
                    if s_group_size == 1 and c["group_size"] > 1:
                        t_template = c["vectors_orig"].reshape(-1, 1)

                    old_dec_raw = _decode_to_vectors_format(
                        t_template, sd["cb"], sd["indices"], backend, resolved_device
                    )
                    old_dec = old_dec_raw
                    if s_group_size == 1 and c["group_size"] > 1:
                        old_dec = old_dec_raw.reshape(c["vectors_orig"].shape)

                    target = _vectors_subtract(
                        c["vectors_orig"], (current_full_sum - old_dec)
                    )

                    target_train = target
                    if s_group_size == 1 and c["group_size"] > 1:
                        target_train = target.reshape(-1, 1)

                    sw = c.get("sample_weights") if s_group_size > 1 else None
                    training, sw_train = _sample_vectors_and_weights(
                        target_train, sw, sample_vectors
                    )
                    vw = c.get("vector_weights") if s_group_size > 1 else None

                    cb, _, _ = learn_codebook_auto(
                        training, min(k, len(training)), 2, backend, resolved_device,
                        vector_weights=vw, initial_codebook=sd["cb"],
                        sample_weights=sw_train,
                    )
                    stage_meta = c["stages_meta"][stage_i]
                    cb, _cb_dtype = _cast_codebook_storage(
                        cb, dtype=stage_meta.get("codebook_dtype", "float32")
                    )
                    stage_meta["codebook_dtype"] = _cb_dtype
                    indices, _ = quantize_vectors_auto(target_train, cb, backend, resolved_device, vector_weights=vw)

                    new_dec_raw = _decode_to_vectors_format(
                        target_train, cb, indices, backend, resolved_device
                    )
                    new_dec = new_dec_raw
                    if s_group_size == 1 and c["group_size"] > 1:
                        new_dec = new_dec_raw.reshape(c["vectors_orig"].shape)

                    if hasattr(current_full_sum, "sub_"):
                        # In-place PyTorch math avoids creating 3 massive intermediate tensors
                        current_full_sum.sub_(old_dec).add_(new_dec)
                    else:
                        current_full_sum = (current_full_sum - old_dec) + new_dec

                    c["stages_data"][stage_i] = {"cb": cb, "indices": indices, "group_size": s_group_size}

                    safe = _safe_tensor_name(c["name"])
                    cb_path = tensor_dir / f"{safe}.s{stage_i}.codebook.f32"
                    _BG_WRITER.submit(_write_codebook, cb_path, cb, _cb_dtype)
                    idx_path = tensor_dir / f"{safe}.s{stage_i}.indices"
                    _BG_WRITER.submit(
                        _write_indices,
                        idx_path,
                        indices.cpu() if hasattr(indices, "cpu") else indices,
                        stage_meta["index_bits"],
                        stage_meta.get("encoding", "raw"),
                    )

                    del old_dec_raw, old_dec, target, target_train, training, cb, indices, new_dec_raw, new_dec
                    import gc
                    gc.collect()

        c["decoded_sum"] = _offload_to_cpu(current_full_sum)
        del current_full_sum
        c["refined_metrics"] = _stage_quality_metrics(c, backend)
        c["source_flat"] = None
        c["decoded_sum"] = None

        for stage_i in range(c_n_stages):
            if "indices" in c["stages_data"][stage_i]:
                c["stages_data"][stage_i]["indices"] = None

        if "vectors_orig" in c:
            c["vectors_orig"] = None
        if "vectors" in c:
            c["vectors"] = None

        import gc
        gc.collect()
