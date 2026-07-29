"""Row-lazy VQ embedding must be a drop-in for the dense decode it replaces.

An embedding is a lookup weight, so serving it row-by-row from the packed payload should
produce EXACTLY what the reference decoder produces for those rows - anything less and the
memory saving is not free. Eligibility is also pinned, because silently serving an
ineligible tensor (scalar stages, outlier sidecars, a dim that straddles groups) would
corrupt rows rather than fail.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from orka.inference.vq_embedding import VQEmbedding, can_serve  # noqa: E402
from orka.pipeline.decode import _decode_tensor  # noqa: E402
from orka.pipeline.pack import pack_checkpoint  # noqa: E402

VOCAB, DIM = 384, 64


def _packed(tmp: Path, **overrides):
    torch.manual_seed(0)
    src = tmp / "m.pt"
    torch.save({"embed_in.weight": torch.randn(VOCAB, DIM)}, src)
    art = tmp / "a.orka"
    kwargs = dict(group_size=8, codebook_sizes=[256, 256], codebook_mode="per-tensor",
                  backend="numpy", device="cpu", normalization="block-max",
                  sample_vectors=256, iterations=4, em_aq_passes=0)
    kwargs.update(overrides)
    manifest = pack_checkpoint(source=src, out_dir=art, **kwargs)
    return art, manifest["tensors"][0]


def test_rows_are_bit_exact_against_the_reference_decoder():
    with tempfile.TemporaryDirectory() as tmp:
        art, meta = _packed(Path(tmp))
        assert can_serve(meta) == (True, "")
        ref = np.asarray(_decode_tensor(art, meta), dtype=np.float32).reshape(VOCAB, DIM)
        emb = VQEmbedding.from_artifact(art, meta)
        got = emb(torch.arange(VOCAB)).numpy()
        assert np.array_equal(got, ref), (
            f"row-lazy decode diverged from the reference decoder "
            f"(max {np.abs(got - ref).max():.3e})"
        )


def test_arbitrary_row_subsets_and_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        art, meta = _packed(Path(tmp))
        ref = np.asarray(_decode_tensor(art, meta), dtype=np.float32).reshape(VOCAB, DIM)
        emb = VQEmbedding.from_artifact(art, meta)

        ids = torch.randint(0, VOCAB, (29,))
        assert np.array_equal(emb(ids).numpy(), ref[ids.numpy()])

        # repeated ids must not interfere, and 2-D input keeps its shape
        rep = torch.tensor([7, 7, 7, 0, VOCAB - 1])
        assert np.array_equal(emb(rep).numpy(), ref[rep.numpy()])
        assert emb(torch.randint(0, VOCAB, (3, 5))).shape == (3, 5, DIM)


def test_resident_payload_is_smaller_than_dense():
    with tempfile.TemporaryDirectory() as tmp:
        art, meta = _packed(Path(tmp))
        emb = VQEmbedding.from_artifact(art, meta)
        r = emb.resident_bytes()
        assert r["total"] < r["dense_fp16_equivalent"], "no saving means no reason to exist"
        # the index stream must not be widened to int64: it is the bulk of the payload
        for name, buf in emb.named_buffers():
            if name.startswith("indices_"):
                assert buf.element_size() <= 2, f"{name} stored as {buf.dtype}"


def test_ineligible_tensors_are_refused_not_mangled():
    # a dim that is not a whole number of groups would make rows straddle group boundaries
    bad = {"name": "e.weight", "shape": [16, 12], "group_size": 8, "block_scale_size": 32,
           "stages": []}
    ok, why = can_serve(bad)
    assert not ok and "group_size" in why

    # position-indexed corrections need their own scatter
    with tempfile.TemporaryDirectory() as tmp:
        art, meta = _packed(Path(tmp), outlier_frac=0.05)
        ok, why = can_serve(meta)
        assert not ok and "outliers" in why
        with pytest.raises(ValueError, match="outliers"):
            VQEmbedding.from_artifact(art, meta)


def test_scalar_stage_layout_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        art, meta = _packed(Path(tmp), codebook_sizes=None,
                            tensor_stages_map={"embed_in.weight": [256, "s8"]})
        ok, why = can_serve(meta)
        assert not ok and "scalar" in why


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu():
    with tempfile.TemporaryDirectory() as tmp:
        art, meta = _packed(Path(tmp))
        cpu = VQEmbedding.from_artifact(art, meta, device="cpu")
        gpu = VQEmbedding.from_artifact(art, meta, device="cuda")
        ids = torch.randint(0, VOCAB, (17,))
        assert np.array_equal(cpu(ids).numpy(), gpu(ids.cuda()).cpu().numpy())


def test_repeated_ids_preserve_order_and_stay_bit_exact():
    """Decoding is deduplicated to unique rows, so the scatter back to positions must
    preserve ORDER, not merely membership - a wrong inverse map would silently permute
    every batch that repeats a token, which real text does constantly."""
    with tempfile.TemporaryDirectory() as tmp:
        art, meta = _packed(Path(tmp))
        emb = VQEmbedding.from_artifact(art, meta)
        ref = np.asarray(_decode_tensor(art, meta), dtype=np.float32).reshape(VOCAB, DIM)

        ids = torch.tensor([5, 5, 3, 5, 0, 3, VOCAB - 1, 0])
        got = emb(ids).numpy()
        assert np.array_equal(got, ref[ids.numpy()])

        # unsorted ids must not come back sorted
        pair = emb(torch.tensor([9, 2])).numpy()
        assert np.array_equal(pair[0], ref[9])
        assert np.array_equal(pair[1], ref[2])


def test_dedup_holds_across_shapes_including_all_identical():
    with tempfile.TemporaryDirectory() as tmp:
        art, meta = _packed(Path(tmp))
        emb = VQEmbedding.from_artifact(art, meta)
        ref = np.asarray(_decode_tensor(art, meta), dtype=np.float32).reshape(VOCAB, DIM)
        for ids in (torch.full((4, 6), 11), torch.randint(0, VOCAB, (3, 5, 2))):
            assert np.array_equal(emb(ids).numpy(), ref[ids.numpy()])
