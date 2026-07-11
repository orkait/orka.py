"""EM-AQ must skip giant tensors: its first step alone re-materializes the full
decoded sum (+4GB per stage on the 1B head) and every pass re-assigns the whole
tensor. The greedy stage-loop result stands, and its decoded_sum must survive
the em_aq cleanup so the finalize metrics still have something to measure.

Also locks the giant path's byte parity: with EM-AQ off, a pack forced through
the giant path is byte-identical to the normal path."""
import json

import pytest

torch = pytest.importorskip("torch")

from orka.codebook import _kmeans_torch  # noqa: E402
from orka.pipeline.pack import pack_checkpoint  # noqa: E402
from orka.pipeline.strategies.refinement import _run_em_aq_refinement  # noqa: E402


def _checkpoint(tmp_path):
    src = tmp_path / "model.pt"
    torch.manual_seed(3)
    torch.save({"layer.weight": torch.randn(64, 16)}, src)
    return src


def _pack(src, out, em_aq_passes):
    return pack_checkpoint(
        source=src, out_dir=out, group_size=8, codebook_sizes=[16, 16],
        codebook_mode="per-tensor", backend="torch", device="cpu",
        normalization="slrq-block", sample_vectors=32, iterations=4,
        em_aq_passes=em_aq_passes,
    )


def _tensor_entry(manifest):
    return manifest["tensors"][0]


def test_refinement_skips_giant_candidate(monkeypatch, tmp_path):
    monkeypatch.setattr(_kmeans_torch, "_LARGE_ASSIGN_ROWS", 4)
    vo = torch.randn(40, 8)
    idx = torch.zeros(40, dtype=torch.int64)
    cb = torch.randn(4, 8)
    c = {
        "name": "layer.weight", "vectors_orig": vo, "group_size": 8,
        "stages_data": {0: {"cb": cb, "indices": idx, "group_size": 8}},
        "stages_meta": [{"codebook_size": 4, "index_bits": 2, "codebook_dtype": "float16"}],
    }
    _run_em_aq_refinement(
        candidates=[c], n_stages=2, skipped_tensors=set(), sample_vectors=None,
        backend="torch", resolved_device="cpu", tensor_dir=tmp_path,
        progress_file=None, em_aq_passes=1,
    )
    assert c.get("refined_metrics") is None
    assert c["stages_data"][0]["indices"] is idx
    assert c["vectors_orig"] is vo


def test_giant_pack_multistage_with_emaq_completes(monkeypatch, tmp_path):
    """Regression: cleanup popped decoded_sum at the last stage expecting EM-AQ to
    rebuild it; with EM-AQ skipping giants, finalize metrics need the greedy sum."""
    monkeypatch.setattr(_kmeans_torch, "_LARGE_ASSIGN_ROWS", 16)
    src = _checkpoint(tmp_path)
    manifest = _pack(src, tmp_path / "giant.orka", em_aq_passes=1)
    t = _tensor_entry(manifest)
    assert t["sqnr"] > 0 and t["mse"] >= 0

    base = _pack(src, tmp_path / "giant0.orka", em_aq_passes=0)
    assert _tensor_entry(base)["mse"] == t["mse"], "giant EM-AQ run must equal greedy"


def test_giant_path_byte_parity_without_emaq(monkeypatch, tmp_path):
    src = _checkpoint(tmp_path)
    normal_dir = tmp_path / "normal.orka"
    _pack(src, normal_dir, em_aq_passes=0)
    monkeypatch.setattr(_kmeans_torch, "_LARGE_ASSIGN_ROWS", 16)
    giant_dir = tmp_path / "giant.orka"
    _pack(src, giant_dir, em_aq_passes=0)

    for sub in sorted(p.relative_to(giant_dir) for p in giant_dir.rglob("*") if p.is_file()):
        if sub.name == "manifest.json":
            gm = json.loads((giant_dir / sub).read_text())
            nm = json.loads((normal_dir / sub).read_text())
            assert _tensor_entry(gm)["mse"] == _tensor_entry(nm)["mse"]
            assert _tensor_entry(gm)["sqnr"] == _tensor_entry(nm)["sqnr"]
            continue
        assert (giant_dir / sub).read_bytes() == (normal_dir / sub).read_bytes(), sub
