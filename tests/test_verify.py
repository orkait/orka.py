"""verify_artifact must reject what the decoder cannot read.

A killed pack leaves truncated sidecars behind. Packing over them keeps the stale bytes, and
the manifest still looks self-consistent because scale_bytes records what landed rather than
what was intended - so an artifact with 24 unreadable sidecars reached the GGUF exporter and
only failed there. These tests pin the gate that stops that.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from orka.artifact.verify import format_problems, verify_artifact  # noqa: E402
from orka.pipeline.pack import pack_checkpoint  # noqa: E402

DIM = 64


def _artifact(tmp: Path) -> Path:
    torch.manual_seed(0)
    sd = {
        "model.layers.0.mlp.w1.weight": torch.randn(DIM * 2, DIM),
        "model.layers.0.mlp.w2.weight": torch.randn(DIM, DIM * 2),
    }
    src = tmp / "m.pt"
    torch.save(sd, src)
    art = tmp / "a.orka"
    pack_checkpoint(source=src, out_dir=art, group_size=8,
                    codebook_sizes=[256, 256], codebook_mode="per-tensor",
                    backend="numpy", device="cpu", normalization="block-max",
                    sample_vectors=128, iterations=3, em_aq_passes=0,
                    only_tensors=list(sd), only_tensors_passthrough=True)
    return art


def _manifest(art: Path) -> dict:
    return json.loads((art / "manifest.json").read_text())


def test_intact_artifact_passes(tmp_path):
    r = verify_artifact(_artifact(tmp_path))
    assert r["ok"], format_problems(r)
    assert r["checked"] == r["tensors"] > 0
    assert "OK" in format_problems(r)


def test_truncated_scales_is_caught(tmp_path):
    """The exact shape of the LFM2.5-2.6B failure: a short block_max_scale sidecar."""
    art = _artifact(tmp_path)
    tm = next(t for t in _manifest(art)["tensors"] if t.get("scales"))
    p = art / tm["scales"]
    p.write_bytes(p.read_bytes()[: p.stat().st_size // 2])

    r = verify_artifact(art)
    assert not r["ok"]
    assert any(x["part"] == "scales" and x["name"] == tm["name"] for x in r["problems"])
    assert "BAD" in format_problems(r)


def test_corrupt_indices_is_caught(tmp_path):
    art = _artifact(tmp_path)
    tm = _manifest(art)["tensors"][0]
    stage = tm["stages"][0] if tm.get("stages") else tm
    p = art / stage["indices"]
    p.write_bytes(b"\x00" * 32)

    r = verify_artifact(art)
    assert not r["ok"]
    assert any(x["part"].startswith("stage") for x in r["problems"])


def test_missing_manifest_is_caught(tmp_path):
    art = _artifact(tmp_path)
    (art / "manifest.json").unlink()
    r = verify_artifact(art)
    assert not r["ok"]
    assert r["problems"][0]["part"] == "manifest.json"


def test_missing_sidecar_is_caught(tmp_path):
    art = _artifact(tmp_path)
    tm = _manifest(art)["tensors"][0]
    stage = tm["stages"][0] if tm.get("stages") else tm
    (art / stage["codebook"]).unlink()
    r = verify_artifact(art)
    assert not r["ok"]


def test_stop_after_bounds_the_scan(tmp_path):
    art = _artifact(tmp_path)
    for t in _manifest(art)["tensors"]:
        if t.get("scales"):
            p = art / t["scales"]
            p.write_bytes(p.read_bytes()[:8])
    r = verify_artifact(art, stop_after=1)
    assert not r["ok"]
    assert r["checked"] < r["tensors"] or r["tensors"] == 1
