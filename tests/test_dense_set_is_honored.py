"""Tensors the role rails mark dense must NOT be quantized.

waterfill_with_roles returns (stages, dense). Passing only `stages` as tensor_stages_map
silently loses the dense set: pack_checkpoint quantizes every eligible tensor and falls back
to its default spec for anything absent from the map. On LFM2.5-2.6B that put all 22 depthwise
conv kernels - the sharpest tensors in the model by Fisher curvature, explicitly given an fp16
prior - at 1.0 bpw in three consecutive packs, and every quality number measured through them
was invalid.

The fix is only_tensors + only_tensors_passthrough=True. These tests pin it.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from orka.pipeline.pack import pack_checkpoint  # noqa: E402

VOCAB, DIM = 128, 64


def _src(tmp: Path) -> Path:
    torch.manual_seed(0)
    sd = {
        "model.layers.0.mlp.w1.weight": torch.randn(DIM * 2, DIM),
        "model.layers.0.mlp.w2.weight": torch.randn(DIM, DIM * 2),
        "model.layers.0.conv.conv.weight": torch.randn(DIM, 1, 3),   # depthwise: must stay dense
    }
    p = tmp / "m.pt"
    torch.save(sd, p)
    return p


def _pack(tmp: Path, only: list[str] | None):
    art = tmp / "a.orka"
    pack_checkpoint(source=_src(tmp), out_dir=art, group_size=8,
                    codebook_sizes=[256, 256], codebook_mode="per-tensor",
                    backend="numpy", device="cpu", normalization="block-max",
                    sample_vectors=128, iterations=3, em_aq_passes=0,
                    only_tensors=only, only_tensors_passthrough=True)
    return art


def test_without_only_tensors_the_dense_tensor_gets_quantized():
    """Reproduces the bug: omitting only_tensors quantizes everything eligible."""
    with tempfile.TemporaryDirectory() as tmp:
        art = _pack(Path(tmp), None)
        names = {t["name"] for t in json.loads((art / "manifest.json").read_text())["tensors"]}
        assert "model.layers.0.conv.conv.weight" in names, \
            "expected the buggy path to quantize the depthwise kernel"


def test_only_tensors_keeps_the_dense_set_out_of_the_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        keep = ["model.layers.0.mlp.w1.weight", "model.layers.0.mlp.w2.weight"]
        art = _pack(Path(tmp), keep)
        names = {t["name"] for t in json.loads((art / "manifest.json").read_text())["tensors"]}
        assert "model.layers.0.conv.conv.weight" not in names
        assert names == set(keep)


def test_dense_tensor_is_preserved_bit_exact_in_passthrough():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        keep = ["model.layers.0.mlp.w1.weight", "model.layers.0.mlp.w2.weight"]
        art = _pack(tmp, keep)
        pt = art / "passthrough.safetensors"
        assert pt.exists(), "dense tensors must be written to passthrough"
        from safetensors.torch import load_file
        got = load_file(str(pt))
        assert "model.layers.0.conv.conv.weight" in got
        want = torch.load(tmp / "m.pt")["model.layers.0.conv.conv.weight"]
        assert np.array_equal(got["model.layers.0.conv.conv.weight"].float().numpy(),
                              want.float().numpy())
