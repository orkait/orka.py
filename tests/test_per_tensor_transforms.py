"""Per-tensor transform overrides via the allocation map.

The pack pipeline applies one global normalization/rotation by default, but the
prep loop and manifest are per-tensor. These tests pin the override plumbing:
a tensor_transforms_map can give individual tensors their own normalization, the
manifest records it per tensor, decode still round-trips, and the override is
refused outside per-tensor codebook mode.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from orka.eval.verify import verify_artifact
from orka.pipeline.pack import pack_checkpoint
from orka.quant.allocate import allocation_tensor_transforms

A = "model.layers.0.mlp.up_proj.weight"
B = "model.layers.0.mlp.down_proj.weight"


def _write_src(root: Path) -> Path:
    rng = np.random.default_rng(7)
    src = root / "model.json"
    tensors = {
        A: rng.standard_normal((16, 32)).round(3).tolist(),
        B: rng.standard_normal((16, 32)).round(3).tolist(),
    }
    src.write_text(json.dumps({"tensors": tensors}))
    return src


def _pack(root, src, **kw):
    return pack_checkpoint(
        src, root / "out.orka", group_size=8, codebook_size=16,
        codebook_sizes=[16, 16], iterations=4, codebook_mode="per-tensor",
        backend="numpy", device="cpu", em_aq_passes=1, block_scale_size=32, **kw,
    )


def _norm_by_name(manifest):
    return {t["name"]: t.get("normalization") for t in manifest["tensors"]}


class PerTensorTransformTest(unittest.TestCase):
    def test_per_tensor_normalization_recorded_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_src(root)
            manifest = _pack(
                root, src,
                normalization="none",  # global default
                tensor_transforms_map={B: {"normalization": "block-max"}},
            )
            norms = _norm_by_name(manifest)
            # A keeps the global, B is overridden per tensor.
            self.assertEqual(norms[A], "none")
            self.assertEqual(norms[B], "block-max")
            # decode still matches the manifest exactly.
            verified = verify_artifact(root / "out.orka")
            self.assertLess(verified["max_mse_delta"], 1e-6)

    def test_default_no_map_uses_global(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_src(root)
            manifest = _pack(root, src, normalization="block-max")
            norms = _norm_by_name(manifest)
            self.assertEqual(norms[A], "block-max")
            self.assertEqual(norms[B], "block-max")

    def test_override_rejected_outside_per_tensor_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_src(root)
            with self.assertRaises(ValueError):
                pack_checkpoint(
                    src, root / "out.orka", group_size=8, codebook_size=16,
                    iterations=4, codebook_mode="global", backend="numpy",
                    device="cpu", tensor_transforms_map={B: {"normalization": "block-max"}},
                )

    def test_allocation_tensor_transforms_extracts_overrides(self) -> None:
        allocation = {"tensors": {
            A: {"stages": [16], "normalization": "none"},
            B: {"stages": [16], "normalization": "block-max", "rotation": "hadamard"},
            "c": {"stages": [16]},  # no transform fields -> omitted
        }}
        got = allocation_tensor_transforms(allocation)
        self.assertEqual(got, {
            A: {"normalization": "none"},
            B: {"normalization": "block-max", "rotation": "hadamard"},
        })


if __name__ == "__main__":
    unittest.main()
