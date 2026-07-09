"""CPU round-trip lock for build_vq_linear / VQLinear.reconstruct_weight.

Packs a tiny synthetic model to a temp .orka, builds the VQLinear via the
public factory, and asserts the layer's reconstruction matches the artifact's
ground-truth decode (orka.pipeline.decode._decode_tensor, the same path
verify_artifact uses). This locks build_vq_linear's behavior so the mechanical
split that follows is provably non-altering: the test passes before and after.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from orka.inference.vq_linear import VQLinear, build_vq_linear
from orka.pipeline.decode import _decode_tensor
from orka.pipeline.pack import pack_checkpoint


def _write_source(path: Path) -> None:
    """4x16 tensor (total=64). group_size=8 + block=8 divide 16 evenly, so the
    group-major transpose path in build_vq_linear is exercised too."""
    rows = lambda v: [[float(v + i + j) for j in range(16)] for i in range(4)]
    path.write_text(
        json.dumps(
            {
                "tensors": {
                    "model.layers.0.self_attn.q_proj.weight": rows(1),
                    "model.layers.0.mlp.up_proj.weight": rows(2),
                }
            }
        )
    )


def _pack(root: Path, **overrides) -> dict:
    source = root / "model.json"
    _write_source(source)
    kwargs = dict(
        group_size=8,
        codebook_size=4,
        iterations=2,
        codebook_mode="per-tensor",
        backend="numpy",
        em_aq_passes=0,
        block_scale_size=8,
    )
    kwargs.update(overrides)
    return pack_checkpoint(source, root / "out.orka", **kwargs)


class BuildVQLinearRoundTripTest(unittest.TestCase):
    def test_reconstruct_matches_artifact_decode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _pack(root, normalization="slrq-block")
            artifact_dir = root / "out.orka"

            by_name = {t["name"]: t for t in manifest["tensors"]}
            self.assertEqual(len(by_name), 2)

            for name, meta in by_name.items():
                shape = [int(x) for x in meta["shape"]]
                out_features, in_features = shape[0], int(np.prod(shape[1:]))

                layer = build_vq_linear(
                    artifact_dir=artifact_dir,
                    tensor_meta=meta,
                    bias=None,
                    device="cpu",
                )

                # --- structural assertions ---
                self.assertIsInstance(layer, VQLinear)
                self.assertEqual(layer.out_features, out_features)
                self.assertEqual(layer.in_features, in_features)
                self.assertEqual(layer.group_size, int(meta["group_size"]))
                n_stages = len(meta.get("stages") or [meta])
                self.assertEqual(layer.n_stages, n_stages)

                # --- behavioral lock: reconstruct vs ground-truth decode ---
                recon = layer.reconstruct_weight()
                self.assertEqual(tuple(recon.shape), (out_features, in_features))
                self.assertTrue(torch.isfinite(recon).all())

                ground_truth = np.asarray(_decode_tensor(artifact_dir, meta), dtype=np.float32)
                gt = torch.from_numpy(ground_truth[: out_features * in_features]).reshape(
                    out_features, in_features
                )

                # reconstruct_weight stores fp16 buffers (codebooks/scales/corr), so
                # match within fp16 round-trip tolerance, not bit-exact fp32.
                diff = (recon - gt).abs()
                self.assertLess(
                    float(diff.max()),
                    5e-2 + 1e-2 * float(gt.abs().max()),
                    f"{name}: reconstruct deviates from decode (max abs diff {float(diff.max())})",
                )

                rel_mse = float((diff.pow(2).mean()) / (gt.pow(2).mean() + 1e-12))
                self.assertTrue(np.isfinite(rel_mse))
                self.assertLess(rel_mse, 1e-2, f"{name}: relMSE too high ({rel_mse})")

                # Deterministic CPU reconstruct+matmul forward must run + stay finite.
                # The Triton/CUDA dispatch kernel hard-requires block_size=32, which
                # this tiny 4x16 artifact (block=8) does not satisfy, so call the
                # python fallback directly.
                x = torch.randn(3, in_features)
                y = layer._forward_python(x)
                self.assertEqual(tuple(y.shape), (3, out_features))
                self.assertTrue(torch.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
