from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from orka.eval.verify import verify_artifact
from orka.pipeline.pack import pack_checkpoint


def _write_source(root: Path) -> Path:
    rng = np.random.default_rng(11)
    src = root / "model.json"
    src.write_text(
        json.dumps(
            {
                "tensors": {
                    "model.layers.0.mlp.up_proj.weight": rng.standard_normal((8, 16))
                    .round(3)
                    .tolist(),
                    "model.layers.0.self_attn.q_proj.weight": rng.standard_normal(
                        (8, 16)
                    )
                    .round(3)
                    .tolist(),
                }
            }
        )
    )
    return src


@unittest.skipUnless(HAS_TORCH, "torch required for distillation")
class DistillTest(unittest.TestCase):
    def test_fwht_autograd_matches_numpy(self) -> None:
        from orka.qat.distill import _fwht_autograd
        from orka.transforms.rotate import _fwht_numpy

        rng = np.random.default_rng(5)
        x = rng.standard_normal((3, 16)).astype(np.float32)
        expected = _fwht_numpy(x)
        got = _fwht_autograd(torch.from_numpy(x.copy())).numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_distill_improves_multistage_rvq(self) -> None:
        """Greedy RVQ stages are jointly suboptimal, so joint codebook
        optimization must strictly reduce error. (Single-stage plain-MSE is
        already optimal given indices - Lloyd centroids ARE cluster means -
        so RVQ is the meaningful case.)"""
        from orka.qat.distill import distill_artifact

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_source(root)
            artifact = root / "model.orka"
            manifest = pack_checkpoint(
                src,
                artifact,
                group_size=4,
                codebook_size=4,
                codebook_sizes=[4, 4],
                iterations=2,
                codebook_mode="per-tensor",
                backend="numpy",
                em_aq_passes=0,
            )
            before = {t["name"]: t["mse"] for t in manifest["tensors"]}

            result = distill_artifact(artifact, steps=200, lr=0.05, device="cpu")
            self.assertEqual(result["tensor_count"], 2)
            self.assertEqual(result["improved_count"], 2)
            for row in result["results"]:
                self.assertLess(row["mse"], before[row["name"]])
                self.assertLessEqual(row["final_loss"], row["initial_loss"])

            verified = verify_artifact(artifact)
            self.assertEqual(verified["verified_tensors"], 2)
            self.assertLess(verified["max_mse_delta"], 1e-6)

    def test_distill_never_worse_single_stage(self) -> None:
        from orka.qat.distill import distill_artifact

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_source(root)
            artifact = root / "single.orka"
            pack_checkpoint(
                src, artifact, group_size=4, codebook_size=4, iterations=2,
                codebook_mode="per-tensor", backend="numpy", em_aq_passes=0,
            )
            result = distill_artifact(artifact, steps=80, lr=0.05, device="cpu")
            for row in result["results"]:
                self.assertLessEqual(row["final_loss"], row["initial_loss"])
            verified = verify_artifact(artifact)
            self.assertLess(verified["max_mse_delta"], 1e-6)

    def test_distill_full_transform_chain(self) -> None:
        """block-max + orthogonal rotation + outliers: differentiable mirror must
        match the production decoder exactly (verify max_mse_delta ~ 0)."""
        from orka.qat.distill import distill_artifact

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_source(root)
            artifact = root / "chain.orka"
            pack_checkpoint(
                src,
                artifact,
                group_size=4,
                codebook_size=4,
                iterations=2,
                codebook_mode="per-tensor",
                backend="numpy",
                em_aq_passes=0,
                normalization="block-max",
                block_scale_size=8,
                rotation="orthogonal",
                rotation_seed=42,
                outlier_frac=0.05,
            )
            result = distill_artifact(artifact, steps=100, lr=0.05, device="cpu")
            self.assertEqual(result["tensor_count"], 2)
            self.assertGreaterEqual(result["improved_count"], 1)
            verified = verify_artifact(artifact)
            self.assertLess(verified["max_mse_delta"], 1e-6)

    def test_distill_with_activation_weighting(self) -> None:
        from orka.qat.distill import distill_artifact

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_source(root)
            artifact = root / "wdist.orka"
            pack_checkpoint(
                src, artifact, group_size=4, codebook_size=4, iterations=2,
                codebook_mode="per-tensor", backend="numpy", em_aq_passes=0,
            )
            acts = {
                "model.layers.0.mlp.up_proj.weight": torch.linspace(0.1, 4.0, 16).repeat(6, 1),
                "model.layers.0.self_attn.q_proj.weight": torch.ones(6, 16),
            }
            result = distill_artifact(
                artifact, steps=120, lr=0.05, device="cpu", activations=acts
            )
            self.assertEqual(result["tensor_count"], 2)
            manifest = json.loads((artifact / "manifest.json").read_text())
            self.assertTrue(manifest["distilled"]["weighted"])
            verified = verify_artifact(artifact)
            self.assertLess(verified["max_mse_delta"], 1e-6)


if __name__ == "__main__":
    unittest.main()
