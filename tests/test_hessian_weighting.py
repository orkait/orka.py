from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from orka.codebook import learn_codebook_auto
from orka.pipeline.pack import pack_checkpoint
from orka.verify import verify_artifact


class WeightedKMeansTest(unittest.TestCase):
    def test_numpy_sample_weights_pull_centroid(self) -> None:
        rows = np.asarray([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        cb, _, _ = learn_codebook_auto(
            rows, 1, 2, "numpy", sample_weights=np.asarray([1.0, 9.0], dtype=np.float32)
        )
        expected = (rows[0] * 1.0 + rows[1] * 9.0) / 10.0
        np.testing.assert_allclose(np.asarray(cb)[0], expected, rtol=1e-5)

    def test_numpy_uniform_weights_match_unweighted(self) -> None:
        rng = np.random.default_rng(0)
        rows = rng.standard_normal((64, 4)).astype(np.float32)
        cb_plain, _, _ = learn_codebook_auto(rows, 4, 4, "numpy", seed=7)
        cb_weighted, _, _ = learn_codebook_auto(
            rows, 4, 4, "numpy", seed=7,
            sample_weights=np.ones(64, dtype=np.float32),
        )
        np.testing.assert_allclose(cb_plain, cb_weighted, rtol=1e-5)

    def test_torch_sample_weights_pull_centroid(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        rows = torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.float32)
        cb, _, _ = learn_codebook_auto(
            rows, 1, 2, "torch", device="cpu",
            sample_weights=torch.tensor([1.0, 9.0]),
        )
        expected = (rows[0] * 1.0 + rows[1] * 9.0) / 10.0
        np.testing.assert_allclose(cb[0].numpy(), expected.numpy(), rtol=1e-5)


class HessianWeightedPackTest(unittest.TestCase):
    def _source(self, root: Path) -> Path:
        rng = np.random.default_rng(3)
        src = root / "model.json"
        src.write_text(
            json.dumps(
                {
                    "tensors": {
                        "model.layers.0.mlp.up_proj.weight": rng.standard_normal(
                            (4, 16)
                        ).round(3).tolist()
                    }
                }
            )
        )
        return src

    def test_pack_with_activations_round_trips(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._source(root)
            acts = {
                "model.layers.0.mlp.up_proj.weight": torch.linspace(
                    0.1, 4.0, 16
                ).repeat(8, 1)
            }
            manifest = pack_checkpoint(
                src,
                root / "w.orka",
                group_size=8,
                codebook_size=4,
                iterations=4,
                codebook_mode="per-tensor",
                backend="numpy",
                em_aq_passes=0,
                awq_activations=acts,
            )
            self.assertTrue(manifest["hessian_weighted"])
            verified = verify_artifact(root / "w.orka")
            self.assertEqual(verified["verified_tensors"], 1)
            self.assertLess(verified["max_mse_delta"], 1e-6)

    def test_activations_allowed_without_awq_gate(self) -> None:
        """Importance weighting must not require ORKA_ENABLE_AWQ."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        self.assertNotIn("ORKA_ENABLE_AWQ", os.environ)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._source(root)
            acts = {
                "model.layers.0.mlp.up_proj.weight": torch.ones(8, 16)
            }
            pack_checkpoint(
                src, root / "ok.orka", group_size=8, codebook_size=4,
                iterations=2, codebook_mode="per-tensor", backend="numpy",
                em_aq_passes=0, awq_activations=acts,
            )

    def test_awq_normalization_still_gated(self) -> None:
        self.assertNotIn("ORKA_ENABLE_AWQ", os.environ)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._source(root)
            with self.assertRaises(RuntimeError):
                pack_checkpoint(
                    src, root / "gated.orka", group_size=8, codebook_size=4,
                    iterations=2, codebook_mode="per-tensor", backend="numpy",
                    em_aq_passes=0, normalization="awq",
                )


if __name__ == "__main__":
    unittest.main()
