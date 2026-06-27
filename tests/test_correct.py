from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from orka.pipeline.pack import pack_checkpoint
from orka.eval.report import report_artifact
from orka.eval.verify import verify_artifact


def _write_source(root: Path) -> Path:
    rng = np.random.default_rng(21)
    src = root / "model.json"
    # Low-rank structure + noise: rank-r correction has real signal to recover.
    u = rng.standard_normal((32, 3))
    v = rng.standard_normal((3, 32))
    w = (u @ v + 0.05 * rng.standard_normal((32, 32))).round(4)
    src.write_text(
        json.dumps(
            {"tensors": {"model.layers.0.mlp.up_proj.weight": w.tolist()}}
        )
    )
    return src


@unittest.skipUnless(HAS_TORCH, "torch required for correction")
class CorrectTest(unittest.TestCase):
    def test_correction_reduces_error_and_stays_verifiable(self) -> None:
        from orka.artifact.correct import correct_artifact

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_source(root)
            artifact = root / "c.orka"
            manifest = pack_checkpoint(
                src, artifact, group_size=8, codebook_size=4, iterations=4,
                codebook_mode="per-tensor", backend="numpy", em_aq_passes=0,
            )
            before = manifest["tensors"][0]["mse"]

            result = correct_artifact(artifact, rank=4, device="cpu")
            self.assertEqual(result["tensor_count"], 1)
            self.assertEqual(result["improved_count"], 1)
            row = result["results"][0]
            self.assertLess(row["mse_after"], before)

            updated = json.loads((artifact / "manifest.json").read_text())
            entry = updated["tensors"][0]
            self.assertEqual(entry["lowrank"]["rank"], 4)
            self.assertLess(entry["mse"], before)

            verified = verify_artifact(artifact)
            self.assertLess(verified["max_mse_delta"], 1e-6)

            report = report_artifact(artifact)
            self.assertGreater(report["artifact_bytes"], 0)

    def test_correction_composes_with_distill_and_transforms(self) -> None:
        from orka.artifact.correct import correct_artifact
        from orka.qat.distill import distill_artifact

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_source(root)
            artifact = root / "cd.orka"
            pack_checkpoint(
                src, artifact, group_size=8, codebook_size=4,
                codebook_sizes=[4, 4], iterations=2,
                codebook_mode="per-tensor", backend="numpy", em_aq_passes=0,
                normalization="block-max", block_scale_size=8,
                outlier_frac=0.02,
            )
            distill_artifact(artifact, steps=80, lr=0.05, device="cpu")
            mid = json.loads((artifact / "manifest.json").read_text())["tensors"][0]["mse"]

            result = correct_artifact(artifact, rank=4, device="cpu")
            self.assertEqual(result["improved_count"], 1)
            final = json.loads((artifact / "manifest.json").read_text())["tensors"][0]["mse"]
            self.assertLess(final, mid)

            verified = verify_artifact(artifact)
            self.assertLess(verified["max_mse_delta"], 1e-6)

    def test_rerun_replaces_rather_than_stacks(self) -> None:
        from orka.artifact.correct import correct_artifact

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_source(root)
            artifact = root / "rr.orka"
            pack_checkpoint(
                src, artifact, group_size=8, codebook_size=4, iterations=4,
                codebook_mode="per-tensor", backend="numpy", em_aq_passes=0,
            )
            first = correct_artifact(artifact, rank=4, device="cpu")
            second = correct_artifact(artifact, rank=4, device="cpu")
            mse_first = first["results"][0]["mse_after"]
            mse_second = second["results"][0]["mse_after"]
            # svd_lowrank uses random (unseeded) projections, so rerun mse differs
            # by a small noise floor (~1e-3). Stacking instead of replacing would
            # change mse by orders more, so a loose tolerance still proves replace.
            self.assertAlmostEqual(mse_first, mse_second, places=2)
            verified = verify_artifact(artifact)
            self.assertLess(verified["max_mse_delta"], 1e-6)


if __name__ == "__main__":
    unittest.main()
