"""SLRQ block normalization + salient-protected round-trip."""

import json
import tempfile
import unittest
from pathlib import Path

from orka import pack_checkpoint, verify_artifact


class SlrqTests(unittest.TestCase):
    def test_slrq_block_normalization_preserves_salient_weights(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception as exc:
            self.skipTest(f"torch required: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "slrq.json"
            out = root / "slrq.orka"
            data = [1.0] * 16
            data[5] = 100.0

            source.write_text(
                json.dumps({"tensors": {"linear.weight": [data]}})
            )

            manifest = pack_checkpoint(
                source=source,
                out_dir=out,
                group_size=16,
                codebook_size=2,
                iterations=1,
                backend="torch",
                device="cpu",
                normalization="slrq-block",
                block_scale_size=16,
            )

            verify = verify_artifact(out)

            tensor_meta = manifest["tensors"][0]
            self.assertIn("salient", tensor_meta)
            self.assertEqual(tensor_meta["salient"]["count"], 1)
            self.assertEqual(verify["verified_tensors"], 1)
            self.assertLess(verify["weighted_mse"], 1.0)

    def test_slrq_numpy_roundtrip(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "slrq_np.json"
            out = root / "slrq_np.orka"
            data = np.array([1.0] * 16, dtype=np.float32).reshape(2, 8)
            data[0, 5] = 100.0

            source.write_text(
                json.dumps({"tensors": {"linear.weight": data.tolist()}})
            )

            manifest = pack_checkpoint(
                source=source,
                out_dir=out,
                group_size=16,
                codebook_size=2,
                iterations=1,
                backend="numpy",
                normalization="slrq-block",
                block_scale_size=16,
            )

            verify = verify_artifact(out)

            tensor_meta = manifest["tensors"][0]
            self.assertIn("salient", tensor_meta)
            self.assertEqual(tensor_meta["salient"]["count"], 1)
            self.assertEqual(verify["verified_tensors"], 1)
            self.assertLess(verify["weighted_mse"], 1.0)


if __name__ == "__main__":
    unittest.main()
