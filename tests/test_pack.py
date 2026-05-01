"""Inspect + pack round-trips for safetensors and torch backends."""

import json
import tempfile
import unittest
from pathlib import Path

from orka import inspect_checkpoint, pack_checkpoint, verify_artifact


class InspectPackTests(unittest.TestCase):
    def test_safetensors_bfloat16_checkpoint_can_be_inspected(self) -> None:
        try:
            import torch
            from safetensors.torch import save_file
        except Exception as exc:
            self.skipTest(f"optional BF16 safetensors dependencies missing: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bf16.safetensors"
            save_file(
                {
                    "model.layers.0.mlp.down_proj.weight": torch.ones(
                        (2, 4), dtype=torch.bfloat16
                    )
                },
                str(path),
            )

            report = inspect_checkpoint(path)

            self.assertEqual(report["total_params"], 8)
            candidate_params = sum(
                tensor["numel"] for tensor in report["tensors"] if tensor["candidate"]
            )
            self.assertEqual(candidate_params, 8)
            self.assertEqual(report["tensors"][0]["shape"], [2, 4])

    def test_safetensors_bfloat16_checkpoint_can_be_packed_with_numpy(self) -> None:
        try:
            import torch
            from safetensors.torch import save_file
            import numpy  # noqa: F401
        except Exception as exc:
            self.skipTest(f"optional BF16 packing dependencies missing: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "bf16.safetensors"
            out = root / "bf16.orka"
            save_file(
                {
                    "model.layers.0.mlp.down_proj.weight": torch.arange(
                        8, dtype=torch.float32
                    )
                    .reshape(2, 4)
                    .to(torch.bfloat16)
                },
                str(source),
            )

            manifest = pack_checkpoint(
                source=source,
                out_dir=out,
                group_size=2,
                codebook_size=2,
                iterations=2,
                backend="numpy",
            )
            verify = verify_artifact(out)

            self.assertEqual(manifest["tensor_count"], 1)
            self.assertEqual(verify["verified_tensors"], 1)

    def test_torch_backend_packs_and_records_device(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception as exc:
            self.skipTest(f"torch backend dependency missing: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "tiny.json"
            out = root / "tiny-torch.orka"
            source.write_text(
                json.dumps(
                    {
                        "tensors": {
                            "linear.weight": [
                                [1.0, 1.0, 0.9, 1.1],
                                [-1.0, -1.0, -1.1, -0.9],
                            ],
                        }
                    }
                )
            )

            manifest = pack_checkpoint(
                source=source,
                out_dir=out,
                group_size=2,
                codebook_size=2,
                iterations=3,
                backend="torch",
                device="cpu",
            )
            verify = verify_artifact(out)

            self.assertEqual(manifest["backend"], "torch")
            self.assertEqual(manifest["device"], "cpu")
            self.assertEqual(manifest["tensor_count"], 1)
            self.assertEqual(verify["verified_tensors"], 1)


if __name__ == "__main__":
    unittest.main()
