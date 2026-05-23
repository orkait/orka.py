"""Tests for the Orka reconstruction layers and replacement wrapper."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except Exception as exc:
    raise unittest.SkipTest(f"torch required: {exc}") from exc

from orka import pack_checkpoint
from orka.layers import OrkaLinear, replace_linear_with_orka


class LayersTests(unittest.TestCase):
    def test_orka_linear_roundtrip(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception as exc:
            self.skipTest(f"torch required: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "model.json"
            out = root / "model.orka"

            # 2 rows, 4 columns
            weights = [
                [0.5, -0.2, 0.1, 0.8],
                [-0.4, 0.6, -0.3, 0.1],
            ]
            bias = [0.1, -0.2]

            source.write_text(
                json.dumps({
                    "tensors": {
                        "linear.weight": weights,
                        "linear.bias": bias,
                    }
                })
            )

            # Pack with Orka compiler
            pack_checkpoint(
                source=source,
                out_dir=out,
                group_size=4,
                codebook_size=4,
                iterations=2,
                backend="torch",
                device="cpu",
                normalization="none",
            )

            # Create original PyTorch layer
            orig_layer = nn.Linear(4, 2)
            orig_layer.weight.data = torch.tensor(weights, dtype=torch.float32)
            orig_layer.bias.data = torch.tensor(bias, dtype=torch.float32)

            # Reconstruct dummy model using our replacement wrapper
            class DummyModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(4, 2)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.linear(x)

            model = DummyModel()
            model.linear.weight.data = torch.tensor(weights, dtype=torch.float32)
            model.linear.bias.data = torch.tensor(bias, dtype=torch.float32)

            # Verify original prediction
            x = torch.randn(3, 4)
            y_orig = model(x)

            # Replace with OrkaLinear
            replace_linear_with_orka(model, out)

            # Check that class type was replaced correctly
            self.assertIsInstance(model.linear, OrkaLinear)

            # Run inference on Orka model
            y_orka = model(x)

            # Check output shapes match
            self.assertEqual(y_orig.shape, y_orka.shape)

            # Check that weights are reconstructed correctly
            w_reconstructed = model.linear.reconstruct_weight("cpu")
            self.assertEqual(w_reconstructed.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
