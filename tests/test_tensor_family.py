from __future__ import annotations

import unittest

from orka.quant import classify_tensor_family


class TensorFamilyClassificationTest(unittest.TestCase):
    def test_gate_proj_is_mlp_not_router(self) -> None:
        self.assertEqual(
            classify_tensor_family("model.layers.0.mlp.gate_proj.weight"),
            "mlp",
        )
        self.assertEqual(
            classify_tensor_family("model.layers.0.mlp.up_proj.weight"),
            "mlp",
        )
        self.assertEqual(
            classify_tensor_family("model.layers.0.mlp.down_proj.weight"),
            "mlp",
        )

    def test_router_names_still_classify_as_router(self) -> None:
        self.assertEqual(
            classify_tensor_family("model.layers.0.router.weight"),
            "router",
        )
        self.assertEqual(
            classify_tensor_family("model.layers.0.block_sparse_moe.gate.weight"),
            "router",
        )


if __name__ == "__main__":
    unittest.main()
