from __future__ import annotations

import argparse
import unittest

from orka.quant.activations import _bundled_calibration_path, _load_awq_activations


def _args(**kw):
    base = dict(
        awq_activations_file=None,
        awq_calibration=None,
        awq_model_dir=None,
        normalization="slrq-block",
        no_hessian=False,
        source="x.safetensors",
        backend="numpy",
        device="cpu",
        calibration_max_prompts=32,
        calibration_max_length=256,
        calibration_max_samples=4096,
    )
    base.update(kw)
    return argparse.Namespace(**base)


class DefaultHessianTest(unittest.TestCase):
    def test_bundled_calibration_ships(self) -> None:
        p = _bundled_calibration_path()
        self.assertTrue(p.exists(), f"bundled calibration missing: {p}")
        self.assertGreater(len(p.read_text().strip().splitlines()), 8)

    def test_no_hessian_opts_out(self) -> None:
        # Explicit opt-out short-circuits to unweighted, no collection attempted.
        self.assertIsNone(_load_awq_activations(_args(no_hessian=True)))

    def test_non_torch_backend_degrades_unweighted(self) -> None:
        # numpy backend can't run the HF forward; must degrade (warn) to None.
        self.assertIsNone(_load_awq_activations(_args(backend="numpy")))

    def test_missing_model_config_degrades_unweighted(self) -> None:
        # torch backend but no config.json next to the source => graceful None.
        self.assertIsNone(
            _load_awq_activations(_args(backend="torch", source="/tmp/orka_nonexistent.safetensors"))
        )


if __name__ == "__main__":
    unittest.main()
