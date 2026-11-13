from __future__ import annotations

import argparse
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orka.quant.activations import _bundled_calibration_path, _load_awq_activations


def _write_checkpoint(model_dir: Path, names: list[str]) -> Path:
    header = {
        name: {"dtype": "F32", "shape": [1, 1], "data_offsets": [4 * i, 4 * (i + 1)]}
        for i, name in enumerate(names)
    }
    raw = json.dumps(header).encode()
    path = model_dir / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(4 * len(names)))
    (model_dir / "config.json").write_text("{}")
    return path


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

    def test_activations_keyed_by_checkpoint_tensor_names(self) -> None:
        text_keys = {f"model.layers.{i}.mlp.down_proj.weight": i for i in range(2)}
        wrapped = {f"model.language_model.layers.{i}.mlp.down_proj.weight": i for i in range(2)}
        vlm_names = [*wrapped, "mtp.layers.0.mlp.down_proj.weight", "model.language_model.embed_tokens.weight"]
        explicit = {"awq_calibration": str(_bundled_calibration_path())}
        cases = {
            "multimodal wrapper prefix is remapped": (
                {**text_keys, "lm_head.weight": 9}, vlm_names, {}, wrapped,
            ),
            "explicit calibration is remapped too": (text_keys, vlm_names, explicit, wrapped),
            "matching names pass through": (text_keys, list(text_keys), {}, text_keys),
            "no overlap degrades to unweighted": ({"encoder.block.0.weight": 1}, vlm_names, {}, None),
        }
        for label, (collected, names, overrides, expected) in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                source = _write_checkpoint(Path(tmp), names)
                with mock.patch(
                    "orka.quant.activations._collect_activations_hf", return_value=collected
                ):
                    got = _load_awq_activations(
                        _args(backend="torch", source=str(source), **overrides)
                    )
                self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
