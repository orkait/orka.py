from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from orka.core._format import (
    _pack_indices,
    _read_indices,
    _unpack_indices,
    _write_indices,
)
from orka.pipeline.pack import pack_checkpoint
from orka.eval.verify import verify_artifact


class IndexEncodingTest(unittest.TestCase):
    def test_zlib_round_trip_packed_and_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(0)
            # Skewed distribution compresses well.
            idx = rng.choice(8, size=4096, p=[0.7, 0.1, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01])
            for bits in (3, 8):
                path = root / f"i{bits}.indices"
                packed, encoding = _write_indices(path, idx, bits)
                self.assertEqual(encoding, "zlib")
                back = _read_indices(path, bits, len(idx), packed=packed, encoding=encoding)
                np.testing.assert_array_equal(np.asarray(back), idx)

    def test_incompressible_stays_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(1)
            idx = rng.integers(0, 256, size=4096)
            path = root / "r.indices"
            packed, encoding = _write_indices(path, idx, 8)
            self.assertEqual(encoding, "raw")
            back = _read_indices(path, 8, len(idx), packed=packed, encoding=encoding)
            np.testing.assert_array_equal(np.asarray(back), idx)

    def test_bitpack_unchanged(self) -> None:
        rng = np.random.default_rng(2)
        idx = rng.integers(0, 32, size=999)
        raw = _pack_indices(idx, 5)
        np.testing.assert_array_equal(_unpack_indices(raw, 5, 999), idx)


class FormatV2ArtifactTest(unittest.TestCase):
    def test_pack_records_dtypes_and_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "model.json"
            # Smooth, repetitive weights -> compressible index streams. Needs
            # enough vectors that zlib overhead amortizes.
            base = np.tile(np.linspace(-1, 1, 16, dtype=np.float32), (256, 1))
            src.write_text(
                json.dumps(
                    {"tensors": {"model.layers.0.mlp.up_proj.weight": base.tolist()}}
                )
            )
            artifact = root / "v2.orka"
            manifest = pack_checkpoint(
                src, artifact, group_size=4, codebook_size=4, iterations=4,
                codebook_mode="per-tensor", backend="numpy", em_aq_passes=0,
                normalization="block-max", block_scale_size=8,
            )
            self.assertEqual(manifest["version"], 3)
            entry = manifest["tensors"][0]
            self.assertEqual(entry["scale_dtype"], "float16")
            stage = entry["stages"][0]
            self.assertEqual(stage["codebook_dtype"], "float16")
            self.assertIn(stage["encoding"], ("zlib", "raw"))
            self.assertEqual(stage["encoding"], "zlib")
            # codebook file is half the f32 size (entries x group dims x 2 bytes)
            cb_path = artifact / stage["codebook"]
            self.assertEqual(
                cb_path.stat().st_size,
                stage["codebook_size"] * stage["group_size"] * 2,
            )

            verified = verify_artifact(artifact)
            self.assertLess(verified["max_mse_delta"], 1e-6)

    def test_em_aq_keeps_encoding_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "model.json"
            base = np.tile(np.linspace(-1, 1, 16, dtype=np.float32), (8, 1))
            src.write_text(
                json.dumps(
                    {"tensors": {"model.layers.0.mlp.up_proj.weight": base.tolist()}}
                )
            )
            artifact = root / "emaq.orka"
            manifest = pack_checkpoint(
                src, artifact, group_size=4, codebook_size=4,
                codebook_sizes=[4, 4], iterations=4,
                codebook_mode="per-tensor", backend="numpy", em_aq_passes=2,
            )
            for stage in manifest["tensors"][0]["stages"]:
                idx_path = artifact / stage["indices"]
                self.assertEqual(stage["index_bytes"], idx_path.stat().st_size)
            verified = verify_artifact(artifact)
            self.assertLess(verified["max_mse_delta"], 1e-6)


if __name__ == "__main__":
    unittest.main()
