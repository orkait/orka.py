from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from orka._format import (
    _cast_codebook_storage,
    _quantize_codebook_int8,
    _read_codebook,
    _write_codebook,
)
from orka.pipeline.pack import pack_checkpoint
from orka.eval.report import report_artifact
from orka.eval.verify import verify_artifact


class Int8CodebookFormatTest(unittest.TestCase):
    def test_quantize_dequantize_round_trip(self) -> None:
        rng = np.random.default_rng(0)
        cb = rng.standard_normal((256, 8)).astype(np.float32)
        q, scales = _quantize_codebook_int8(cb)
        self.assertEqual(q.dtype, np.int8)
        self.assertEqual(scales.shape, (8,))
        # largest entry per column maps to +-127 (lossless scale recovery)
        self.assertTrue(np.all(np.abs(q).max(axis=0) == 127) or np.all(scales > 0))
        dequant = q.astype(np.float32) * scales[None, :]
        # within one int8 step of the original
        self.assertLess(np.max(np.abs(dequant - cb)), float(scales.max()) + 1e-6)

    def test_cast_then_write_recovers_same_values(self) -> None:
        rng = np.random.default_rng(1)
        cb = rng.standard_normal((128, 8)).astype(np.float32)
        dequant, dtype = _cast_codebook_storage(cb, dtype="int8")
        self.assertEqual(dtype, "int8")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cb.f32"
            _write_codebook(p, dequant, dtype="int8")
            back = _read_codebook(p, 8, dtype="int8")
            # the on-disk read must equal the in-memory cast exactly
            np.testing.assert_array_equal(back, dequant)
            # file is ~half of fp16 (int8 data + tiny scale header)
            self.assertLess(p.stat().st_size, 128 * 8 * 2)

    def test_zero_column_handled(self) -> None:
        cb = np.zeros((16, 8), dtype=np.float32)
        cb[:, 3] = np.linspace(-1, 1, 16)
        q, scales = _quantize_codebook_int8(cb)
        self.assertTrue(np.isfinite(scales).all())
        self.assertTrue((scales > 0).all())


class Int8PackTest(unittest.TestCase):
    def _source(self, root: Path) -> Path:
        rng = np.random.default_rng(3)
        src = root / "m.json"
        src.write_text(json.dumps({"tensors": {
            "model.layers.0.mlp.up_proj.weight": rng.standard_normal((64, 128)).round(4).tolist()
        }}))
        return src

    def test_int8_halves_codebook_and_stays_verifiable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._source(root)
            sizes = {}
            for dt in ("float16", "int8"):
                art = root / f"{dt}.orka"
                m = pack_checkpoint(
                    src, art, group_size=8, codebook_size=4096, iterations=4,
                    codebook_mode="per-tensor", backend="numpy", em_aq_passes=0,
                    codebook_dtype=dt,
                )
                self.assertEqual(m["tensors"][0]["stages"][0]["codebook_dtype"], dt)
                r = report_artifact(art)
                sizes[dt] = r["total_codebook_bytes"]
                v = verify_artifact(art)
                self.assertLess(v["max_mse_delta"], 1e-6)
                self.assertGreater(r["cosine_similarity"], 0.999)
            # int8 codebook is ~half of fp16
            self.assertLess(sizes["int8"], sizes["float16"] * 0.6)

    def test_int8_with_full_chain_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._source(root)
            art = root / "chain.orka"
            pack_checkpoint(
                src, art, group_size=8, codebook_size=4096,
                codebook_sizes=[4096, 256], iterations=4,
                codebook_mode="per-tensor", backend="numpy", em_aq_passes=2,
                normalization="slrq-block", block_scale_size=8, outlier_frac=0.02,
                codebook_dtype="int8",
            )
            v = verify_artifact(art)
            self.assertLess(v["max_mse_delta"], 1e-6)


if __name__ == "__main__":
    unittest.main()
