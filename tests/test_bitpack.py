"""Bit-packing of codebook indices: round-trip + size guarantees."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from orka._format import (
    _pack_indices,
    _read_indices,
    _unpack_indices,
    _write_indices,
)


class BitPackTests(unittest.TestCase):
    def test_roundtrip_all_widths(self) -> None:
        rng = np.random.default_rng(0)
        for bits in range(1, 17):
            for count in (0, 1, 7, 8, 9, 1000):
                hi = 1 << bits
                idx = rng.integers(0, hi, size=count, dtype=np.int64)
                packed = _pack_indices(idx, bits)
                out = _unpack_indices(packed, bits, count)
                self.assertEqual(len(out), count, f"count bits={bits} count={count}")
                np.testing.assert_array_equal(
                    np.asarray(out, dtype=np.int64), idx,
                    err_msg=f"roundtrip bits={bits} count={count}",
                )

    def test_max_values_preserved(self) -> None:
        for bits in (1, 3, 8, 10, 12, 16):
            maxval = (1 << bits) - 1
            idx = np.array([0, maxval, maxval, 0, maxval], dtype=np.int64)
            out = _unpack_indices(_pack_indices(idx, bits), bits, len(idx))
            np.testing.assert_array_equal(np.asarray(out, dtype=np.int64), idx)

    def test_packed_smaller_for_odd_widths(self) -> None:
        # 10-bit index in 1000 vectors: fixed uint16 = 2000 bytes; packed = ceil(10000/8) = 1250.
        idx = np.zeros(1000, dtype=np.int64)
        packed = _pack_indices(idx, 10)
        self.assertEqual(len(packed), (1000 * 10 + 7) // 8)
        self.assertLess(len(packed), 1000 * 2)


    def test_write_indices_packs_odd_width_and_reads_back(self) -> None:
        rng = np.random.default_rng(1)
        idx = rng.integers(0, 1 << 10, size=500, dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.indices"
            packed = _write_indices(p, idx, 10)
            self.assertTrue(packed, "10-bit indices should be bit-packed")
            self.assertEqual(p.stat().st_size, (500 * 10 + 7) // 8)
            out = _read_indices(p, 10, 500, packed=True)
            np.testing.assert_array_equal(np.asarray(out, dtype=np.int64), idx)

    def test_write_indices_fixed_width_for_byte_aligned(self) -> None:
        rng = np.random.default_rng(2)
        idx = rng.integers(0, 256, size=500, dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.indices"
            packed = _write_indices(p, idx, 8)
            self.assertFalse(packed, "8-bit indices stay fixed-width uint8")
            self.assertEqual(p.stat().st_size, 500)
            out = _read_indices(p, 8, 500, packed=False)
            np.testing.assert_array_equal(np.asarray(out, dtype=np.int64), idx)

    def test_pack_decode_roundtrip_with_odd_bit_stage(self) -> None:
        """Full pack -> verify with a 10-bit stage (rvq-10-8). Decode must unpack
        the bit-packed stage-0 indices, or reconstruction is garbage."""
        import json
        from orka import pack_checkpoint, verify_artifact

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "m.json"
            out = root / "m.orka"
            rng = np.random.default_rng(3)
            # >1024 vectors (>8192 values at group 8) so stage-0 k=1024 -> 10-bit indices
            weights = rng.standard_normal((128, 128)).astype(np.float32).tolist()
            source.write_text(json.dumps({"tensors": {"linear.weight": weights}}))

            manifest = pack_checkpoint(
                source=source, out_dir=out,
                group_size=8, codebook_sizes=[1024, 256],
                iterations=4, backend="numpy", em_aq_passes=0,
            )
            stage0 = manifest["tensors"][0]["stages"][0]
            self.assertEqual(stage0["index_bits"], 10)
            self.assertTrue(stage0.get("packed"), "10-bit stage must record packed=True")

            result = verify_artifact(out)
            self.assertEqual(result["verified_tensors"], 1)
            # decode must match the manifest's own recorded mse (proves correct unpack)
            self.assertLess(result["max_mse_delta"], 1e-6)


if __name__ == "__main__":
    unittest.main()
