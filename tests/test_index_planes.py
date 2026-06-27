"""Round-trip + density gate for the byte-aligned bit-plane index layout.

This is the VRAM-resident packing for arbitrary RVQ index widths (orka's non-traditional
moat: codebooks at native widths, not forced to 4/8-bit). The low plane is a coalesced
uint8 array; the high plane carries only bits above 8, so total storage is exactly
count*width bits. uint8 (width<=8) is the degenerate single-plane case.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from orka.core._format import _pack_index_planes, _unpack_index_planes


class IndexPlaneRoundTripTest(unittest.TestCase):
    def test_roundtrip_all_widths(self):
        rng = np.random.RandomState(0)
        for width in (8, 9, 10, 12, 14, 16):
            count = 1000
            idx = rng.randint(0, 2 ** width, size=count).astype(np.uint64)
            lo, hi = _pack_index_planes(idx, width)
            back = _unpack_index_planes(lo, hi, width, count)
            np.testing.assert_array_equal(back.astype(np.uint64), idx, err_msg=f"width={width}")

    def test_density_is_exact_width(self):
        # total bytes == ceil(count*width/8): no int16 padding waste
        count = 4096
        for width in (8, 10, 12, 16):
            idx = np.zeros(count, dtype=np.uint64)
            lo, hi = _pack_index_planes(idx, width)
            total_bits = (lo.nbytes + hi.nbytes) * 8
            self.assertEqual(total_bits, math.ceil(count * width / 8) * 8, f"width={width}")
            # vs int16 (16 bits): planes are width/16 of the footprint
            int16_bits = count * 16
            self.assertLessEqual(total_bits, int16_bits)

    def test_low_plane_is_coalesced_uint8(self):
        idx = np.arange(256, dtype=np.uint64)  # exercises low-byte values 0..255
        lo, hi = _pack_index_planes(idx, 12)
        self.assertEqual(lo.dtype, np.uint8)
        self.assertEqual(lo.shape[0], 256)
        np.testing.assert_array_equal(lo.astype(np.uint64), idx & 0xFF)

    def test_width8_has_empty_high_plane(self):
        idx = np.arange(256, dtype=np.uint64)
        lo, hi = _pack_index_planes(idx, 8)
        self.assertEqual(hi.size, 0)
        back = _unpack_index_planes(lo, hi, 8, 256)
        np.testing.assert_array_equal(back.astype(np.uint64), idx)


if __name__ == "__main__":
    unittest.main()
