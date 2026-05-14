"""Payload size estimation and parameter parsing."""

import unittest

from orka import _parse_params, estimate_payload


class PayloadEstimateTests(unittest.TestCase):
    def test_calc_size_for_vq8_eight_b_model(self) -> None:
        estimate = estimate_payload(
            params=8_030_000_000, group_size=8, codebook_size=256
        )
        self.assertEqual(estimate.index_bits, 8)
        self.assertEqual(estimate.vector_count, 1_003_750_000)
        self.assertEqual(estimate.index_bytes, 1_003_750_000)
        self.assertAlmostEqual(estimate.bits_per_weight, 1.0)

    def test_parse_decimal_param_suffix_without_float_rounding(self) -> None:
        self.assertEqual(_parse_params("8.03b"), 8_030_000_000)

    def test_calc_size_for_vq16_two_byte_indices(self) -> None:
        estimate = estimate_payload(
            params=8_030_000_000, group_size=8, codebook_size=8192
        )
        self.assertEqual(estimate.index_bits, 13)
        self.assertEqual(estimate.vector_count, 1_003_750_000)
        self.assertEqual(estimate.index_bytes, 1_631_093_750)


if __name__ == "__main__":
    unittest.main()
