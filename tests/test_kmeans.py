import unittest


class KMeansTests(unittest.TestCase):
    def test_numpy_centroid_sums_match_add_at_accumulation(self) -> None:
        import numpy as np

        from orka.codebook.kmeans import _numpy_centroid_sums

        rows = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [-1.0, -2.0, -3.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([2, 0, 2, 1], dtype=np.int64)
        expected = np.zeros((4, 3), dtype=np.float32)
        np.add.at(expected, indices, rows)

        actual = _numpy_centroid_sums(rows, indices, 4)

        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_numpy_assign_reuses_cached_row_norms_without_changing_results(self) -> None:
        import numpy as np

        from orka.codebook.kmeans import _numpy_assign

        rng = np.random.default_rng(123)
        rows = rng.normal(size=(257, 7)).astype(np.float32)
        codebook = rng.normal(size=(19, 7)).astype(np.float32)
        row_norm_sq = np.sum(rows * rows, axis=1, dtype=np.float32)

        expected_indices, expected_mse = _numpy_assign(rows, codebook, chunk_size=31)
        actual_indices, actual_mse = _numpy_assign(
            rows,
            codebook,
            chunk_size=31,
            r_norm_sq=row_norm_sq,
        )

        np.testing.assert_array_equal(actual_indices, expected_indices)
        self.assertAlmostEqual(actual_mse, expected_mse, places=7)


if __name__ == "__main__":
    unittest.main()
