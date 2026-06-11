from __future__ import annotations

import unittest

import numpy as np

from orka.transforms.outliers import _extract_outliers


class SalienceOutlierTest(unittest.TestCase):
    def test_salience_overrides_magnitude(self) -> None:
        """A modest weight on a hot column must beat a large weight on a
        dead column when importance is supplied."""
        cols = 4
        w = np.zeros((2, cols), dtype=np.float32)
        w[0, 0] = 10.0   # large weight, dead column
        w[1, 2] = 2.0    # modest weight, hot column
        h = np.array([1e-6, 1e-6, 100.0, 1e-6], dtype=np.float32)

        pos_mag, _, _ = _extract_outliers(w.reshape(-1, 2).copy(), 0.13, w.size)
        self.assertIn(0, pos_mag)  # magnitude picks the big dead weight

        pos_sal, val_sal, kept = _extract_outliers(
            w.reshape(-1, 2).copy(), 0.13, w.size, col_importance=h, cols=cols
        )
        self.assertIn(6, pos_sal)  # flat index of w[1, 2]
        self.assertNotIn(0, pos_sal)
        self.assertAlmostEqual(float(val_sal[0]), 2.0)
        self.assertEqual(float(kept.reshape(-1)[6]), 0.0)

    def test_importance_shape_mismatch_falls_back_to_magnitude(self) -> None:
        w = np.arange(16, dtype=np.float32).reshape(-1, 2)
        pos, _, _ = _extract_outliers(
            w.copy(), 0.1, 16, col_importance=np.ones(5, dtype=np.float32), cols=4
        )
        self.assertIn(15, pos)  # largest magnitude

    def test_torch_path_matches_numpy(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        rng = np.random.default_rng(0)
        w = rng.standard_normal((8, 16)).astype(np.float32)
        h = rng.random(16).astype(np.float32)
        pos_np, _, _ = _extract_outliers(w.reshape(-1, 4).copy(), 0.05, w.size,
                                         col_importance=h, cols=16)
        pos_t, _, _ = _extract_outliers(
            torch.from_numpy(w.reshape(-1, 4).copy()), 0.05, w.size,
            col_importance=torch.from_numpy(h), cols=16)
        self.assertEqual(set(pos_np.tolist()), set(pos_t.tolist()))


if __name__ == "__main__":
    unittest.main()
