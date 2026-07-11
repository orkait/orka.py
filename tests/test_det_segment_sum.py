"""Determinism + correctness gates for the fused segmented-sum Lloyd update."""
from __future__ import annotations

import unittest

try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def _cuda_triton_available() -> bool:
    if not HAS_TORCH or not torch.cuda.is_available():
        return False
    from orka.codebook._segsum_kernel import _HAVE_TRITON

    return _HAVE_TRITON


@unittest.skipUnless(HAS_TORCH, "torch required")
class DetSegmentSumCpuTest(unittest.TestCase):
    def test_cpu_fallback_matches_index_add(self) -> None:
        from orka.codebook._kmeans_torch import _det_segment_sum

        g = torch.Generator().manual_seed(7)
        rows = torch.randn(10_000, 8, generator=g)
        keys = torch.randint(0, 64, (10_000,), generator=g)
        sums, lengths = _det_segment_sum(keys, rows, 64)
        ref = torch.zeros(64, 8, dtype=torch.float64).index_add_(0, keys, rows.double())
        self.assertTrue(torch.allclose(sums.double(), ref, rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.equal(lengths, torch.bincount(keys, minlength=64)))

    def test_empty_keys(self) -> None:
        from orka.codebook._kmeans_torch import _det_segment_sum

        sums, lengths = _det_segment_sum(
            torch.zeros(0, dtype=torch.long), torch.zeros(0, 8), 16
        )
        self.assertEqual(tuple(sums.shape), (16, 8))
        self.assertEqual(int(lengths.sum()), 0)


@unittest.skipUnless(_cuda_triton_available(), "CUDA + triton required")
class DetSegmentSumCudaTest(unittest.TestCase):
    def _case(self, n, d, k, seed=0):
        from orka.codebook._kmeans_torch import _det_segment_sum

        g = torch.Generator(device="cuda").manual_seed(seed)
        rows = torch.randn(n, d, device="cuda", generator=g)
        keys = torch.randint(0, k, (n,), device="cuda", generator=g)
        return keys, rows, _det_segment_sum(keys, rows, k)

    def test_deterministic_run_to_run(self) -> None:
        from orka.codebook._kmeans_torch import _det_segment_sum

        keys, rows, (sums_a, len_a) = self._case(500_000, 8, 256)
        sums_b, len_b = _det_segment_sum(keys, rows, 256)
        self.assertTrue(torch.equal(sums_a, sums_b))
        self.assertTrue(torch.equal(len_a, len_b))

    def test_matches_fp64_reference(self) -> None:
        for n, d, k in ((500_000, 8, 256), (200_000, 4, 16), (100_000, 16, 65536)):
            keys, rows, (sums, lengths) = self._case(n, d, k, seed=k)
            ref = torch.zeros(k, d, dtype=torch.float64, device="cuda").index_add_(
                0, keys, rows.double()
            )
            self.assertTrue(
                torch.allclose(sums.double(), ref, rtol=1e-4, atol=1e-4),
                f"mismatch at n={n} d={d} k={k}",
            )
            self.assertTrue(torch.equal(lengths, torch.bincount(keys, minlength=k)))

    def test_one_dimensional_values(self) -> None:
        from orka.codebook._kmeans_torch import _det_segment_sum

        g = torch.Generator(device="cuda").manual_seed(3)
        sw = torch.rand(300_000, device="cuda", generator=g)
        keys = torch.randint(0, 256, (300_000,), device="cuda", generator=g)
        sums, _ = _det_segment_sum(keys, sw, 256)
        self.assertEqual(tuple(sums.shape), (256,))
        ref = torch.zeros(256, dtype=torch.float64, device="cuda").index_add_(
            0, keys, sw.double()
        )
        self.assertTrue(torch.allclose(sums.double(), ref, rtol=1e-5, atol=1e-5))

    def test_skewed_and_empty_clusters(self) -> None:
        from orka.codebook._kmeans_torch import _det_segment_sum

        g = torch.Generator(device="cuda").manual_seed(11)
        rows = torch.randn(400_000, 8, device="cuda", generator=g)
        # heavy skew + guaranteed-empty clusters
        keys = (torch.rand(400_000, device="cuda", generator=g).pow(4) * 128).long()
        sums, lengths = _det_segment_sum(keys, rows, 256)
        self.assertTrue(bool((lengths[128:] == 0).all()))
        self.assertTrue(bool((sums[128:] == 0).all()))
        ref = torch.zeros(256, 8, dtype=torch.float64, device="cuda").index_add_(
            0, keys, rows.double()
        )
        self.assertTrue(torch.allclose(sums.double(), ref, rtol=1e-4, atol=1e-4))

    def test_learn_codebook_reproducible_per_seed(self) -> None:
        from orka.codebook._kmeans_torch import _learn_codebook_torch

        g = torch.Generator(device="cuda").manual_seed(21)
        rows = torch.randn(200_000, 8, device="cuda", generator=g)
        cb_a, idx_a, mse_a = _learn_codebook_torch(rows, 64, 4, "cuda", seed=42)
        cb_b, idx_b, mse_b = _learn_codebook_torch(rows, 64, 4, "cuda", seed=42)
        self.assertTrue(torch.equal(cb_a, cb_b))
        self.assertTrue(torch.equal(idx_a, idx_b))
        self.assertEqual(mse_a, mse_b)


if __name__ == "__main__":
    unittest.main()
