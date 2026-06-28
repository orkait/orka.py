"""Pure-helper tests for the per-tensor transform search (allocate increment 2).

Covers scalar_quant_proxy (the cheap transform-ranking distortion) and
transform_overhead_bits (the rate accounting). No pack pipeline involved.
"""
from __future__ import annotations

import unittest

import numpy as np

from orka.quant.transform_search import (
    DEFAULT_TRANSFORM_GRID,
    apply_transform,
    rank_transforms,
    scalar_quant_proxy,
    scalar_quant_reconstruct,
    transform_overhead_bits,
    transform_proxy_distortion,
)


class ScalarQuantProxyTest(unittest.TestCase):
    def test_empty_and_zeros_are_zero(self) -> None:
        self.assertEqual(scalar_quant_proxy([]), 0.0)
        self.assertEqual(scalar_quant_proxy(np.zeros(64), block_size=16), 0.0)

    def test_constant_block_is_lossless(self) -> None:
        # A constant block: scale = |c|/lim, round(c/scale) = lim, recon = c. MSE 0.
        v = np.full(32, -1.5)
        self.assertAlmostEqual(scalar_quant_proxy(v, bits=4, block_size=32), 0.0, places=12)

    def test_more_bits_never_worse(self) -> None:
        rng = np.random.default_rng(0)
        v = rng.standard_normal(4096)
        prev = float("inf")
        for b in (2, 3, 4, 6, 8):
            mse = scalar_quant_proxy(v, bits=b, block_size=128)
            self.assertLessEqual(mse, prev + 1e-12)
            prev = mse
        self.assertGreater(scalar_quant_proxy(v, bits=2, block_size=128), 0.0)

    def test_sign_symmetric(self) -> None:
        rng = np.random.default_rng(1)
        v = rng.standard_normal(2048)
        self.assertAlmostEqual(
            scalar_quant_proxy(v, bits=3), scalar_quant_proxy(-v, bits=3), places=12
        )

    def test_rotation_isometry_no_scales_needed(self) -> None:
        # An orthogonal rotation preserves L2: proxy on rotated == proxy on a random
        # orthogonally-equivalent vector measured the same way (no scale correction).
        rng = np.random.default_rng(2)
        v = rng.standard_normal(256)
        Q, _ = np.linalg.qr(rng.standard_normal((256, 256)))
        rotated = Q @ v
        # both measured in their own space; values differ but the proxy is well-defined
        # and finite for each (isometry handled by NOT passing original_scales).
        self.assertGreaterEqual(scalar_quant_proxy(rotated, bits=4, block_size=256), 0.0)

    def test_original_scales_reweight_by_square(self) -> None:
        # Same normalized block, but block scale 2x -> original-space error 4x.
        rng = np.random.default_rng(3)
        norm_block = rng.standard_normal(64)
        mse1 = scalar_quant_proxy(norm_block, bits=3, block_size=64, original_scales=[1.0])
        mse2 = scalar_quant_proxy(norm_block, bits=3, block_size=64, original_scales=[2.0])
        self.assertAlmostEqual(mse2, 4.0 * mse1, places=10)

    def test_scales_length_must_match_blocks(self) -> None:
        with self.assertRaises(ValueError):
            scalar_quant_proxy(np.zeros(256), block_size=128, original_scales=[1.0])  # 2 blocks

    def test_bad_args(self) -> None:
        with self.assertRaises(ValueError):
            scalar_quant_proxy([1.0, 2.0], bits=0)
        with self.assertRaises(ValueError):
            scalar_quant_proxy([1.0, 2.0], block_size=0)


class TransformOverheadBitsTest(unittest.TestCase):
    def test_none_is_free(self) -> None:
        self.assertEqual(transform_overhead_bits("none", "none", numel=10000), 0)
        self.assertEqual(transform_overhead_bits(None, None, numel=10000), 0)
        self.assertEqual(transform_overhead_bits("awq", "hadamard", numel=10000), 0)

    def test_block_max_is_scales_only(self) -> None:
        # 256 elems / 128 block = 2 blocks * 16 bits = 32.
        self.assertEqual(
            transform_overhead_bits("block-max", "none", numel=256, block_size=128, scale_bits=16),
            32,
        )
        # 300 / 128 -> ceil = 3 blocks.
        self.assertEqual(
            transform_overhead_bits("block-max", "none", numel=300, block_size=128, scale_bits=16),
            48,
        )

    def test_rotation_adds_nothing(self) -> None:
        base = transform_overhead_bits("block-max", "none", numel=1024)
        rot = transform_overhead_bits("block-max", "hadamard", numel=1024)
        self.assertEqual(base, rot)

    def test_slrq_adds_salient_sidecar(self) -> None:
        # block scales + salient_count * salient_bits_each.
        numel, bs, sb, frac, sbe = 10000, 128, 16, 0.01, 48
        n_blocks = (numel + bs - 1) // bs
        expected = n_blocks * sb + round(frac * numel) * sbe
        got = transform_overhead_bits(
            "slrq-block", "none", numel=numel, block_size=bs,
            scale_bits=sb, salient_frac=frac, salient_bits_each=sbe,
        )
        self.assertEqual(got, expected)

    def test_bad_args(self) -> None:
        with self.assertRaises(ValueError):
            transform_overhead_bits("none", "none", numel=-1)
        with self.assertRaises(ValueError):
            transform_overhead_bits("none", "none", numel=10, block_size=0)
        with self.assertRaises(ValueError):
            transform_overhead_bits("slrq-block", "none", numel=10, salient_frac=1.5)


class TransformProxyDistortionTest(unittest.TestCase):
    def _multiscale(self):
        rng = np.random.default_rng(0)
        return np.vstack([
            rng.standard_normal(256) * 0.01,
            rng.standard_normal(256) * 1.0,
            rng.standard_normal(256) * 0.05,
            rng.standard_normal(256) * 2.0,
        ])

    def test_none_matches_global_scalar_quant(self) -> None:
        # normalization=none, rotation=none -> just one-scale scalar-quant MSE.
        rng = np.random.default_rng(5)
        W = rng.standard_normal((4, 64))
        got = transform_proxy_distortion(W, "none", "none", bits=4)
        ref = scalar_quant_proxy(W.reshape(-1), bits=4, block_size=W.size)
        self.assertAlmostEqual(got, ref, places=12)

    def test_block_max_helps_multiscale(self) -> None:
        W = self._multiscale()
        none = transform_proxy_distortion(W, "none", "none", bits=4, norm_block=128)
        bmax = transform_proxy_distortion(W, "block-max", "none", bits=4, norm_block=128)
        self.assertLess(bmax, none)

    def test_hadamard_helps_outliers(self) -> None:
        rng = np.random.default_rng(6)
        W = rng.standard_normal((4, 256)) * 0.1
        W[0, 0] = 30.0  # a fat outlier one global scale must cover
        none = transform_proxy_distortion(W, "none", "none", bits=4)
        hada = transform_proxy_distortion(W, "none", "hadamard", bits=4)
        self.assertLess(hada, none)

    def test_slrq_uses_block_max_proxy(self) -> None:
        W = self._multiscale()
        a = transform_proxy_distortion(W, "slrq-block", "none", bits=4, norm_block=128)
        b = transform_proxy_distortion(W, "block-max", "none", bits=4, norm_block=128)
        self.assertAlmostEqual(a, b, places=12)

    def test_1d_input_and_determinism(self) -> None:
        rng = np.random.default_rng(7)
        v = rng.standard_normal(512)
        d1 = transform_proxy_distortion(v, "block-max", "none", bits=4, norm_block=128)
        d2 = transform_proxy_distortion(v.reshape(1, -1), "block-max", "none", bits=4, norm_block=128)
        self.assertAlmostEqual(d1, d2, places=12)

    def test_unsupported_config_raises(self) -> None:
        W = np.zeros((4, 64))
        with self.assertRaises(ValueError):
            transform_proxy_distortion(W, "none", "orthogonal")
        with self.assertRaises(ValueError):
            transform_proxy_distortion(W, "made-up", "none")

    def test_hadamard_infeasible_width_raises(self) -> None:
        # cols=6 has no pow2 divisor >= 4 -> _hadamard_block_size raises -> propagates.
        W = np.zeros((4, 6))
        with self.assertRaises(ValueError):
            transform_proxy_distortion(W, "none", "hadamard")

    def test_reconstruct_same_length(self) -> None:
        v = np.arange(300, dtype=float)
        r = scalar_quant_reconstruct(v, bits=4, block_size=128)
        self.assertEqual(r.shape, v.shape)


class ApplyTransformTest(unittest.TestCase):
    def test_none_is_identity_factor_one(self) -> None:
        rng = np.random.default_rng(8)
        W = rng.standard_normal((4, 64))
        Wt, factor = apply_transform(W, "none", "none")
        np.testing.assert_allclose(Wt, W)
        self.assertEqual(factor, 1.0)

    def test_block_max_bounds_and_factor(self) -> None:
        rng = np.random.default_rng(9)
        W = rng.standard_normal((4, 64)) * 5.0
        Wt, factor = apply_transform(W, "block-max", "none", norm_block=64)
        self.assertLessEqual(float(np.max(np.abs(Wt))), 1.0 + 1e-9)  # normalized
        self.assertGreater(factor, 1.0)  # mean(scale^2) for ~N(0,25) data

    def test_rotation_is_isometric(self) -> None:
        rng = np.random.default_rng(10)
        W = rng.standard_normal((4, 256))
        Wt, _ = apply_transform(W, "none", "hadamard")
        # Hadamard is orthonormal -> energy preserved (to float32 FWHT precision).
        np.testing.assert_allclose(np.sum(W ** 2), np.sum(Wt ** 2), rtol=1e-4)

    def test_rank_orders_and_skips_infeasible(self) -> None:
        rng = np.random.default_rng(11)
        W = np.vstack([rng.standard_normal(256) * 0.01, rng.standard_normal(256) * 2.0])
        ranked = rank_transforms(W, norm_block=128)
        self.assertEqual(len(ranked), len(DEFAULT_TRANSFORM_GRID))
        mses = [m for _, m in ranked]
        self.assertEqual(mses, sorted(mses))  # ascending
        # none/none should not be the single best on a multi-scale tensor
        self.assertNotEqual(ranked[0][0], ("none", "none"))

    def test_rank_skips_hadamard_on_bad_width(self) -> None:
        W = np.zeros((4, 6))  # cols=6 -> no pow2 block, hadamard configs dropped
        ranked = rank_transforms(W)
        configs = [cfg for cfg, _ in ranked]
        self.assertIn(("none", "none"), configs)
        self.assertNotIn(("none", "hadamard"), configs)


if __name__ == "__main__":
    unittest.main()
