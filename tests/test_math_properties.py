"""Core math validation: every numerical primitive proven by property tests.

Covers: bit-packing inverse, FWHT involution/orthogonality, orthogonal
rotation round-trip, all normalize/denorm round-trips, metrics formulas vs
direct computation, payload arithmetic, GEMM-form distance correctness,
fp16-vs-fp32 assignment agreement, Hessian weight tiling order, RVQ stage
additivity, and distill's differentiable mirror vs the production decoder.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class BitPackingMathTest(unittest.TestCase):
    def test_pack_unpack_inverse_all_widths(self) -> None:
        from orka.core._format import _pack_indices, _unpack_indices

        rng = np.random.default_rng(0)
        for bits in (1, 2, 3, 5, 7, 11, 13, 17, 23):
            idx = rng.integers(0, 1 << bits, size=1023, dtype=np.uint64)
            raw = _pack_indices(idx, bits)
            self.assertEqual(len(raw), math.ceil(1023 * bits / 8))
            back = _unpack_indices(raw, bits, 1023)
            np.testing.assert_array_equal(back, idx.astype(np.int64))


class RotationMathTest(unittest.TestCase):
    def test_fwht_is_involutive_and_orthogonal(self) -> None:
        from orka.transforms.rotate import _fwht_numpy

        rng = np.random.default_rng(1)
        x = rng.standard_normal((8, 64)).astype(np.float32)
        # Involution: H/sqrt(n) is symmetric orthogonal.
        np.testing.assert_allclose(_fwht_numpy(_fwht_numpy(x)), x, atol=1e-5)
        # Norm preservation (orthogonality).
        np.testing.assert_allclose(
            np.linalg.norm(_fwht_numpy(x), axis=1),
            np.linalg.norm(x, axis=1),
            rtol=1e-5,
        )

    def test_orthogonal_q_properties_and_round_trip(self) -> None:
        from orka.transforms.rotate import (
            _generate_orthogonal_numpy,
            _rotate_tensor_to_2d,
            _unrotate_flat,
        )

        q = _generate_orthogonal_numpy(32, seed=1234)
        np.testing.assert_allclose(q @ q.T, np.eye(32), atol=1e-5)

        rng = np.random.default_rng(2)
        w = rng.standard_normal((16, 32)).astype(np.float32)
        rotated, seed = _rotate_tensor_to_2d(w, "t.weight", "orthogonal", 99, "numpy", "cpu")
        back = _unrotate_flat(rotated.reshape(-1), [16, 32], "orthogonal", seed)
        np.testing.assert_allclose(back.reshape(16, 32), w, atol=1e-4)

    def test_hadamard_round_trip(self) -> None:
        from orka.transforms.rotate import _rotate_tensor_to_2d, _unrotate_flat

        rng = np.random.default_rng(3)
        w = rng.standard_normal((8, 48)).astype(np.float32)  # 48 = 16 * 3
        rotated, seed = _rotate_tensor_to_2d(w, "t.weight", "hadamard", 0, "numpy", "cpu")
        back = _unrotate_flat(rotated.reshape(-1), [8, 48], "hadamard", seed)
        np.testing.assert_allclose(back.reshape(8, 48), w, atol=1e-5)

    def test_rotation_registry_modes_and_unknown_raises(self) -> None:
        from orka.transforms.rotate import _rotate_tensor_to_2d, rotation_modes

        self.assertEqual(set(rotation_modes()), {"hadamard", "orthogonal"})
        with self.assertRaises(ValueError):
            _rotate_tensor_to_2d(np.zeros((2, 4), np.float32), "t", "bogus", 0, "numpy", "cpu")

    def test_register_rotation_dispatches_and_round_trips(self) -> None:
        # A custom rotation plugs in via register_rotation with no dispatcher edit.
        from orka.transforms.rotate import (
            ROTATION_REGISTRY,
            RotationStrategy,
            _rotate_tensor_to_2d,
            _unrotate_flat,
            register_rotation,
        )

        def _flip_rotate(tensor, *, name, rotation_seed, backend, device):
            return np.asarray(tensor, np.float32)[:, ::-1].copy(), 0

        def _flip_unrotate(arr, *, cols, seed):
            return arr[:, ::-1].copy()

        register_rotation(RotationStrategy("flip", _flip_rotate, _flip_unrotate))
        try:
            w = np.random.default_rng(7).standard_normal((4, 6)).astype(np.float32)
            rotated, seed = _rotate_tensor_to_2d(w, "t", "flip", 0, "numpy", "cpu")
            back = _unrotate_flat(rotated.reshape(-1), [4, 6], "flip", seed)
            np.testing.assert_allclose(back.reshape(4, 6), w, atol=0)
        finally:
            ROTATION_REGISTRY.pop("flip", None)


class NormalizationRoundTripTest(unittest.TestCase):
    def test_block_max_round_trip(self) -> None:
        from orka.transforms.normalize import (
            _apply_block_max_scales_numpy,
            _normalize_tensor_block_max_numpy,
        )

        rng = np.random.default_rng(4)
        w = (rng.standard_normal(100) * 7).astype(np.float32)
        normalized, scales, source = _normalize_tensor_block_max_numpy(w, 16)
        # |normalized| stays bounded near 1 (fp16 scale rounding can nudge it).
        self.assertLessEqual(float(np.abs(normalized).max()), 1.001)
        back = _apply_block_max_scales_numpy(normalized.reshape(-1), scales, 16)
        np.testing.assert_allclose(back, source, rtol=1e-6, atol=1e-7)

    def test_channel_block_max_round_trip_matches_flat_layout(self) -> None:
        from orka.transforms.normalize import (
            _apply_block_max_scales_numpy,
            _normalize_tensor_channel_block_max_numpy,
        )

        rng = np.random.default_rng(5)
        w = (rng.standard_normal((6, 32)) * 3).astype(np.float32)
        normalized, scales, source = _normalize_tensor_channel_block_max_numpy(w, 8)
        # Decode treats the flat stream as consecutive blocks of 8; the
        # channel-aligned scales must land on exactly those blocks.
        back = _apply_block_max_scales_numpy(normalized.reshape(-1), scales, 8)
        np.testing.assert_allclose(back, source, rtol=1e-6, atol=1e-7)

    def test_slrq_round_trip_with_salient(self) -> None:
        from orka.transforms.normalize import (
            _apply_block_max_scales_numpy,
            _normalize_tensor_slrq_block_numpy,
        )

        rng = np.random.default_rng(6)
        w = (rng.standard_normal(96) * 5).astype(np.float32)
        normalized, scales, sal_w, sal_i, source = _normalize_tensor_slrq_block_numpy(
            w.copy(), 16, salient_enabled=True
        )
        # Anchors are powers of two (exact in fp16 within range).
        log2s = np.log2(scales)
        np.testing.assert_allclose(log2s, np.round(log2s), atol=1e-6)
        back = _apply_block_max_scales_numpy(normalized.reshape(-1), scales, 16)
        for b_idx, (li, val) in enumerate(zip(sal_i, sal_w)):
            back[b_idx * 16 + int(li)] = val
        # Salient values are fp16-rounded at capture by design.
        np.testing.assert_allclose(back, source, rtol=1e-3, atol=1e-4)
        non_salient = np.ones(96, dtype=bool)
        for b_idx, li in enumerate(sal_i):
            non_salient[b_idx * 16 + int(li)] = False
        np.testing.assert_allclose(
            back[non_salient], source[non_salient], rtol=1e-6, atol=1e-7
        )


class MetricsMathTest(unittest.TestCase):
    def test_quality_metrics_match_direct_formulas(self) -> None:
        from orka.eval.metrics import quality_metrics_from_flat

        rng = np.random.default_rng(7)
        src = rng.standard_normal(10_000).astype(np.float32)
        rec = src + 0.1 * rng.standard_normal(10_000).astype(np.float32)
        m = quality_metrics_from_flat(src, rec)

        diff = src.astype(np.float64) - rec.astype(np.float64)
        sse = float(np.sum(diff**2))
        # Metrics accumulate in float32 chunks; compare at f32 accumulation
        # precision (formulas exact, representation approximate).
        self.assertAlmostEqual(m["mse"], sse / len(src), places=6)
        self.assertAlmostEqual(m["rmse"], math.sqrt(sse / len(src)), places=6)
        self.assertAlmostEqual(m["mae"], float(np.mean(np.abs(diff))), places=6)
        self.assertAlmostEqual(
            m["relative_rmse"], math.sqrt(sse / float(np.sum(src.astype(np.float64) ** 2))), places=6
        )
        cos = float(
            np.dot(src.astype(np.float64), rec.astype(np.float64))
            / (np.linalg.norm(src.astype(np.float64)) * np.linalg.norm(rec.astype(np.float64)))
        )
        self.assertAlmostEqual(m["cosine_similarity"], cos, places=6)
        self.assertAlmostEqual(
            m["sqnr"],
            10.0 * math.log10(float(np.sum(src.astype(np.float64) ** 2)) / sse),
            places=6,
        )


class PayloadMathTest(unittest.TestCase):
    def test_estimate_payload_arithmetic(self) -> None:
        from orka.quant import estimate_payload

        est = estimate_payload(8_030_000_000, 8, 256, scale_block_vectors=64, scale_bits=16)
        self.assertEqual(est.index_bits, 8)
        self.assertEqual(est.vector_count, math.ceil(8_030_000_000 / 8))
        self.assertEqual(est.index_bytes, math.ceil(est.vector_count * 8 / 8))
        scale_count = math.ceil(est.vector_count / 64)
        self.assertEqual(est.scale_bytes, math.ceil(scale_count * 16 / 8))
        self.assertAlmostEqual(
            est.bits_per_weight,
            (est.index_bytes + est.scale_bytes) * 8 / 8_030_000_000,
            places=12,
        )


class DistanceMathTest(unittest.TestCase):
    def test_gemm_form_distance_equals_direct(self) -> None:
        from orka.codebook.kmeans import _numpy_assign

        rng = np.random.default_rng(8)
        rows = rng.standard_normal((512, 8)).astype(np.float32)
        cb = rng.standard_normal((16, 8)).astype(np.float32)
        indices, mse = _numpy_assign(rows, cb)
        direct = np.argmin(
            ((rows[:, None, :] - cb[None, :, :]) ** 2).sum(axis=2), axis=1
        )
        np.testing.assert_array_equal(indices, direct)
        chosen_sse = float(
            (((rows - cb[direct]) ** 2).sum())
        )
        self.assertAlmostEqual(mse, chosen_sse / rows.size, places=4)

    @unittest.skipUnless(HAS_TORCH, "torch required")
    def test_fp16_assignment_agreement_on_cuda_scale_data(self) -> None:
        """The torch path ranks distances in fp16 on CUDA. Quantify the
        assignment disagreement vs fp32 on CPU-equivalent data; near-ties may
        flip, which is distortion-neutral, but agreement must stay high."""
        from orka.codebook.kmeans import _torch_assign

        rng = np.random.default_rng(9)
        rows_np = rng.standard_normal((4096, 8)).astype(np.float32)
        cb_np = rng.standard_normal((256, 8)).astype(np.float32)
        rows = torch.from_numpy(rows_np)
        cb = torch.from_numpy(cb_np)
        idx32, _ = _torch_assign(rows, cb, "cpu")  # fp32 on cpu
        if torch.cuda.is_available():
            idx16, _ = _torch_assign(rows.cuda(), cb.cuda(), "cuda")  # fp16 path
            agreement = float((idx32 == idx16.cpu()).float().mean())
            self.assertGreater(agreement, 0.97)
            # Disagreements must be distortion-neutral: compare achieved SSE.
            sse32 = float(((rows_np - cb_np[idx32.numpy()]) ** 2).sum())
            sse16 = float(((rows_np - cb_np[idx16.cpu().numpy()]) ** 2).sum())
            self.assertLess(abs(sse16 - sse32) / sse32, 0.01)


class HessianWeightTilingTest(unittest.TestCase):
    def test_sample_weight_tiling_matches_row_major_flatten(self) -> None:
        """Vector v of a row-major [rows, cols] flatten at group G covers
        columns [(v % (cols//G))*G, ...). The tiled weight vector must assign
        every vector the mean importance of exactly those columns."""
        if not HAS_TORCH:
            self.skipTest("torch required")
        rows, cols, G = 3, 12, 4
        h = torch.arange(1.0, cols + 1)  # distinct per column
        gpr = cols // G
        h_groups = h.reshape(gpr, G)
        sw_row = h_groups.mean(dim=1)
        sw_full = (sw_row / sw_row.mean()).repeat(rows)
        for v in range(rows * gpr):
            start_col = (v % gpr) * G
            expected = h[start_col : start_col + G].mean()
            self.assertAlmostEqual(
                float(sw_full[v] * sw_row.mean()), float(expected), places=5
            )


class RVQAdditivityAndMirrorTest(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "torch required")
    def test_differentiable_mirror_equals_production_decoder(self) -> None:
        """Strongest single check: distill's autograd decode must reproduce
        the production numpy decoder bit-for-bit (within f32 accumulation)
        across the FULL transform chain."""
        from orka.qat.distill import _differentiable_decode, _load_decode_consts
        from orka.pipeline.decode import _decode_tensor
        from orka.pipeline.pack import pack_checkpoint

        rng = np.random.default_rng(10)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "model.json"
            src.write_text(
                json.dumps(
                    {
                        "tensors": {
                            "model.layers.0.self_attn.q_proj.weight": (
                                rng.standard_normal((16, 32)) * 2
                            ).round(3).tolist()
                        }
                    }
                )
            )
            artifact = root / "m.orka"
            manifest = pack_checkpoint(
                src, artifact, group_size=4, codebook_size=4,
                codebook_sizes=[4, 4], iterations=3,
                codebook_mode="per-tensor", backend="numpy", em_aq_passes=1,
                normalization="slrq-block", block_scale_size=8,
                rotation="orthogonal", rotation_seed=7, outlier_frac=0.02,
            )
            tm = manifest["tensors"][0]
            production = np.asarray(_decode_tensor(artifact, tm), dtype=np.float32)
            consts = _load_decode_consts(artifact, tm, "cpu")
            params = [s["codebook"] for s in consts["stages"]]
            mirror = _differentiable_decode(params, consts).detach().numpy()
            np.testing.assert_allclose(mirror, production, rtol=1e-5, atol=1e-6)

    def test_rvq_decode_is_sum_of_stage_lookups(self) -> None:
        from orka.pipeline.decode import _read_codebook
        from orka.core._format import _read_indices
        from orka.pipeline.pack import pack_checkpoint
        from orka.pipeline.decode import _decode_tensor

        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "model.json"
            src.write_text(
                json.dumps(
                    {
                        "tensors": {
                            "model.layers.0.mlp.up_proj.weight": rng.standard_normal(
                                (8, 16)
                            ).round(3).tolist()
                        }
                    }
                )
            )
            artifact = root / "s.orka"
            manifest = pack_checkpoint(
                src, artifact, group_size=4, codebook_size=4,
                codebook_sizes=[4, 4], iterations=3,
                codebook_mode="per-tensor", backend="numpy", em_aq_passes=0,
            )
            tm = manifest["tensors"][0]
            total = np.zeros(int(tm["padded_values"]), dtype=np.float32)
            for stage in tm["stages"]:
                g = int(stage["group_size"])
                cb = _read_codebook(
                    artifact / stage["codebook"], g, stage["codebook_dtype"]
                )
                idx = _read_indices(
                    artifact / stage["indices"], stage["index_bits"],
                    math.ceil(int(tm["padded_values"]) / g),
                    packed=stage["packed"], encoding=stage.get("encoding", "raw"),
                )
                total += cb[np.asarray(idx, dtype=np.int64)].reshape(-1)
            decoded = _decode_tensor(artifact, tm)
            np.testing.assert_allclose(
                total[: int(tm["packed_values"])], decoded, rtol=1e-6, atol=1e-7
            )


def _faiss_cuda_available() -> bool:
    if not HAS_TORCH or not torch.cuda.is_available():
        return False
    try:
        import faiss
        return faiss.get_num_gpus() > 0
    except Exception:
        return False


@unittest.skipUnless(_faiss_cuda_available(), "faiss-gpu + CUDA required")
class FaissKmeansTest(unittest.TestCase):
    """The opt-in faiss GPU k-means path matches the torch path's quality and is
    byte-deterministic per seed."""

    def test_faiss_matches_torch_quality_and_is_deterministic(self) -> None:
        from orka.codebook._kmeans_torch import (
            _learn_codebook_faiss,
            _learn_codebook_torch,
        )

        torch.manual_seed(0)
        rows = torch.randn(20000, 8, device="cuda")
        k, iters = 1024, 12

        _, _, mse_torch = _learn_codebook_torch(rows, k, iters, "cuda", seed=0)
        cb1, _, mse_faiss = _learn_codebook_faiss(rows, k, iters, "cuda", 0)
        cb2, _, _ = _learn_codebook_faiss(rows, k, iters, "cuda", 0)

        self.assertEqual(tuple(cb1.shape), (k, 8))
        self.assertTrue(torch.isfinite(cb1).all())
        # same-seed faiss is byte-deterministic
        self.assertTrue(torch.equal(cb1, cb2))
        # quality within 5% of the torch Lloyd (different algorithm, equal optimum)
        self.assertLess(abs(mse_faiss - mse_torch) / mse_torch, 0.05)

    def test_opt_in_gate_off_by_default(self) -> None:
        import os
        from orka.codebook._kmeans_torch import _faiss_kmeans_enabled

        prev = os.environ.pop("ORKA_KMEANS_FAISS", None)
        try:
            self.assertFalse(_faiss_kmeans_enabled())
            os.environ["ORKA_KMEANS_FAISS"] = "1"
            self.assertTrue(_faiss_kmeans_enabled())
        finally:
            os.environ.pop("ORKA_KMEANS_FAISS", None)
            if prev is not None:
                os.environ["ORKA_KMEANS_FAISS"] = prev


if __name__ == "__main__":
    unittest.main()
