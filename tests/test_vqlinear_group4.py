"""Parity gate for compressed-resident VQLinear at group_size != 8.

The CUDA float4 fast path is group_size==8 only; the Triton kernels
(_vq_decode_n1 / _vq_gemm_kernel) are general. The dispatcher used to hard-assert
group_size==8, forcing every other config onto the dense fallback. This locks that a
group_size=4 VQLinear forward (the good-quality regime) reconstructs the SAME output as
the dense decode path, for both the N=1 decode kernel and the N>1 GEMM kernel.

Requires CUDA + Triton; skips cleanly otherwise.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@unittest.skipUnless(_has_cuda(), "CUDA required for VQLinear kernels")
class VQLinearGroup4ParityTest(unittest.TestCase):
    def _pack_and_load(self, group_size, block_scale_size, codebook_size, n_stages=2):
        import torch
        from orka.pipeline.pack import pack_checkpoint
        from orka.pipeline.decode import _decode_tensor
        from orka.inference.vq_linear import build_vq_linear

        M, K = 256, 256   # 16384 vectors at group 4 - enough for codebooks up to 8192
        rng = np.random.RandomState(0)
        w = rng.standard_normal((M, K)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "model.json"
            name = "model.layers.0.mlp.down_proj.weight"
            src.write_text(json.dumps({"tensors": {name: w.tolist()}}))
            art = root / "art.orka"
            pack_checkpoint(
                src, art, group_size=group_size, codebook_size=codebook_size, iterations=4,
                codebook_mode="per-tensor", sample_vectors=None, backend="numpy",
                normalization="block-max", block_scale_size=block_scale_size,
                codebook_sizes=[codebook_size] * n_stages, em_aq_passes=0,
            )
            manifest = json.loads((art / "manifest.json").read_text())
            tm = manifest["tensors"][0]
            self.assertEqual(tm["group_size"], group_size)

            dense_flat = np.asarray(_decode_tensor(art, tm), dtype=np.float32).reshape(M, K)
            W = torch.from_numpy(dense_flat).to("cuda", torch.float16)

            layer = build_vq_linear(art, tm, bias=None, device="cuda").to("cuda").eval()
            return layer, W, M, K

    def _check(self, group_size, block_scale_size, codebook_size, tier, n_stages=2):
        import torch
        import torch.nn.functional as F

        layer, W, M, K = self._pack_and_load(group_size, block_scale_size, codebook_size, n_stages)
        self.assertEqual(layer.n_stages, n_stages)
        # storage tier by index width: uint8 (<=8b) / planed (10,12b) / int16 (else)
        if tier == "planed":
            self.assertTrue(hasattr(layer, "indices_lo_0"))
            self.assertFalse(hasattr(layer, "indices_0"))
            self.assertGreater(layer._plane_width[0], 8)
        else:
            self.assertEqual(layer.indices_0.dtype, getattr(torch, tier))
            self.assertEqual(layer._plane_width[0], 0)
        for N in (1, 8):
            x = torch.randn(N, K, device="cuda", dtype=torch.float16)
            with torch.no_grad():
                y_kernel = layer(x).float()
                y_dense = F.linear(x, W).float()
            torch.testing.assert_close(y_kernel, y_dense, rtol=3e-2, atol=3e-2)

    def test_uint8_indices_match_dense(self):
        # 256-entry codebook -> uint8 (8-bit), half the int16 VRAM
        self._check(group_size=4, block_scale_size=16, codebook_size=256, tier="uint8")

    def test_planed10_indices_match_dense(self):
        # 1024-entry -> 10-bit -> bit-planes (lo uint8 + 2-bit hi)
        self._check(group_size=4, block_scale_size=16, codebook_size=1024, tier="planed")

    def test_planed12_indices_match_dense(self):
        # 4096-entry -> 12-bit -> bit-planes (lo uint8 + 4-bit hi); the rvq-12-12 regime
        self._check(group_size=4, block_scale_size=16, codebook_size=4096, tier="planed")

    def test_int16_indices_match_dense(self):
        # 8192-entry -> 13-bit -> int16 (planes only engage for 10/12)
        self._check(group_size=4, block_scale_size=16, codebook_size=8192, tier="int16")

    def test_group8_still_matches_dense(self):
        self._check(group_size=8, block_scale_size=32, codebook_size=256, tier="uint8")

    def test_planed_group8_matches_dense(self):
        # group-8 + 1024 codebook -> 10-bit planes -> warp kernel G=8 half2 path
        self._check(group_size=8, block_scale_size=32, codebook_size=1024, tier="planed")


    def test_planed_1stage_matches_dense(self):
        self._check(group_size=4, block_scale_size=16, codebook_size=1024, tier="planed", n_stages=1)

    def test_planed_3stage_matches_dense(self):
        self._check(group_size=4, block_scale_size=16, codebook_size=1024, tier="planed", n_stages=3)


if __name__ == "__main__":
    unittest.main()
