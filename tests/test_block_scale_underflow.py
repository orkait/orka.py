"""Block scales must survive their own fp16 sidecar.

Block-max normalization guards ``scales == 0`` but the scale is stored fp16, so any
block whose max is below fp16's smallest subnormal (2**-24) flushed to 0.0 and the
normalize divide produced inf/nan. k-means then trained on poisoned vectors.

Real trigger: dead vocab rows in an embedding. Measured on MiniCPM5-1B
model.embed_tokens.weight - 49,190 of 6,266,880 blocks poisoned, 1.57M non-finite
values, tensor SQNR 3.63 dB where a clean pack of the same spec gives ~15 dB.

slrq-block already floored its scales for exactly this reason; block-max,
channel-block-max and awq-block-max did not.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from orka.transforms.normalize import (
    _normalize_tensor_block_max_numpy,
    _normalize_tensor_block_max_torch,
    _normalize_tensor_channel_block_max_numpy,
    _normalize_tensor_channel_block_max_torch,
)

BLOCK = 32
TINY = 1e-9  # nonzero, but far below 2**-24 ~ 5.96e-8


def _tensor_with_dead_rows() -> torch.Tensor:
    """[8, 64]: rows 0-5 normal, rows 6-7 near-zero like unused vocab entries."""
    g = torch.Generator().manual_seed(0)
    w = torch.randn(8, 64, generator=g) * 0.05
    w[6:] = TINY
    return w


class BlockScaleUnderflowTest(unittest.TestCase):
    def test_torch_block_max_stays_finite(self) -> None:
        w = _tensor_with_dead_rows()
        norm, scales, _ = _normalize_tensor_block_max_torch(w, BLOCK, "cpu")
        self.assertTrue(torch.isfinite(norm).all(), "normalized weights contain inf/nan")
        self.assertTrue((scales > 0).all(), "a stored block scale flushed to zero")

    def test_numpy_block_max_stays_finite(self) -> None:
        w = _tensor_with_dead_rows().numpy()
        norm, scales, _ = _normalize_tensor_block_max_numpy(w, BLOCK)
        self.assertTrue(np.isfinite(norm).all(), "normalized weights contain inf/nan")
        self.assertTrue((scales > 0).all(), "a stored block scale flushed to zero")

    def test_torch_channel_block_max_stays_finite(self) -> None:
        w = _tensor_with_dead_rows()
        norm, scales, _ = _normalize_tensor_channel_block_max_torch(w, BLOCK, "cpu")
        self.assertTrue(torch.isfinite(norm).all())
        self.assertTrue((scales > 0).all())

    def test_numpy_channel_block_max_stays_finite(self) -> None:
        w = _tensor_with_dead_rows().numpy()
        norm, scales, _ = _normalize_tensor_channel_block_max_numpy(w, BLOCK)
        self.assertTrue(np.isfinite(norm).all())
        self.assertTrue((scales > 0).all())

    def test_scales_round_trip_through_fp16(self) -> None:
        """The stored scale must reproduce the weights: w == normalized * scale."""
        w = _tensor_with_dead_rows()
        norm, scales, _ = _normalize_tensor_block_max_torch(w, BLOCK, "cpu")
        rebuilt = (norm.reshape(-1, BLOCK) * scales[:, None]).reshape(w.shape)
        torch.testing.assert_close(rebuilt, w, rtol=1e-3, atol=1e-9)

    def test_backends_agree(self) -> None:
        w = _tensor_with_dead_rows()
        nt, st, _ = _normalize_tensor_block_max_torch(w, BLOCK, "cpu")
        nn_, sn, _ = _normalize_tensor_block_max_numpy(w.numpy(), BLOCK)
        np.testing.assert_allclose(st.numpy(), sn, rtol=0, atol=0)
        np.testing.assert_allclose(nt.numpy(), nn_, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
