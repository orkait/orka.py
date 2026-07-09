import math
import unittest

import torch

from orka.quant.trellis import tcq_decode, tcq_encode

HAVE_CUDA = torch.cuda.is_available()


def _sqnr(W, Wh):
    return 10 * math.log10((W.pow(2).sum() / (W - Wh).pow(2).sum().clamp(min=1e-12)).item())


def _scalar(W, R):
    sig = W.std()
    lim = 4 * sig
    lv = 2 ** R
    d = 2 * lim / (lv - 1)
    return torch.clamp(torch.round((W + lim) / d), 0, lv - 1) * d - lim


class TrellisTest(unittest.TestCase):
    def _dev(self):
        return "cuda" if HAVE_CUDA else "cpu"

    def test_roundtrip_exact(self):
        torch.manual_seed(0)
        W = torch.randn(64, 256, device=self._dev()) * 0.02
        idx, levels, _ = tcq_encode(W, R=3)
        rec = tcq_decode(idx, levels)
        # decode(encode(W)) must reproduce the encoder's own reconstruction exactly
        self.assertEqual(rec.shape, W.shape)
        self.assertTrue(torch.equal(rec, levels[idx]))

    def test_beats_scalar_at_low_rate(self):
        torch.manual_seed(0)
        W = torch.randn(128, 512, device=self._dev()) * 0.02
        for R in (2, 3):
            idx, levels, _ = tcq_encode(W, R)
            tcq = _sqnr(W, tcq_decode(idx, levels))
            scal = _sqnr(W, _scalar(W, R))
            # trellis gain: TCQ must not be worse than scalar at the same rate
            self.assertGreaterEqual(tcq, scal - 1e-6, f"R={R}: TCQ {tcq:.2f} < scalar {scal:.2f}")

    def test_rate2_gain_is_large(self):
        # at R=2 the trellis gain should be clearly positive on Gaussian-ish weights
        torch.manual_seed(1)
        W = torch.randn(128, 512, device=self._dev()) * 0.02
        idx, levels, _ = tcq_encode(W, 2)
        gain = _sqnr(W, tcq_decode(idx, levels)) - _sqnr(W, _scalar(W, 2))
        self.assertGreater(gain, 1.0)


if __name__ == "__main__":
    unittest.main()
