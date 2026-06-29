import math
import unittest

import torch

from orka.quant.lattice import (
    e8_encode,
    e8_decode,
    nearest_e8,
    incoherence_rotation,
)

HAVE_CUDA = torch.cuda.is_available()


def _sqnr(W, Wh):
    return 10 * math.log10((W.pow(2).sum() / (W - Wh).pow(2).sum().clamp(min=1e-12)).item())


class LatticeTest(unittest.TestCase):
    def _dev(self):
        return "cuda" if HAVE_CUDA else "cpu"

    def test_rotation_orthogonal(self):
        R = incoherence_rotation(7, self._dev())
        I = R @ R.t()
        self.assertTrue(torch.allclose(I, torch.eye(8, device=R.device), atol=1e-5))

    def test_rotation_deterministic_from_seed(self):
        a = incoherence_rotation(42, self._dev())
        b = incoherence_rotation(42, self._dev())
        self.assertTrue(torch.equal(a, b))

    def test_nearest_e8_is_lattice_point(self):
        x = torch.randn(1000, 8, device=self._dev())
        q = nearest_e8(x)
        # E8 points: either all-integer with even sum (D8), or all-half-integer with even 2x sum
        two = torch.round(q * 2)
        self.assertTrue(torch.allclose(q * 2, two, atol=1e-4))  # all half-integers
        # E8 has a lower normalized second moment than Z^8 (packing gain ~0.65 dB),
        # so MEAN error beats coordinate-wise integer rounding (not every point).
        err_e8 = ((x - q) ** 2).sum(-1).mean()
        err_int = ((x - torch.round(x)) ** 2).sum(-1).mean()
        self.assertLess(err_e8.item(), err_int.item())

    def test_encode_decode_roundtrip_exact(self):
        torch.manual_seed(0)
        W = torch.randn(96, 256, device=self._dev()) * 0.02
        scales = [0.05, 0.02]
        recon, keys, bpw = e8_encode(W, scales, seed=3)
        dec = e8_decode(keys, scales, seed=3, numel=W.numel(), shape=W.shape, device=W.device)
        self.assertTrue(torch.allclose(recon, dec, atol=1e-5))
        self.assertGreater(bpw, 0.0)

    def test_two_stage_beats_one_stage(self):
        torch.manual_seed(1)
        W = torch.randn(128, 512, device=self._dev()) * 0.02
        r1, _, _ = e8_encode(W, [0.04], seed=1)
        r2, _, _ = e8_encode(W, [0.05, 0.02], seed=1)
        self.assertGreater(_sqnr(W, r2), _sqnr(W, r1))


if __name__ == "__main__":
    unittest.main()


class IncoherenceTest(unittest.TestCase):
    def _dev(self):
        return "cuda" if HAVE_CUDA else "cpu"

    def test_input_incoherence_roundtrip_identity(self):
        # forward then inverse must recover W exactly (orthonormal block-FWHT + signs)
        from orka.quant.lattice import input_incoherence, inverse_incoherence
        torch.manual_seed(0)
        W = torch.randn(48, 576, device=self._dev()) * 0.1
        Wr, signs, bs = input_incoherence(W, seed=123)
        back = inverse_incoherence(Wr, signs, bs)
        self.assertLess((W - back).abs().max().item(), 1e-4)

    def test_incoherence_seed_deterministic(self):
        from orka.quant.lattice import input_incoherence
        torch.manual_seed(0)
        W = torch.randn(16, 128, device=self._dev())
        a, sa, _ = input_incoherence(W, seed=5)
        b, sb, _ = input_incoherence(W, seed=5)
        self.assertTrue(torch.equal(a, b) and torch.equal(sa, sb))
