import unittest

import numpy as np
import torch

from orka.quant.lattice import e8_encode, E8_DIM
from orka.quant.lattice_pack import _pack_keys

HAVE_CUDA = torch.cuda.is_available()


class LatticePackTest(unittest.TestCase):
    def _dev(self):
        return "cuda" if HAVE_CUDA else "cpu"

    @unittest.skipUnless(HAVE_CUDA, "rANS key packing needs CUDA")
    def test_pack_unpack_keys_roundtrip(self):
        # _pack_keys rANS-codes a stack of [N,8] int keys; verify exact recovery
        # through the shift + ans round-trip (mirrors reconstruct_state_dict).
        import struct
        from orka.quant.ans import ans_decompress
        torch.manual_seed(0)
        W = torch.randn(64, 256, device=self._dev()) * 0.02
        scales = [0.05, 0.02]
        _, keys, _ = e8_encode(W, scales, seed=1)
        blob = _pack_keys(keys)
        shift = struct.unpack("<i", blob[:4])[0]
        sym = ans_decompress(blob[4:], self._dev())
        nvec = keys[0].shape[0]
        restored = (sym + shift).reshape(len(scales), nvec, E8_DIM)
        for s in range(len(scales)):
            self.assertTrue(
                torch.equal(restored[s], keys[s].to(restored.dtype)),
                f"stage {s} keys corrupted through pack/unpack",
            )


if __name__ == "__main__":
    unittest.main()
