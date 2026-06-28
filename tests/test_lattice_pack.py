import unittest

import numpy as np
import torch

from orka.quant.lattice import e8_encode, E8_DIM
from orka.quant.lattice_pack import _pack_keys

HAVE_CUDA = torch.cuda.is_available()


class LatticePackTest(unittest.TestCase):
    def _dev(self):
        return "cuda" if HAVE_CUDA else "cpu"

    def test_pack_unpack_keys_roundtrip(self):
        # _pack_keys must zlib a stack of [N,8] int keys that unpacks identically.
        torch.manual_seed(0)
        W = torch.randn(64, 256, device=self._dev()) * 0.02
        scales = [0.05, 0.02]
        _, keys, _ = e8_encode(W, scales, seed=1)
        blob = _pack_keys(keys)
        import zlib
        raw = np.frombuffer(zlib.decompress(blob), dtype=np.int16)
        nvec = keys[0].shape[0]
        restored = raw.reshape(len(scales), nvec, E8_DIM)
        for s in range(len(scales)):
            self.assertTrue(
                np.array_equal(restored[s], keys[s].to(torch.int16).cpu().numpy()),
                f"stage {s} keys corrupted through pack/unpack",
            )


if __name__ == "__main__":
    unittest.main()
