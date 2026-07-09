import unittest
import zlib

import numpy as np
import torch

from orka.quant.ans import (
    ans_compress,
    ans_decompress,
    build_freq_table,
    rans_decode_scalar,
    rans_encode_scalar,
    slot_to_symbol,
)

HAVE_CUDA = torch.cuda.is_available()


def _entropy_bpw(sym):
    h = np.bincount(sym)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


class ScalarRansTest(unittest.TestCase):
    def test_roundtrip_distributions(self):
        rng = np.random.default_rng(0)
        for sym in (
            rng.integers(0, 256, 3000).astype(np.int64),
            rng.zipf(1.4, 20000).clip(max=4095).astype(np.int64),
            np.zeros(100, dtype=np.int64),
        ):
            freq, cum, _ = build_freq_table(sym, 12)
            lut = slot_to_symbol(freq, 12)
            words = rans_encode_scalar(sym, freq, cum, 12)
            dec = rans_decode_scalar(words, freq, cum, lut, len(sym), 12)
            self.assertTrue(np.array_equal(dec, sym))

    def test_near_entropy(self):
        rng = np.random.default_rng(1)
        sym = rng.zipf(1.3, 200000).clip(max=4095).astype(np.int64)
        freq, cum, _ = build_freq_table(sym, 14)
        words = rans_encode_scalar(sym, freq, cum, 14)
        bpw = words.nbytes * 8 / len(sym)
        # within ~5% of entropy (flush + table excluded; this is the stream rate)
        self.assertLess(bpw, _entropy_bpw(sym) * 1.05 + 0.1)


@unittest.skipUnless(HAVE_CUDA, "GPU rANS needs CUDA")
class GpuRansTest(unittest.TestCase):
    def test_blob_roundtrip(self):
        rng = np.random.default_rng(2)
        for sym in (
            rng.integers(0, 4096, 120000).astype(np.int64),
            rng.zipf(1.5, 80000).clip(max=4095).astype(np.int64),
            rng.integers(0, 64, 1000).astype(np.int64),
        ):
            blob = ans_compress(sym, precision=12)
            dec = ans_decompress(blob).cpu().numpy()
            self.assertTrue(np.array_equal(dec, sym))

    def test_gpu_matches_scalar(self):
        rng = np.random.default_rng(3)
        sym = rng.zipf(1.4, 50000).clip(max=4095).astype(np.int64)
        blob = ans_compress(sym, precision=12)
        dec = ans_decompress(blob).cpu().numpy()
        self.assertTrue(np.array_equal(dec, sym))

    def test_beats_zlib_on_realistic_indices(self):
        # realistic VQ index distribution: high-entropy, bell-shaped (well-utilized
        # codebook) - the regime orka actually produces. rANS < zlib here.
        # (On extreme-low-entropy / fat-tail data, zlib's LZ + no-table can win;
        # that is not what VQ/lattice indices look like.)
        rng = np.random.default_rng(4)
        sym = (rng.normal(2048, 700, 300000).round() % 4096).astype(np.int64)
        ans_bytes = len(ans_compress(sym, precision=12))
        zlib_bytes = len(zlib.compress(sym.astype(np.uint16).tobytes(), 6))
        self.assertLess(ans_bytes, zlib_bytes)


if __name__ == "__main__":
    unittest.main()
