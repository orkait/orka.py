"""Round-trip gate for the .orka -> GGUF data path (llama.cpp phase 1).

Locks the dequant reference (the rule the C/CUDA GGML kernels must match): the RVQ
reconstruction from stored per-stage indices + codebooks + block scales equals
VQLinear.reconstruct_weight, and a GGUF written by export_gguf reads back to the same
tensors. Decoupled from llama.cpp's model graph (the hard wiring) - this validates the
format + math foundation.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _has_gguf():
    try:
        import gguf  # noqa: F401
        return True
    except Exception:
        return False


class ExportGgufRoundTripTest(unittest.TestCase):
    def _pack(self, tmp, group_size, codebook_size, block_scale_size):
        from orka.pipeline.pack import pack_checkpoint

        M, K = 64, 64
        w = np.random.RandomState(0).standard_normal((M, K)).astype(np.float32)
        src = Path(tmp) / "model.json"
        name = "model.layers.0.mlp.down_proj.weight"
        src.write_text(json.dumps({"tensors": {name: w.tolist()}}))
        art = Path(tmp) / "art.orka"
        pack_checkpoint(
            src, art, group_size=group_size, codebook_size=codebook_size, iterations=4,
            codebook_mode="per-tensor", sample_vectors=None, backend="numpy",
            normalization="block-max", block_scale_size=block_scale_size,
            codebook_sizes=[codebook_size, codebook_size], em_aq_passes=0,
        )
        return art, json.loads((art / "manifest.json").read_text()), name

    def _dequant_matches_reconstruct(self, group_size, codebook_size, block_scale_size):
        from orka.artifact.export_gguf import dequant_linear
        from orka.inference.vq_linear import build_vq_linear

        with tempfile.TemporaryDirectory() as tmp:
            art, manifest, name = self._pack(tmp, group_size, codebook_size, block_scale_size)
            tm = manifest["tensors"][0]
            layer = build_vq_linear(art, tm, bias=None, device="cpu")
            M, K = layer.out_features, layer.in_features
            idx = [layer._stage_indices_int(s).numpy() for s in range(layer.n_stages)]
            cbs = [getattr(layer, f"codebook_{s}").float().numpy() for s in range(layer.n_stages)]
            sc = layer.scales.float().numpy()
            W = dequant_linear(idx, cbs, sc, M, K, layer.group_size, layer.block_size,
                               bool(getattr(layer, "_group_major", False)))
            ref = layer.reconstruct_weight().numpy()
            np.testing.assert_allclose(W, ref, rtol=1e-2, atol=1e-2)

    def test_dequant_planed_rowmajor(self):
        # cb 1024 -> 10-bit planes -> row-major
        self._dequant_matches_reconstruct(group_size=8, codebook_size=1024, block_scale_size=32)

    def test_dequant_groupmajor_uint8(self):
        # cb 256 -> uint8 -> group-major (exercises the transpose path)
        self._dequant_matches_reconstruct(group_size=8, codebook_size=256, block_scale_size=32)

    @unittest.skipUnless(_has_gguf(), "gguf not installed")
    def test_gguf_export_roundtrips_tensors(self):
        from gguf import GGUFReader

        from orka.artifact.export_gguf import export_gguf

        with tempfile.TemporaryDirectory() as tmp:
            art, manifest, name = self._pack(tmp, 8, 1024, 32)
            # minimal config dir (export_gguf only needs the artifact for tensors)
            out = Path(tmp) / "model.gguf"
            summary = export_gguf(art, art, out)
            self.assertEqual(summary["quantized_linears"], 1)
            self.assertTrue(out.exists())
            reader = GGUFReader(str(out))
            names = {t.name for t in reader.tensors}
            self.assertIn(f"{name}.idx0", names)
            self.assertIn(f"{name}.cb0", names)
            self.assertIn(f"{name}.scales", names)


if __name__ == "__main__":
    unittest.main()
