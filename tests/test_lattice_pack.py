import unittest

import numpy as np
import torch

from orka.quant.lattice import e8_encode, E8_DIM
from orka.quant.lattice_pack import _is_quantizable, _pack_keys

HAVE_CUDA = torch.cuda.is_available()


class LatticeCoverageTest(unittest.TestCase):
    """_is_quantizable must cover any 2-D Linear except the output head, regardless of
    architecture. The old allow-list ('self_attn'/'mlp') silently skipped feed_forward
    and mamba on a FalconH1 hybrid -> only 9% of params quantized, fictional bpw."""

    def test_covers_non_transformer_linears(self):
        lin = torch.nn.Linear(16, 16)
        for name in (
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.mlp.down_proj",
            "model.layers.0.feed_forward.gate_proj",   # FalconH1 MLP (old check missed)
            "model.layers.0.mamba.in_proj",            # SSM linear (old check missed)
            "backbone.layers.3.mixer.out_proj",
        ):
            self.assertTrue(_is_quantizable(name, lin), name)

    def test_excludes_output_head_and_non_linear(self):
        lin = torch.nn.Linear(16, 16)
        self.assertFalse(_is_quantizable("lm_head", lin))
        self.assertFalse(_is_quantizable("model.embed_out", lin))
        # non-Linear modules are never quantized (Conv1d / Embedding stay fp16)
        self.assertFalse(_is_quantizable("model.layers.0.mamba.conv1d", torch.nn.Conv1d(8, 8, 4)))
        self.assertFalse(_is_quantizable("model.embed_tokens", torch.nn.Embedding(32, 16)))


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
