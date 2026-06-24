"""orka.hf loader: the decoded state dict must match the canonical decoder.

load_orka_model builds an HF model and loads _orka_state_dict into it. The HF arch path
needs transformers + a real config, so the unit gate here locks the load-bearing core:
_orka_state_dict reconstructs exactly what orka.pipeline.decode._decode_tensor produces,
for every quantized tensor, plus carries the passthrough tensors.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from orka.hf import _orka_state_dict
from orka.pipeline.decode import _decode_tensor
from orka.pipeline.pack import pack_checkpoint


class OrkaStateDictParityTest(unittest.TestCase):
    def _pack(self, root: Path) -> Path:
        source = root / "model.json"
        source.write_text(
            json.dumps(
                {
                    "tensors": {
                        "model.layers.0.self_attn.q_proj.weight": [
                            [1.0, 2.0, 128.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [9.0, 32.0, 11.0, 12.0],
                            [13.0, 14.0, 15.0, 16.0],
                        ]
                    }
                }
            )
        )
        artifact = root / "artifact.orka"
        pack_checkpoint(
            source,
            artifact,
            group_size=2,
            codebook_size=4,
            iterations=2,
            codebook_mode="per-tensor",
            sample_vectors=None,
            backend="numpy",
            normalization="block-max",
            block_scale_size=2,
            em_aq_passes=0,
        )
        return artifact

    def test_state_dict_matches_decode_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = self._pack(root)
            manifest = json.loads((artifact / "manifest.json").read_text())

            state = _orka_state_dict(artifact)

            for tm in manifest["tensors"]:
                name = tm["name"]
                self.assertIn(name, state, f"{name} missing from state dict")
                expected = np.asarray(_decode_tensor(artifact, tm), dtype=np.float32).reshape(
                    [int(x) for x in tm["shape"]]
                )
                got = state[name].numpy().astype(np.float32)
                self.assertEqual(list(got.shape), [int(x) for x in tm["shape"]])
                np.testing.assert_allclose(got, expected, rtol=0, atol=0)

    def test_dtype_cast_applied(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = self._pack(root)
            state = _orka_state_dict(artifact, dtype=torch.float16)
            for t in state.values():
                self.assertEqual(t.dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
