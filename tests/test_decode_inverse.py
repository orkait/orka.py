"""Behavior gate for the decode-side inverse (un-rotate + de-normalize).

The pack pipeline is locked by the structural oracle, but decode is NOT - the oracle
hashes pack-side manifest metadata only. This file gates the decode inverse before the
decode-inverse-symmetry refactor.

Key invariant: the numpy decode path (`_decode_tensor`) and the torch decode path
(`_decode_tensor_torch`) reconstruct the SAME tensor from the same artifact, for every
(normalization x rotation) combination. The refactor changes how each inverse is
computed (routing through the strategy registries); numpy<->torch parity must survive.
"""

from __future__ import annotations

import itertools
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from orka.pipeline.decode import _decode_tensor, _decode_tensor_torch
from orka.pipeline.pack import pack_checkpoint

# Modes packable without AWQ activations (deterministic, no feature flag needed).
NORMS = ["none", "block-max", "channel-block-max", "slrq-block"]
ROTATIONS = ["none", "orthogonal", "hadamard"]

TENSOR_NAME = "model.layers.0.self_attn.q_proj.weight"


def _source_weight():
    # cols=16 -> power-of-two friendly for the block-FWHT hadamard path.
    return np.random.RandomState(0).standard_normal((4, 16)).astype(np.float32)


class DecodeInverseParityTest(unittest.TestCase):
    def _pack_and_decode_both(self, normalization, rotation):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "model.json"
            weight = _source_weight()
            source.write_text(json.dumps({"tensors": {TENSOR_NAME: weight.tolist()}}))

            artifact = root / "artifact.orka"
            pack_checkpoint(
                source,
                artifact,
                group_size=4,
                codebook_size=4,
                iterations=2,
                codebook_mode="per-tensor",
                sample_vectors=None,
                backend="numpy",
                normalization=normalization,
                block_scale_size=8,
                rotation=rotation,
                rotation_seed=1234,
                em_aq_passes=0,
            )

            manifest = json.loads((artifact / "manifest.json").read_text())
            tm = manifest["tensors"][0]

            np_flat = np.asarray(_decode_tensor(artifact, tm), dtype=np.float32).reshape(-1)
            torch_out = _decode_tensor_torch(artifact, tm, "cpu")
            torch_flat = torch_out.detach().cpu().numpy().astype(np.float32).reshape(-1)
            return np_flat, torch_flat, weight

    def test_numpy_and_torch_decode_agree_for_every_mode(self):
        for normalization, rotation in itertools.product(NORMS, ROTATIONS):
            with self.subTest(normalization=normalization, rotation=rotation):
                np_flat, torch_flat, weight = self._pack_and_decode_both(normalization, rotation)
                self.assertEqual(np_flat.shape[0], weight.size)
                self.assertEqual(torch_flat.shape[0], weight.size)
                self.assertTrue(np.isfinite(np_flat).all(), "numpy decode produced non-finite values")
                self.assertTrue(np.isfinite(torch_flat).all(), "torch decode produced non-finite values")
                # The load-bearing invariant: both inverse paths agree.
                np.testing.assert_allclose(torch_flat, np_flat, rtol=2e-3, atol=2e-3)


class AWQDecodeInverseParityTest(unittest.TestCase):
    """AWQ col-scale inverse parity. AWQ packing needs ORKA_ENABLE_AWQ + per-column
    activations; awq-block-max additionally requires the torch backend. Locks numpy<->torch
    decode parity for the col-scale (and block+col) inverse before centralizing it."""

    def _pack_and_decode_both(self, normalization, backend):
        prev = os.environ.get("ORKA_ENABLE_AWQ")
        os.environ["ORKA_ENABLE_AWQ"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "model.json"
                weight = _source_weight()  # [4, 16]
                source.write_text(json.dumps({"tensors": {TENSOR_NAME: weight.tolist()}}))
                # per-column activation stat (positive), cols must match weight cols (16).
                # torch backend (awq-block-max) consumes torch tensors; numpy backend numpy arrays.
                act = np.abs(np.linspace(0.1, 4.0, 16)).reshape(1, 16).repeat(4, 0).astype(np.float32)
                if backend == "torch":
                    import torch

                    acts = {TENSOR_NAME: torch.from_numpy(act)}
                else:
                    acts = {TENSOR_NAME: act}

                artifact = root / "artifact.orka"
                pack_checkpoint(
                    source,
                    artifact,
                    group_size=4,
                    codebook_size=4,
                    iterations=2,
                    codebook_mode="per-tensor",
                    sample_vectors=None,
                    backend=backend,
                    normalization=normalization,
                    block_scale_size=8,
                    rotation="none",
                    awq_activations=acts,
                    em_aq_passes=0,
                )
                manifest = json.loads((artifact / "manifest.json").read_text())
                tm = manifest["tensors"][0]
                self.assertEqual(tm["normalization"], normalization)

                np_flat = np.asarray(_decode_tensor(artifact, tm), dtype=np.float32).reshape(-1)
                torch_out = _decode_tensor_torch(artifact, tm, "cpu")
                torch_flat = torch_out.detach().cpu().numpy().astype(np.float32).reshape(-1)
                return np_flat, torch_flat, weight
        finally:
            if prev is None:
                os.environ.pop("ORKA_ENABLE_AWQ", None)
            else:
                os.environ["ORKA_ENABLE_AWQ"] = prev

    def _assert_parity(self, normalization, backend):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        np_flat, torch_flat, weight = self._pack_and_decode_both(normalization, backend)
        self.assertEqual(np_flat.shape[0], weight.size)
        self.assertEqual(torch_flat.shape[0], weight.size)
        self.assertTrue(np.isfinite(np_flat).all())
        self.assertTrue(np.isfinite(torch_flat).all())
        np.testing.assert_allclose(torch_flat, np_flat, rtol=2e-3, atol=2e-3)

    def test_awq_parity(self):
        self._assert_parity("awq", backend="numpy")

    def test_awq_block_max_parity(self):
        # awq-block-max requires the torch backend at pack time.
        self._assert_parity("awq-block-max", backend="torch")


if __name__ == "__main__":
    unittest.main()
