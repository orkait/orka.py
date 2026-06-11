from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from orka.pipeline.pack import pack_checkpoint
from orka.verify import verify_artifact


@unittest.skipUnless(HAS_TORCH, "torch required")
class CompensationMathTest(unittest.TestCase):
    def test_identity_hessian_matches_greedy(self) -> None:
        """With isotropic inputs (H = I), the block-OBS update direction is
        zero-coupled across groups, so compensated assignment must equal
        plain greedy assignment."""
        from orka.codebook import learn_codebook_auto, quantize_vectors_auto
        from orka.compensation import compensated_assign

        rng = np.random.default_rng(0)
        W = torch.from_numpy(rng.standard_normal((16, 32)).astype(np.float32))
        # huge isotropic sample -> H ~ I
        X = torch.from_numpy(rng.standard_normal((8192, 32)).astype(np.float32))
        vecs = W.reshape(-1, 4)
        cb, _, _ = learn_codebook_auto(vecs, 16, 6, "torch", "cpu", seed=1)
        idx_greedy, _ = quantize_vectors_auto(vecs, cb, "torch", "cpu")
        idxs, decoded = compensated_assign(W, [cb], 4, X, damp=0.01)
        agreement = float((idxs[0] == idx_greedy).float().mean())
        self.assertGreater(agreement, 0.95)

    def test_compensation_reduces_output_error_on_correlated_inputs(self) -> None:
        from orka.codebook import learn_codebook_auto, quantize_vectors_auto
        from orka.compensation import compensated_assign

        rng = np.random.default_rng(1)
        # strongly correlated input columns -> compensation has signal
        base = rng.standard_normal((4096, 4)).astype(np.float32)
        mix = rng.standard_normal((4, 32)).astype(np.float32)
        X = torch.from_numpy(base @ mix + 0.1 * rng.standard_normal((4096, 32)).astype(np.float32))
        W = torch.from_numpy(rng.standard_normal((24, 32)).astype(np.float32))
        vecs = W.reshape(-1, 8)
        cb, _, _ = learn_codebook_auto(vecs, 8, 6, "torch", "cpu", seed=2)
        idx_greedy, _ = quantize_vectors_auto(vecs, cb, "torch", "cpu")
        greedy = cb[idx_greedy].reshape(W.shape)
        _, comp = compensated_assign(W, [cb], 8, X)
        e_greedy = (X @ (W - greedy).T).pow(2).mean().item()
        e_comp = (X @ (W - comp).T).pow(2).mean().item()
        self.assertLess(e_comp, e_greedy)


@unittest.skipUnless(HAS_TORCH, "torch required")
class CompensatedPackTest(unittest.TestCase):
    def test_pack_with_compensation_round_trips(self) -> None:
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "model.json"
            name = "model.layers.0.mlp.up_proj.weight"
            W = rng.standard_normal((16, 32)).round(3)
            src.write_text(json.dumps({"tensors": {name: W.tolist()}}))
            acts = {name: torch.from_numpy(
                rng.standard_normal((512, 32)).astype(np.float32))}
            artifact = root / "comp.orka"
            manifest = pack_checkpoint(
                src, artifact, group_size=8, codebook_size=16,
                codebook_sizes=[16, 16], iterations=4,
                codebook_mode="per-tensor", backend="torch", device="cpu",
                em_aq_passes=2, awq_activations=acts, error_compensation=True,
            )
            self.assertTrue(manifest["error_compensation"])
            # manifest metrics must match disk decode exactly
            verified = verify_artifact(artifact)
            self.assertEqual(verified["verified_tensors"], 1)
            self.assertLess(verified["max_mse_delta"], 1e-6)

    def test_compensation_skipped_without_activations(self) -> None:
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "model.json"
            name = "model.layers.0.mlp.up_proj.weight"
            src.write_text(json.dumps(
                {"tensors": {name: rng.standard_normal((8, 16)).round(3).tolist()}}))
            artifact = root / "nocomp.orka"
            manifest = pack_checkpoint(
                src, artifact, group_size=8, codebook_size=8, iterations=2,
                codebook_mode="per-tensor", backend="torch", device="cpu",
                em_aq_passes=0, error_compensation=True,
            )
            self.assertTrue(manifest["error_compensation"])
            verified = verify_artifact(artifact)
            self.assertLess(verified["max_mse_delta"], 1e-6)


if __name__ == "__main__":
    unittest.main()
