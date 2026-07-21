"""The CSR correction must survive the forward path caching it.

``VQLinear._correction_sparse`` frees the raw corr_rowptr/corr_col/corr_val
buffers once it has built the sparse tensor (a deliberate memory win - the
int64/fp32 sparse copy would otherwise sit alongside them). Everything that
reads the correction afterwards must therefore read it from wherever it now
lives, not assume the raw buffers still hold it:

  * ``reconstruct_weight`` decodes the full dense weight - dropping the
    correction there is a SILENT numeric error, not a crash.
  * ``state_dict`` must serialize the correction whether or not a forward has
    run, so a save/load round-trip is not order-dependent.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from orka.inference.vq_linear import build_vq_linear
from orka.pipeline.pack import pack_checkpoint


def _pack_with_corrections(root: Path) -> tuple[Path, dict]:
    """slrq-block keeps one exact salient weight per block, so every packed
    tensor carries a non-empty CSR correction."""
    source = root / "model.json"
    rows = lambda v: [[float(v + i + j) for j in range(16)] for i in range(4)]
    source.write_text(
        json.dumps({"tensors": {"model.layers.0.self_attn.q_proj.weight": rows(1)}})
    )
    manifest = pack_checkpoint(
        source,
        root / "out.orka",
        group_size=8,
        codebook_size=4,
        iterations=2,
        codebook_mode="per-tensor",
        backend="numpy",
        em_aq_passes=0,
        block_scale_size=8,
        normalization="slrq-block",
    )
    return root / "out.orka", manifest["tensors"][0]


class CorrectionSurvivesSparseCacheTest(unittest.TestCase):
    def _layer(self, root: Path):
        artifact_dir, meta = _pack_with_corrections(root)
        layer = build_vq_linear(
            artifact_dir=artifact_dir, tensor_meta=meta, bias=None, device="cpu"
        )
        self.assertGreater(
            int(layer.corr_col.numel()), 0, "fixture must produce a real correction"
        )
        return layer

    def test_reconstruct_weight_unchanged_after_forward_caches_sparse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layer = self._layer(Path(tmp))
            before = layer.reconstruct_weight().clone()

            # What every correction-carrying forward does (dispatch.py N=1/N>1,
            # _forward_planed eager fallback): build + cache the sparse tensor.
            self.assertIsNotNone(layer._correction_sparse())
            self.assertEqual(int(layer.corr_col.numel()), 0, "raw buffers are freed")

            after = layer.reconstruct_weight()
            torch.testing.assert_close(
                after,
                before,
                msg="reconstruct_weight dropped the correction after a forward",
            )

    def test_state_dict_keeps_correction_after_forward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layer = self._layer(Path(tmp))
            nnz = int(layer.corr_col.numel())
            val_sum = float(layer.corr_val.float().abs().sum())

            layer._correction_sparse()
            sd = layer.state_dict()

            self.assertEqual(int(sd["corr_col"].numel()), nnz)
            self.assertEqual(int(sd["corr_rowptr"].numel()), layer.out_features + 1)
            self.assertAlmostEqual(
                float(sd["corr_val"].float().abs().sum()), val_sum, places=3
            )
            self.assertEqual(sd["corr_col"].dtype, torch.int32)
            self.assertEqual(sd["corr_val"].dtype, torch.float16)

    def test_state_dict_round_trip_preserves_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir, meta = _pack_with_corrections(root)
            layer = build_vq_linear(
                artifact_dir=artifact_dir, tensor_meta=meta, bias=None, device="cpu"
            )
            expected = layer.reconstruct_weight().clone()
            layer._correction_sparse()

            fresh = build_vq_linear(
                artifact_dir=artifact_dir, tensor_meta=meta, bias=None, device="cpu"
            )
            fresh.load_state_dict(layer.state_dict())
            torch.testing.assert_close(
                fresh.reconstruct_weight(),
                expected,
                msg="state_dict round-trip lost the correction",
            )


if __name__ == "__main__":
    unittest.main()
