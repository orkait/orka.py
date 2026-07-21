"""Regression locks for the codemode follow-up fixes.

  * is_quant_candidate is one predicate, not three copies.
  * BackgroundWriter starts lazily and still flushes when never started.
  * _warn_once surfaces an accelerator fallback exactly once.
  * mse_scale reports the outcome, not the request.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path

from orka._runtime.io import BackgroundWriter
from orka.core._checkpoint import is_quant_candidate
from orka.core._util import _WARNED_ONCE, _warn_once
from orka.pipeline.pack import pack_checkpoint


class QuantCandidateTest(unittest.TestCase):
    def test_shape_and_marker_rules(self) -> None:
        self.assertTrue(is_quant_candidate("model.layers.0.mlp.up_proj.weight", [8, 8]))
        self.assertFalse(is_quant_candidate("model.layers.0.mlp.up_proj.bias", [8]))
        self.assertFalse(is_quant_candidate("x.weight", [8]))
        for marker in (".bias", ".norm", ".layernorm", "rotary_emb", "attention.bias"):
            self.assertFalse(
                is_quant_candidate(f"model.layers.0{marker}.weight", [8, 8]), marker
            )

    def test_pack_and_allocate_share_the_predicate(self) -> None:
        import orka.pipeline.pack as pack_mod
        import orka.quant.allocate as alloc_mod

        self.assertIs(pack_mod.is_quant_candidate, is_quant_candidate)
        self.assertIs(alloc_mod.is_quant_candidate, is_quant_candidate)


class BackgroundWriterLazyStartTest(unittest.TestCase):
    def test_thread_starts_on_first_submit_only(self) -> None:
        writer = BackgroundWriter()
        self.assertIsNone(writer.thread)

        seen = []
        writer.submit(seen.append, 1)
        self.assertIsNotNone(writer.thread)
        writer.wait()
        self.assertEqual(seen, [1])
        writer.stop()

    def test_wait_and_stop_are_safe_when_never_started(self) -> None:
        writer = BackgroundWriter()
        writer.wait()
        writer.stop()
        self.assertIsNone(writer.thread)


class WarnOnceTest(unittest.TestCase):
    def test_warns_once_per_key(self) -> None:
        key = "test.warn_once.fixture"
        _WARNED_ONCE.discard(key)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _warn_once(key, "first")
                _warn_once(key, "second")
            self.assertEqual(len(caught), 1)
            self.assertIn("first", str(caught[0].message))
        finally:
            _WARNED_ONCE.discard(key)


def _pack(root: Path, **overrides) -> dict:
    source = root / "model.json"
    source.write_text(
        json.dumps(
            {
                "tensors": {
                    "model.layers.0.mlp.up_proj.weight": [
                        [float(i + j) for j in range(16)] for i in range(4)
                    ]
                }
            }
        )
    )
    kwargs = dict(
        group_size=8,
        codebook_size=4,
        iterations=2,
        codebook_mode="per-tensor",
        backend="numpy",
        em_aq_passes=0,
        block_scale_size=8,
    )
    kwargs.update(overrides)
    return pack_checkpoint(source, root / "out.orka", **kwargs)


class MseScaleTruthfulnessTest(unittest.TestCase):
    def test_manifest_flag_downgraded_when_preconditions_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _pack(Path(tmp), normalization="none", mse_scale=True)
            self.assertFalse(
                manifest["mse_scale"],
                "mse_scale needs a block-scale normalization; flag must downgrade",
            )

    def test_manifest_flag_kept_when_preconditions_met(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _pack(Path(tmp), normalization="block-max", mse_scale=True)
            self.assertTrue(manifest["mse_scale"])
            self.assertTrue(
                all(t.get("mse_scale_applied") for t in manifest["tensors"]),
                "per-tensor outcome must be recorded when refinement ran",
            )

    def test_key_absent_when_mse_scale_not_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _pack(Path(tmp), normalization="block-max")
            for tensor in manifest["tensors"]:
                self.assertNotIn("mse_scale_applied", tensor)


if __name__ == "__main__":
    unittest.main()
