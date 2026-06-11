from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orka.pipeline.pack import pack_checkpoint
from orka.verify import verify_artifact


def _write_multi_family_source(path: Path) -> None:
    rows = lambda v: [[float(v + i + j) for j in range(16)] for i in range(4)]
    path.write_text(
        json.dumps(
            {
                "tensors": {
                    "model.layers.0.self_attn.q_proj.weight": rows(1),
                    "model.layers.0.mlp.up_proj.weight": rows(2),
                }
            }
        )
    )


class SharedCodebookGroupSizeTest(unittest.TestCase):
    def test_global_mode_packs_multi_family_model(self) -> None:
        """Dynamic family group sizing must not break shared codebooks.

        Regression: attention/mlp family overrides produced mixed vector
        widths, crashing the global-codebook concat with a ValueError.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "model.json"
            _write_multi_family_source(source)
            for mode in ("global", "family"):
                manifest = pack_checkpoint(
                    source,
                    root / f"{mode}.orka",
                    group_size=8,
                    codebook_size=4,
                    iterations=2,
                    codebook_mode=mode,
                    backend="numpy",
                    em_aq_passes=0,
                )
                self.assertEqual(manifest["tensor_count"], 2)
                self.assertFalse(manifest["dynamic_group_sizing"])
                for entry in manifest["tensors"]:
                    self.assertEqual(entry["group_size"], 8)

    def test_per_tensor_mode_keeps_family_group_sizing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "model.json"
            _write_multi_family_source(source)
            manifest = pack_checkpoint(
                source,
                root / "pt.orka",
                group_size=8,
                codebook_size=4,
                iterations=2,
                codebook_mode="per-tensor",
                backend="numpy",
                em_aq_passes=0,
            )
            self.assertTrue(manifest["dynamic_group_sizing"])
            by_name = {t["name"]: t for t in manifest["tensors"]}
            self.assertEqual(
                by_name["model.layers.0.self_attn.q_proj.weight"]["group_size"], 4
            )
            self.assertEqual(
                by_name["model.layers.0.mlp.up_proj.weight"]["group_size"], 16
            )


class CodebookCacheHitTest(unittest.TestCase):
    def test_cache_hit_repack_round_trips(self) -> None:
        """Regression: stage-0 cache hit raised UnboundLocalError on 'vw'."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "model.json"
            _write_multi_family_source(source)
            cache = root / "cbcache"
            kwargs = dict(
                group_size=8,
                codebook_size=4,
                iterations=2,
                codebook_mode="per-tensor",
                backend="numpy",
                em_aq_passes=0,
                codebook_cache_dir=cache,
            )
            first = pack_checkpoint(source, root / "r1.orka", **kwargs)
            second = pack_checkpoint(source, root / "r2.orka", **kwargs)
            self.assertEqual(first["tensor_count"], second["tensor_count"])
            verified = verify_artifact(root / "r2.orka")
            self.assertEqual(verified["verified_tensors"], 2)
            self.assertLess(verified["max_mse_delta"], 1e-6)


if __name__ == "__main__":
    unittest.main()
