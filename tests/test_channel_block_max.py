from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orka.pipeline.pack import pack_checkpoint
from orka.artifact.reconstruct import reconstruct_artifact
from orka.report import report_artifact
from orka.verify import verify_artifact


class ChannelBlockMaxRoundTripTest(unittest.TestCase):
    def test_channel_block_max_round_trips_through_public_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "model.json"
            source.write_text(
                json.dumps(
                    {
                        "tensors": {
                            "model.layers.0.self_attn.q_proj.weight": [
                                [1.0, 2.0, 128.0, 4.0],
                                [5.0, 6.0, 7.0, 8.0],
                                [9.0, 32.0, 11.0, 12.0],
                            ]
                        }
                    }
                )
            )

            artifact = root / "artifact.orka"
            try:
                manifest = pack_checkpoint(
                    source,
                    artifact,
                    group_size=2,
                    codebook_size=4,
                    iterations=2,
                    codebook_mode="per-tensor",
                    sample_vectors=None,
                    backend="numpy",
                    normalization="channel-block-max",
                    block_scale_size=2,
                    em_aq_passes=0,
                )
            except ValueError as exc:
                self.fail(f"channel-block-max should be supported by pack: {exc}")

            self.assertEqual(manifest["normalization"], "channel-block-max")
            tensor_meta = manifest["tensors"][0]
            self.assertEqual(tensor_meta["normalization"], "channel-block-max")
            self.assertEqual(tensor_meta["block_scale_size"], 2)
            self.assertIsNotNone(tensor_meta["scales"])
            self.assertGreater(tensor_meta["scale_count"], 0)

            report = report_artifact(artifact)
            self.assertEqual(report["normalization"], "channel-block-max")
            self.assertEqual(report["tensor_count"], 1)
            self.assertGreater(report["total_scale_bytes"], 0)

            verified = verify_artifact(artifact)
            self.assertEqual(verified["verified_tensors"], 1)
            self.assertLess(verified["max_mse_delta"], 1e-6)

            output = root / "reconstructed.json"
            reconstructed = reconstruct_artifact(artifact, output, output_format="json")
            self.assertEqual(reconstructed["tensor_count"], 1)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
