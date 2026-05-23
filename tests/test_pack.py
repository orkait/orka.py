"""Inspect + pack round-trips for safetensors and torch backends."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orka import (
    inspect_checkpoint,
    pack_checkpoint,
    merge_orka_artifacts,
    reconstruct_artifact,
    report_artifact,
    verify_artifact,
)


class InspectPackTests(unittest.TestCase):
    def test_safetensors_bfloat16_checkpoint_can_be_inspected(self) -> None:
        try:
            import torch
            from safetensors.torch import save_file
        except Exception as exc:
            self.skipTest(f"optional BF16 safetensors dependencies missing: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bf16.safetensors"
            save_file(
                {
                    "model.layers.0.mlp.down_proj.weight": torch.ones(
                        (2, 4), dtype=torch.bfloat16
                    )
                },
                str(path),
            )

            report = inspect_checkpoint(path)

            self.assertEqual(report["total_params"], 8)
            candidate_params = sum(
                tensor["numel"] for tensor in report["tensors"] if tensor["candidate"]
            )
            self.assertEqual(candidate_params, 8)
            self.assertEqual(report["tensors"][0]["shape"], [2, 4])

    def test_safetensors_bfloat16_checkpoint_can_be_packed_with_numpy(self) -> None:
        try:
            import torch
            from safetensors.torch import save_file
            import numpy  # noqa: F401
        except Exception as exc:
            self.skipTest(f"optional BF16 packing dependencies missing: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "bf16.safetensors"
            out = root / "bf16.orka"
            save_file(
                {
                    "model.layers.0.mlp.down_proj.weight": torch.arange(
                        8, dtype=torch.float32
                    )
                    .reshape(2, 4)
                    .to(torch.bfloat16)
                },
                str(source),
            )

            manifest = pack_checkpoint(
                source=source,
                out_dir=out,
                group_size=2,
                codebook_size=2,
                iterations=2,
                backend="numpy",
            )
            verify = verify_artifact(out)

            self.assertEqual(manifest["tensor_count"], 1)
            self.assertEqual(verify["verified_tensors"], 1)

    def test_torch_backend_packs_and_records_device(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception as exc:
            self.skipTest(f"torch backend dependency missing: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "tiny.json"
            out = root / "tiny-torch.orka"
            source.write_text(
                json.dumps(
                    {
                        "tensors": {
                            "linear.weight": [
                                [1.0, 1.0, 0.9, 1.1],
                                [-1.0, -1.0, -1.1, -0.9],
                            ],
                        }
                    }
                )
            )

            manifest = pack_checkpoint(
                source=source,
                out_dir=out,
                group_size=2,
                codebook_size=2,
                iterations=3,
                backend="torch",
                device="cpu",
            )
            verify = verify_artifact(out)

            self.assertEqual(manifest["backend"], "torch")
            self.assertEqual(manifest["device"], "cpu")
            self.assertEqual(manifest["tensor_count"], 1)
            self.assertEqual(verify["verified_tensors"], 1)

    def test_outlier_only_metrics_without_normalization_or_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outliers.json"
            out = root / "outliers.orka"
            source.write_text(
                json.dumps(
                    {
                        "tensors": {
                            "linear.weight": [
                                [1.0, 2.0, 3.0, 50.0],
                                [-1.0, -2.0, -3.0, -50.0],
                            ],
                        }
                    }
                )
            )

            manifest = pack_checkpoint(
                source=source,
                out_dir=out,
                group_size=2,
                codebook_size=2,
                iterations=1,
                backend="numpy",
                outlier_frac=0.25,
            )
            verify = verify_artifact(out)

            self.assertEqual(manifest["tensor_count"], 1)
            self.assertEqual(verify["verified_tensors"], 1)
            outliers = manifest["tensors"][0]["outliers"]
            self.assertEqual(outliers["positions_dtype"], "uint8")
            self.assertEqual(outliers["values_dtype"], "float16")
            self.assertEqual(outliers["positions_bytes"], outliers["count"])
            self.assertEqual(outliers["values_bytes"], outliers["count"] * 2)

    def test_per_tensor_pack_streams_manifest_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "streaming.json"
            out = root / "streaming.orka"
            source.write_text(
                json.dumps(
                    {
                        "tensors": {
                            "linear0.weight": [[1.0, 2.0], [3.0, 4.0]],
                            "linear1.weight": [[5.0, 6.0], [7.0, 8.0]],
                            "linear2.weight": [[9.0, 10.0], [11.0, 12.0]],
                        }
                    }
                )
            )

            with patch("orka.pipeline.pack._persist_manifest") as persist_manifest:
                persist_manifest.side_effect = AssertionError(
                    "per-tensor pack must stream manifest entries"
                )
                manifest = pack_checkpoint(
                    source=source,
                    out_dir=out,
                    group_size=2,
                    codebook_size=2,
                    iterations=1,
                    backend="numpy",
                    codebook_mode="per-tensor",
                    em_aq_passes=0,
                )

            verify = verify_artifact(out)

            self.assertEqual(manifest["tensor_count"], 3)
            self.assertEqual(verify["verified_tensors"], 3)
            self.assertFalse(persist_manifest.called)

    def test_slrq_manifest_metrics_match_verify_metrics(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "slrq_metrics.json"
            out = root / "slrq_metrics.orka"
            data = np.array([1.0] * 16, dtype=np.float32).reshape(2, 8)
            data[0, 5] = 100.0
            source.write_text(json.dumps({"tensors": {"linear.weight": data.tolist()}}))

            manifest = pack_checkpoint(
                source=source,
                out_dir=out,
                group_size=8,
                codebook_size=2,
                iterations=1,
                backend="numpy",
                normalization="slrq-block",
                block_scale_size=8,
            )
            verify = verify_artifact(out)
            tensor = manifest["tensors"][0]
            worst = verify["worst_tensors"][0]

            self.assertAlmostEqual(tensor["mse"], worst["mse"], places=6)
            self.assertAlmostEqual(
                tensor["cosine_similarity"], worst["cosine_similarity"], places=6
            )

    def test_hadamard_manifest_metrics_match_verify_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "hadamard_metrics.json"
            out = root / "hadamard_metrics.orka"
            source.write_text(
                json.dumps(
                    {
                        "tensors": {
                            "linear.weight": [
                                [1.0, 2.0, 3.0, 4.0],
                                [-1.0, -2.0, -3.0, -4.0],
                            ]
                        }
                    }
                )
            )

            manifest = pack_checkpoint(
                source=source,
                out_dir=out,
                group_size=4,
                codebook_size=2,
                iterations=1,
                backend="numpy",
                rotation="hadamard",
            )
            verify = verify_artifact(out)
            tensor = manifest["tensors"][0]
            worst = verify["worst_tensors"][0]

            self.assertEqual(tensor["rotation"], "hadamard")
            self.assertAlmostEqual(tensor["mse"], worst["mse"], places=6)
            self.assertAlmostEqual(
                tensor["cosine_similarity"], worst["cosine_similarity"], places=6
            )

    def test_report_includes_slrq_salient_bytes(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "slrq_report.json"
            out = root / "slrq_report.orka"
            data = np.array([1.0] * 16, dtype=np.float32).reshape(2, 8)
            data[0, 3] = 100.0
            source.write_text(json.dumps({"tensors": {"linear.weight": data.tolist()}}))

            manifest = pack_checkpoint(
                source=source,
                out_dir=out,
                group_size=8,
                codebook_size=2,
                iterations=1,
                backend="numpy",
                normalization="slrq-block",
                block_scale_size=8,
            )
            report = report_artifact(out)
            salient = manifest["tensors"][0]["salient"]

            self.assertGreater(report["total_salient_bytes"], 0)
            self.assertEqual(salient["indices_dtype"], "uint8")
            self.assertEqual(salient["weights_dtype"], "float16")
            self.assertEqual(salient["indices_bytes"], salient["count"])
            self.assertEqual(salient["weights_bytes"], salient["count"] * 2)

    def test_verify_counts_passthrough_tensors(self) -> None:
        try:
            import safetensors  # noqa: F401
        except Exception as exc:
            self.skipTest(f"optional safetensors dependency missing: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "passthrough_verify.json"
            out = root / "passthrough_verify.orka"
            source.write_text(
                json.dumps(
                    {
                        "tensors": {
                            "linear.weight": [[1.0, 0.0], [0.5, -0.5]],
                            "linear.bias": [1.0, 2.0],
                        }
                    }
                )
            )

            pack_checkpoint(
                source=source,
                out_dir=out,
                group_size=2,
                codebook_size=2,
                iterations=1,
                backend="numpy",
            )
            verify = verify_artifact(out)

            self.assertEqual(verify["verified_tensors"], 1)
            self.assertEqual(verify["verified_passthrough_tensors"], 1)

    def test_reconstruct_safetensors_includes_passthrough_tensors(self) -> None:
        try:
            import safetensors  # noqa: F401
        except Exception as exc:
            self.skipTest(f"optional safetensors dependency missing: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "reconstruct.json"
            out = root / "reconstruct.orka"
            recon = root / "reconstructed.safetensors"
            source.write_text(
                json.dumps(
                    {
                        "tensors": {
                            "linear.weight": [[1.0, 0.0], [0.5, -0.5]],
                            "linear.bias": [1.0, 2.0],
                        }
                    }
                )
            )

            pack_checkpoint(
                source=source,
                out_dir=out,
                group_size=2,
                codebook_size=2,
                iterations=1,
                backend="numpy",
            )
            result = reconstruct_artifact(out, recon, output_format="safetensors")

            self.assertEqual(result["tensor_count"], 2)

    def test_partition_and_merge_reconstructs_full_model(self) -> None:
        try:
            import safetensors  # noqa: F401
        except Exception as exc:
            self.skipTest(f"optional safetensors dependency missing: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "partition.json"
            part0 = root / "part0.orka"
            part1 = root / "part1.orka"
            merged = root / "merged.orka"
            source.write_text(
                json.dumps(
                    {
                        "tensors": {
                            "linear0.weight": [[1.0, 2.0], [3.0, 4.0]],
                            "linear1.weight": [[5.0, 6.0], [7.0, 8.0]],
                            "linear2.weight": [[9.0, 10.0], [11.0, 12.0]],
                            "linear.bias": [1.0, 2.0],
                        }
                    }
                )
            )

            manifest0 = pack_checkpoint(
                source=source,
                out_dir=part0,
                group_size=2,
                codebook_size=2,
                iterations=1,
                backend="numpy",
                tensor_partition_count=2,
                tensor_partition_index=0,
            )
            manifest1 = pack_checkpoint(
                source=source,
                out_dir=part1,
                group_size=2,
                codebook_size=2,
                iterations=1,
                backend="numpy",
                tensor_partition_count=2,
                tensor_partition_index=1,
            )

            self.assertEqual(len(manifest0["tensors"]), 2)
            self.assertEqual(len(manifest1["tensors"]), 1)

            merged_manifest = merge_orka_artifacts(
                input_artifacts=[part0, part1], out_dir=merged
            )
            self.assertEqual(merged_manifest["tensor_count"], 3)

            verify = verify_artifact(merged)
            self.assertEqual(verify["verified_tensors"], 3)
            self.assertEqual(verify["verified_passthrough_tensors"], 1)

    def test_awq_normalization_is_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "awq_disabled.json"
            out = root / "awq_disabled.orka"
            source.write_text(
                json.dumps({"tensors": {"linear.weight": [[1.0, 2.0], [3.0, 4.0]]}})
            )

            with patch.dict(os.environ, {"ORKA_ENABLE_AWQ": ""}):
                with self.assertRaisesRegex(RuntimeError, "AWQ support is disabled"):
                    pack_checkpoint(
                        source=source,
                        out_dir=out,
                        group_size=2,
                        codebook_size=2,
                        iterations=1,
                        backend="numpy",
                        normalization="awq",
                    )

    def test_awq_activation_weights_are_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "awq_weights_disabled.json"
            out = root / "awq_weights_disabled.orka"
            source.write_text(
                json.dumps({"tensors": {"linear.weight": [[1.0, 2.0], [3.0, 4.0]]}})
            )

            with patch.dict(os.environ, {"ORKA_ENABLE_AWQ": ""}):
                with self.assertRaisesRegex(RuntimeError, "AWQ support is disabled"):
                    pack_checkpoint(
                        source=source,
                        out_dir=out,
                        group_size=2,
                        codebook_size=2,
                        iterations=1,
                        backend="numpy",
                        awq_activations={"linear.weight": [[1.0, 1.0]]},
                    )


if __name__ == "__main__":
    unittest.main()
