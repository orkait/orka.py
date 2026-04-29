import json
import tempfile
import unittest
from pathlib import Path

from orka import (
    _parse_params,
    _summarize_eval_rows,
    build_parser,
    estimate_payload,
    inspect_checkpoint,
    pack_checkpoint,
    verify_artifact,
)


class OrkaSelfTests(unittest.TestCase):
    def test_calc_size_for_vq8_eight_b_model(self) -> None:
        estimate = estimate_payload(
            params=8_030_000_000, group_size=8, codebook_size=256
        )
        self.assertEqual(estimate.index_bits, 8)
        self.assertEqual(estimate.vector_count, 1_003_750_000)
        self.assertEqual(estimate.index_bytes, 1_003_750_000)
        self.assertAlmostEqual(estimate.bits_per_weight, 1.0)

    def test_parse_decimal_param_suffix_without_float_rounding(self) -> None:
        self.assertEqual(_parse_params("8.03b"), 8_030_000_000)

    def test_calc_size_for_vq16_two_byte_indices(self) -> None:
        estimate = estimate_payload(
            params=8_030_000_000, group_size=8, codebook_size=8192
        )
        self.assertEqual(estimate.index_bits, 13)
        self.assertEqual(estimate.vector_count, 1_003_750_000)
        self.assertEqual(estimate.index_bytes, 1_631_093_750)

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

    def test_eval_summary_reports_loss_delta_and_perplexity_ratio(self) -> None:
        rows = [
            {"prompt": "a", "token_count": 3, "original_loss": 1.0, "orka_loss": 1.5},
            {"prompt": "b", "token_count": 1, "original_loss": 2.0, "orka_loss": 3.0},
        ]
        summary = _summarize_eval_rows(rows)
        self.assertEqual(summary["prompt_count"], 2)
        self.assertEqual(summary["token_count"], 4)
        self.assertAlmostEqual(summary["original_loss"], 1.25)
        self.assertAlmostEqual(summary["orka_loss"], 1.875)
        self.assertAlmostEqual(summary["loss_delta"], 0.625)
        self.assertGreater(summary["perplexity_ratio"], 1.0)

    def test_eval_command_is_registered(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["eval", "model.orka", "--prompts", "prompts.txt", "--out", "eval.json"]
        )
        self.assertEqual(args.command, "eval")
        self.assertEqual(args.artifact, "model.orka")
        self.assertEqual(args.prompts, "prompts.txt")

    def test_eval_sweep_command_is_registered(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "eval-sweep",
                "sweep.json",
                "--prompts",
                "prompts.txt",
                "--out",
                "eval-sweep.json",
                "--model-dir",
                "model",
                "--device",
                "cuda",
                "--max-runs",
                "3",
            ]
        )
        self.assertEqual(args.command, "eval-sweep")
        self.assertEqual(args.sweep, "sweep.json")
        self.assertEqual(args.prompts, "prompts.txt")
        self.assertEqual(args.max_runs, 3)


def run_selftests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OrkaSelfTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run_selftests())
