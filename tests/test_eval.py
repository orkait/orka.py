"""Eval summary aggregation + CLI registration."""

import unittest

from orka import _summarize_eval_rows, build_parser


class EvalTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
