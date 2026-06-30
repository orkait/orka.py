from __future__ import annotations

import unittest

from orka.quant import (
    classify_tensor_family,
    is_output_head,
    is_recurrent_block,
    output_head_names,
    recurrent_block_names,
)


class StructuralDetectionTest(unittest.TestCase):
    """Structural (shape / sibling-param) detection - the robust primary, vs the name
    fallbacks below. Keys on what tensors ARE, so it survives non-standard naming."""

    def test_output_head_by_vocab_width(self):
        shapes = {
            "model.embed_tokens.weight": (32784, 1024),   # embedding (vocab-width)
            "lm_head.weight": (32784, 1024),              # head (vocab-width)
            "model.layers.0.feed_forward.gate_proj.weight": (2048, 1024),
            "model.layers.0.self_attn.q_proj.weight": (512, 1024),
            "model.layers.0.mamba.A_log": (24,),          # 1-D, ignored
        }
        # explicit vocab
        self.assertEqual(
            output_head_names(shapes, vocab_size=32784),
            {"model.embed_tokens.weight", "lm_head.weight"},
        )
        # no config -> dominant output dim is vocab; same result
        self.assertEqual(output_head_names(shapes, None),
                         {"model.embed_tokens.weight", "lm_head.weight"})

    def test_recurrent_by_sibling_state_params(self):
        # mamba block owns A_log/dt_bias -> its in/out_proj are recurrent; a model that
        # names the same block 'mixer' is caught identically (no name dependence).
        names = [
            "model.layers.0.mamba.A_log", "model.layers.0.mamba.dt_bias",
            "model.layers.0.mamba.in_proj.weight", "model.layers.0.mamba.out_proj.weight",
            "model.layers.0.mamba.conv1d.weight",
            "backbone.layers.1.mixer.A_log", "backbone.layers.1.mixer.in_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.feed_forward.gate_proj.weight",
        ]
        rec = recurrent_block_names(names)
        self.assertIn("model.layers.0.mamba.in_proj.weight", rec)
        self.assertIn("model.layers.0.mamba.out_proj.weight", rec)
        self.assertIn("backbone.layers.1.mixer.in_proj.weight", rec)   # 'mixer' naming
        self.assertNotIn("model.layers.0.self_attn.q_proj.weight", rec)
        self.assertNotIn("model.layers.0.feed_forward.gate_proj.weight", rec)

    def test_recurrent_empty_without_state_params(self):
        # a pure transformer has no recurrence state params -> nothing skipped
        names = ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.up_proj.weight"]
        self.assertEqual(recurrent_block_names(names), set())


class HeadAndRecurrentPredicateTest(unittest.TestCase):
    """Shared name predicates (one source of truth for the quantizer + the error-comp
    gate). is_output_head -> keep fp16 / skip OBS; is_recurrent_block -> skip OBS."""

    def test_is_output_head(self) -> None:
        for n in ("lm_head.weight", "model.embed_out.weight", "transformer.output.weight"):
            self.assertTrue(is_output_head(n), n)
        for n in ("model.layers.0.mlp.down_proj.weight", "model.embed_tokens.weight"):
            self.assertFalse(is_output_head(n), n)

    def test_is_recurrent_block_across_conventions(self) -> None:
        # Must catch both FalconH1 ('mamba') and pure-Mamba ('mixer') naming - the old
        # 'mamba'-only substring missed mixer and re-broke perplexity.
        for n in (
            "model.layers.0.mamba.in_proj.weight",
            "backbone.layers.0.mixer.in_proj.weight",
            "net.blocks.3.ssm.proj.weight",
        ):
            self.assertTrue(is_recurrent_block(n), n)
        for n in (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.feed_forward.gate_proj.weight",
        ):
            self.assertFalse(is_recurrent_block(n), n)


class TensorFamilyClassificationTest(unittest.TestCase):
    def test_gate_proj_is_mlp_not_router(self) -> None:
        self.assertEqual(
            classify_tensor_family("model.layers.0.mlp.gate_proj.weight"),
            "mlp",
        )
        self.assertEqual(
            classify_tensor_family("model.layers.0.mlp.up_proj.weight"),
            "mlp",
        )
        self.assertEqual(
            classify_tensor_family("model.layers.0.mlp.down_proj.weight"),
            "mlp",
        )

    def test_router_names_still_classify_as_router(self) -> None:
        self.assertEqual(
            classify_tensor_family("model.layers.0.router.weight"),
            "router",
        )
        self.assertEqual(
            classify_tensor_family("model.layers.0.block_sparse_moe.gate.weight"),
            "router",
        )


if __name__ == "__main__":
    unittest.main()
