"""ArchProfile: the single source of truth for per-tensor identification (output head /
vocab-width embedding / recurrent-SSM), structural-primary with name fallback.

These lock the contract every pipeline depends on: detection keys on what a tensor IS
(vocab-width shape, sibling recurrence state params) not what it is NAMED, so it survives
mlp->feed_forward, mamba->mixer, etc.
"""
import unittest

import torch
import torch.nn as nn

from orka.quant import (
    ArchProfile,
    is_output_head,
    is_recurrent_block,
    output_head_names,
    recurrent_block_names,
)


class StructuralPrimitiveTest(unittest.TestCase):
    def test_output_head_names_by_vocab_width(self):
        shapes = {
            "model.embed_tokens.weight": (32784, 1024),
            "lm_head.weight": (32784, 1024),
            "model.layers.0.feed_forward.gate_proj.weight": (2048, 1024),
            "model.layers.0.mamba.A_log": (24,),
        }
        self.assertEqual(output_head_names(shapes, 32784),
                         {"model.embed_tokens.weight", "lm_head.weight"})
        self.assertEqual(output_head_names(shapes, None),  # dominant dim == vocab
                         {"model.embed_tokens.weight", "lm_head.weight"})

    def test_recurrent_by_sibling_state_params_naming_agnostic(self):
        names = [
            "model.layers.0.mamba.A_log", "model.layers.0.mamba.in_proj.weight",
            "model.layers.0.mamba.out_proj.weight",
            "backbone.layers.1.mixer.dt_bias", "backbone.layers.1.mixer.in_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ]
        rec = recurrent_block_names(names)
        self.assertIn("model.layers.0.mamba.in_proj.weight", rec)
        self.assertIn("backbone.layers.1.mixer.in_proj.weight", rec)   # 'mixer', not 'mamba'
        self.assertNotIn("model.layers.0.self_attn.q_proj.weight", rec)

    def test_name_fallbacks(self):
        self.assertTrue(is_output_head("lm_head.weight"))
        self.assertFalse(is_output_head("model.layers.0.mlp.down_proj.weight"))
        self.assertTrue(is_recurrent_block("x.mamba.in_proj.weight"))
        self.assertFalse(is_recurrent_block("x.self_attn.q_proj.weight"))


class ArchProfileFromShapesTest(unittest.TestCase):
    def setUp(self):
        self.shapes = {
            "lm_head.weight": (50000, 512),
            "model.embed_tokens.weight": (50000, 512),
            "model.layers.0.self_attn.q_proj.weight": (512, 512),
            "model.layers.0.feed_forward.down_proj.weight": (512, 1024),
            "model.layers.0.mamba.A_log": (16,),
            "model.layers.0.mamba.in_proj.weight": (2048, 512),
            "model.layers.0.mamba.out_proj.weight": (512, 1024),
        }
        self.p = ArchProfile.from_shapes(self.shapes, vocab_size=50000)

    def test_head_detection_structural_and_weight_suffix_agnostic(self):
        # works whether queried with the state_dict name or a bare module path
        self.assertTrue(self.p.is_output_head("lm_head.weight"))
        self.assertTrue(self.p.is_output_head("lm_head"))
        self.assertTrue(self.p.is_output_head("model.embed_tokens.weight"))
        self.assertFalse(self.p.is_output_head("model.layers.0.self_attn.q_proj.weight"))

    def test_head_detection_by_shape_when_name_unknown(self):
        p = ArchProfile.from_shapes({}, vocab_size=50000)  # no resolved set
        self.assertTrue(p.is_output_head("weirdly.named.head.weight", (50000, 512)))
        self.assertFalse(p.is_output_head("weirdly.named.proj.weight", (512, 512)))

    def test_recurrent_detection(self):
        self.assertTrue(self.p.is_recurrent("model.layers.0.mamba.in_proj.weight"))
        self.assertTrue(self.p.is_recurrent("model.layers.0.mamba.out_proj.weight"))
        self.assertFalse(self.p.is_recurrent("model.layers.0.self_attn.q_proj.weight"))
        self.assertFalse(self.p.is_recurrent("model.layers.0.feed_forward.down_proj.weight"))

    def test_error_comp_skip_reason(self):
        self.assertIn("head", self.p.error_comp_skip_reason("lm_head.weight"))
        self.assertIn("recurrent", self.p.error_comp_skip_reason("model.layers.0.mamba.in_proj.weight"))
        self.assertIsNone(self.p.error_comp_skip_reason("model.layers.0.self_attn.q_proj.weight"))
        self.assertIsNone(self.p.error_comp_skip_reason("model.layers.0.feed_forward.down_proj.weight"))

    def test_empty_profile_falls_back_to_names(self):
        p = ArchProfile.from_shapes({})
        self.assertIsNotNone(p.error_comp_skip_reason("lm_head.weight"))            # name fallback
        self.assertIsNotNone(p.error_comp_skip_reason("x.mamba.in_proj.weight"))    # name fallback
        self.assertIsNone(p.error_comp_skip_reason("x.self_attn.q_proj.weight"))


class ArchProfileFromModelTest(unittest.TestCase):
    def test_head_by_identity_and_recurrent_by_params(self):
        # tiny module mimicking an SSM-hybrid: head Linear + a mamba block with A_log
        m = nn.Module()
        m.config = type("C", (), {"vocab_size": 64})()
        block = nn.Module()
        block.register_parameter("A_log", nn.Parameter(torch.zeros(8)))
        block.in_proj = nn.Linear(16, 32)
        block.out_proj = nn.Linear(16, 16)
        layer = nn.Module(); layer.mixer = block
        layers = nn.ModuleList([layer])
        m.layers = layers
        m.lm_head = nn.Linear(16, 64)
        m.get_output_embeddings = lambda: m.lm_head

        p = ArchProfile.from_model(m)
        self.assertTrue(p.is_output_head("lm_head"))                 # by get_output_embeddings identity
        self.assertTrue(p.is_recurrent("layers.0.mixer.in_proj"))    # by sibling A_log, 'mixer' naming
        self.assertTrue(p.is_recurrent("layers.0.mixer.out_proj"))
        self.assertFalse(p.is_output_head("layers.0.mixer.in_proj"))


if __name__ == "__main__":
    unittest.main()
