"""Error-compensation must skip the output head and SSM/mamba layers.

Block-OBS minimises the LINEAR output error E[(Wx-What x)^2], valid only when the
layer feeds a (locally) linear path - attention/MLP projections. It is wrong for the
output head (downstream softmax) and mamba/SSM projections (downstream nonlinear
scan). Applying it there made perplexity WORSE than plain VQ and caused degenerate
generation on FalconH1-0.5B. These layers must fall back to plain VQ.
"""
import unittest

from orka.pipeline.strategies.error_compensation import (
    _skip_error_comp,
    maybe_compensate_candidate,
)


class ErrorCompSkipTest(unittest.TestCase):
    def test_skips_output_head_and_mamba(self):
        for name in (
            "lm_head.weight",
            "model.embed_out.weight",
            "model.layers.0.mamba.in_proj.weight",
            "model.layers.5.mamba.out_proj.weight",
        ):
            self.assertTrue(_skip_error_comp(name), name)

    def test_keeps_standard_projections(self):
        for name in (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.feed_forward.gate_proj.weight",
        ):
            self.assertFalse(_skip_error_comp(name), name)

    def test_candidate_short_circuits_for_skipped_layer(self):
        # A skipped layer returns False (not compensated) before touching activations,
        # even with torch backend - so it falls back to the plain VQ assignment.
        c = {"name": "lm_head.weight"}
        self.assertFalse(
            maybe_compensate_candidate(
                c, backend="torch", awq_activations={"lm_head.weight": object()},
                resolved_device="cpu", progress_file=None, out_dir=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
