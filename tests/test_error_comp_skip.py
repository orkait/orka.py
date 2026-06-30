"""Error-compensation must skip the output head and recurrent/SSM layers.

Block-OBS minimises the LINEAR output error E[(Wx-What x)^2], valid only when the
layer feeds a (locally) linear path - attention/MLP projections. It is wrong for the
output head (downstream softmax) and recurrent/SSM projections (downstream nonlinear
scan). Applying it there made perplexity WORSE than plain VQ and caused degenerate
generation on FalconH1-0.5B. These layers must fall back to plain VQ.

The skip is structural: it routes through orka.quant.family (is_output_head /
is_recurrent_block), one source of truth shared with the weight quantizer, instead of
a hardcoded substring tuple. The old tuple was ('lm_head','embed_out','mamba'); a
pure-Mamba model names its block '...mixer.in_proj' (no 'mamba' substring), so the old
check silently MISSED it and re-broke perplexity. The mixer case below locks that.
"""
import unittest

from orka.pipeline.strategies.error_compensation import (
    _error_comp_skip_reason,
    maybe_compensate_candidate,
)


class ErrorCompSkipTest(unittest.TestCase):
    def test_skips_output_head(self):
        for name in ("lm_head.weight", "model.embed_out.weight", "transformer.output.weight"):
            self.assertIsNotNone(_error_comp_skip_reason(name), name)
            self.assertIn("head", _error_comp_skip_reason(name))

    def test_skips_recurrent_ssm_across_naming_conventions(self):
        for name in (
            "model.layers.0.mamba.in_proj.weight",       # FalconH1 naming
            "model.layers.5.mamba.out_proj.weight",
            "backbone.layers.0.mixer.in_proj.weight",    # pure-Mamba naming (old check MISSED)
            "model.layers.2.mixer.out_proj.weight",
            "net.blocks.3.ssm.proj.weight",
        ):
            reason = _error_comp_skip_reason(name)
            self.assertIsNotNone(reason, name)
            self.assertIn("recurrent", reason)

    def test_keeps_standard_projections(self):
        for name in (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.feed_forward.gate_proj.weight",
        ):
            self.assertIsNone(_error_comp_skip_reason(name), name)

    def test_candidate_short_circuits_for_skipped_layer(self):
        # A skipped layer returns False (not compensated) before touching activations,
        # even with torch backend - so it falls back to the plain VQ assignment.
        for name in ("lm_head.weight", "backbone.layers.0.mixer.in_proj.weight"):
            c = {"name": name}
            self.assertFalse(
                maybe_compensate_candidate(
                    c, backend="torch", awq_activations={name: object()},
                    resolved_device="cpu", progress_file=None, out_dir=None,
                ),
                name,
            )

    def test_structural_skip_set_is_authoritative(self):
        # When pack passes the structurally-resolved skip set, it wins even for a name the
        # heuristic would NOT catch (e.g. an oddly-named SSM linear). This is the robust
        # path: detection by shape / sibling state params, not by the layer's own name.
        odd = "net.block.0.recurrence_unit.proj_in.weight"
        self.assertIsNone(_error_comp_skip_reason(odd))  # name heuristic misses it
        c = {"name": odd}
        self.assertFalse(
            maybe_compensate_candidate(
                c, backend="torch", awq_activations={odd: object()},
                resolved_device="cpu", progress_file=None, out_dir=None,
                skip_names={odd},  # structural set says skip -> authoritative
            )
        )


if __name__ == "__main__":
    unittest.main()
