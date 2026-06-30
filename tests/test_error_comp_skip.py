"""The error-compensation gate must skip the output head and recurrent/SSM layers.

Block-OBS minimises the LINEAR output error E[(Wx-What x)^2], valid only when the layer
feeds a (locally) linear path - attention/MLP projections. It is wrong for the output head
(downstream softmax) and recurrent/SSM projections (downstream nonlinear scan); applying it
there made perplexity WORSE on FalconH1-0.5B. The decision is delegated to the shared
ArchProfile (orka.quant.arch) - the identification source of truth - so this test checks the
GATE wiring; the detection contract itself is locked in test_arch.py.
"""
import unittest

from orka.quant import ArchProfile
from orka.pipeline.strategies.error_compensation import maybe_compensate_candidate


class ErrorCompGateTest(unittest.TestCase):
    def test_skips_head_and_recurrent_via_profile(self):
        # structural profile: lm_head by vocab-width, mamba/mixer by sibling A_log
        shapes = {
            "lm_head.weight": (1000, 64),
            "net.0.mixer.A_log": (8,),
            "net.0.mixer.in_proj.weight": (128, 64),
            "net.0.self_attn.q_proj.weight": (64, 64),
        }
        profile = ArchProfile.from_shapes(shapes, vocab_size=1000)
        for name in ("lm_head.weight", "net.0.mixer.in_proj.weight"):
            self.assertFalse(
                maybe_compensate_candidate(
                    {"name": name}, backend="torch", awq_activations={name: object()},
                    resolved_device="cpu", progress_file=None, out_dir=None, profile=profile,
                ),
                name,
            )

    def test_name_fallback_when_no_profile(self):
        # no profile -> name-based fallback still catches the head / mamba-named layer
        for name in ("lm_head.weight", "model.layers.0.mamba.in_proj.weight"):
            self.assertFalse(
                maybe_compensate_candidate(
                    {"name": name}, backend="torch", awq_activations={name: object()},
                    resolved_device="cpu", progress_file=None, out_dir=None,
                ),
                name,
            )

    def test_global_preconditions_short_circuit(self):
        # non-torch backend / no activations -> not compensated (handled once in pack)
        c = {"name": "model.layers.0.self_attn.q_proj.weight"}
        self.assertFalse(maybe_compensate_candidate(
            c, backend="numpy", awq_activations={}, resolved_device="cpu",
            progress_file=None, out_dir=None))
        self.assertFalse(maybe_compensate_candidate(
            c, backend="torch", awq_activations=None, resolved_device="cpu",
            progress_file=None, out_dir=None))


if __name__ == "__main__":
    unittest.main()
