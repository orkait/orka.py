"""Pluggable post-assignment strategy contract.

Locks the Strategy-pattern wiring: every registered strategy honours the protocol, the
order is load-bearing, the em_aq skip-when-compensated dependency holds, and a new
strategy plugs in through the same generic loop with no pipeline edit.
"""

from __future__ import annotations

import types
import unittest

from orka.pipeline.strategies import (
    POST_ASSIGNMENT_STRATEGIES,
    STRATEGY_REGISTRY,
    PostAssignmentStrategy,
)
from orka.pipeline.strategies.refinement import EMAQStrategy, MSEScaleStrategy


def _ctx(**kw):
    """Minimal duck-typed PackCtx stand-in."""
    return types.SimpleNamespace(**kw)


class StrategyContractTest(unittest.TestCase):
    def test_all_registered_honour_protocol(self):
        for s in POST_ASSIGNMENT_STRATEGIES:
            self.assertIsInstance(s, PostAssignmentStrategy)
            self.assertTrue(s.name)
            self.assertTrue(callable(s.applies) and callable(s.apply))

    def test_order_is_load_bearing(self):
        # error_compensation must precede em_aq (sets the skip flag) which precedes mse_scale.
        self.assertEqual(
            [s.name for s in POST_ASSIGNMENT_STRATEGIES],
            ["error_compensation", "em_aq", "mse_scale"],
        )

    def test_emaq_skipped_when_compensated(self):
        emaq = EMAQStrategy()
        ctx = _ctx(n_stages=2, em_aq_passes=3)
        self.assertTrue(emaq.applies(ctx, {}))                     # normal: runs
        self.assertFalse(emaq.applies(ctx, {"_compensated": True}))  # compensated: skipped

    def test_emaq_gates_on_stages_and_passes(self):
        emaq = EMAQStrategy()
        self.assertFalse(emaq.applies(_ctx(n_stages=1, em_aq_passes=3), {}))
        self.assertFalse(emaq.applies(_ctx(n_stages=2, em_aq_passes=0), {}))

    def test_mse_scale_gates_on_flag(self):
        mse = MSEScaleStrategy()
        self.assertTrue(mse.applies(_ctx(mse_scale=True), {}))
        self.assertFalse(mse.applies(_ctx(mse_scale=False), {}))

    def test_registry_documents_executable_strategies(self):
        names = {e["name"] for e in STRATEGY_REGISTRY}
        for s in POST_ASSIGNMENT_STRATEGIES:
            self.assertIn(s.name, names, f"{s.name} missing from STRATEGY_REGISTRY")

    def test_new_strategy_plugs_into_generic_loop(self):
        # A brand-new strategy runs through the same apply loop with no pipeline change.
        calls = []

        class Tag(PostAssignmentStrategy):
            name = "tag"

            def applies(self, ctx, c):
                return c.get("tag_me", False)

            def apply(self, ctx, c):
                calls.append(c["name"])

        loop = [Tag()]
        for c in ({"name": "a", "tag_me": True}, {"name": "b", "tag_me": False}):
            for strat in loop:
                if strat.applies(None, c):
                    strat.apply(None, c)
        self.assertEqual(calls, ["a"])


if __name__ == "__main__":
    unittest.main()
