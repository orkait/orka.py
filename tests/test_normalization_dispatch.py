"""Behavior lock for the normalization dispatcher.

`_apply_normalization` routes a normalization mode string to one of the per-method
kernels and assembles a fixed 6-tuple:

    (tensor, row_scales, source_flat, awq_col_scales, salient_weights, salient_indices)

The per-method numerics (block-max / slrq / channel-block-max round-trips) are already
locked in test_math_properties. This file locks the DISPATCHER contract: which mode
populates which slots, the None-pattern, shape preservation, and that the routed method
reconstructs the original. It is the regression gate for making normalization pluggable -
the registry refactor must keep every assertion here green.
"""

from __future__ import annotations

import unittest

import numpy as np

from orka.transforms.normalize import (
    NORMALIZATION_REGISTRY,
    NormalizationResult,
    _apply_block_max_scales_numpy,
    _apply_normalization,
    normalization_modes,
    register_normalization,
)

BLOCK = 16


def _fixed_input(rows: int = 8, cols: int = 32):
    rng = np.random.RandomState(0)
    return rng.standard_normal((rows, cols)).astype(np.float32)


def _norm(w, mode, *, awq_activations=None, slrq_salient=True):
    return _apply_normalization(
        w,
        "t.weight",
        mode,
        awq_activations,
        0.5,            # awq_alpha
        BLOCK,          # block_scale_size
        "numpy",        # backend
        "cpu",          # device (unused on numpy)
        None,           # awq_fallbacks
        slrq_salient=slrq_salient,
    )


class NormalizationDispatchTest(unittest.TestCase):
    def test_returns_six_tuple_for_every_mode(self):
        for mode in ("none", "block-max", "channel-block-max", "slrq-block", "awq"):
            out = _norm(_fixed_input(), mode)
            self.assertEqual(len(out), 6, f"{mode} did not return a 6-tuple")

    def test_none_passes_through_with_empty_slots(self):
        w = _fixed_input()
        tensor, row_scales, source_flat, awq_cols, sal_w, sal_i = _norm(w, "none")
        self.assertIsNone(row_scales)
        self.assertIsNone(awq_cols)
        self.assertIsNone(sal_w)
        self.assertIsNone(sal_i)
        np.testing.assert_allclose(np.asarray(tensor).reshape(-1), w.reshape(-1), rtol=0, atol=0)
        np.testing.assert_allclose(np.asarray(source_flat), w.reshape(-1), rtol=0, atol=0)

    def test_block_max_routes_and_reconstructs(self):
        w = _fixed_input()
        tensor, row_scales, source_flat, awq_cols, sal_w, sal_i = _norm(w, "block-max")
        self.assertIsNotNone(row_scales)
        self.assertIsNone(awq_cols)
        self.assertIsNone(sal_w)
        self.assertIsNone(sal_i)
        # normalized magnitude bounded (fp16 scale rounding can nudge slightly past 1)
        self.assertLessEqual(float(np.abs(np.asarray(tensor)).max()), 1.001)
        back = _apply_block_max_scales_numpy(np.asarray(tensor).reshape(-1), row_scales, BLOCK)
        np.testing.assert_allclose(np.asarray(back), w.reshape(-1), rtol=2e-2, atol=2e-2)

    def test_channel_block_max_routes_and_reconstructs(self):
        w = _fixed_input()
        tensor, row_scales, source_flat, awq_cols, sal_w, sal_i = _norm(w, "channel-block-max")
        self.assertIsNotNone(row_scales)
        self.assertIsNone(sal_w)
        self.assertIsNone(sal_i)
        back = _apply_block_max_scales_numpy(np.asarray(tensor).reshape(-1), row_scales, BLOCK)
        np.testing.assert_allclose(np.asarray(back), w.reshape(-1), rtol=2e-2, atol=2e-2)

    def test_slrq_block_emits_salient(self):
        w = _fixed_input()
        tensor, row_scales, source_flat, awq_cols, sal_w, sal_i = _norm(w, "slrq-block")
        self.assertIsNotNone(row_scales)
        self.assertIsNotNone(sal_w)
        self.assertIsNotNone(sal_i)

    def test_slrq_block_without_salient_drops_salient(self):
        w = _fixed_input()
        _, row_scales, _, _, sal_w, sal_i = _norm(w, "slrq-block", slrq_salient=False)
        self.assertIsNotNone(row_scales)
        # salient escape disabled -> no salient sidecar payload
        empty = (sal_w is None) or (np.asarray(sal_w).size == 0)
        self.assertTrue(empty, "salient weights should be empty when slrq_salient=False")

    def test_awq_without_activations_falls_back_to_passthrough(self):
        w = _fixed_input()
        tensor, row_scales, source_flat, awq_cols, sal_w, sal_i = _norm(w, "awq", awq_activations=None)
        self.assertIsNone(row_scales)
        self.assertIsNone(awq_cols)
        np.testing.assert_allclose(np.asarray(tensor).reshape(-1), w.reshape(-1), rtol=0, atol=0)


class NormalizationRegistryTest(unittest.TestCase):
    def test_modes_match_registry(self):
        self.assertEqual(
            set(normalization_modes()),
            {"slrq-block", "awq", "awq-block-max", "channel-block-max", "block-max"},
        )

    def test_register_new_mode_dispatches_without_editing_chain(self):
        sentinel = object()

        def _passthrough_marker(tensor, **_):
            arr = np.asarray(tensor, dtype=np.float32)
            return NormalizationResult(tensor=arr, source_flat=arr.reshape(-1), awq_col_scales=sentinel)

        register_normalization("unit-test-mode", _passthrough_marker)
        try:
            _, _, _, awq_cols, _, _ = _norm(_fixed_input(), "unit-test-mode")
            self.assertIs(awq_cols, sentinel)  # routed to the freshly-registered handler
        finally:
            NORMALIZATION_REGISTRY.pop("unit-test-mode", None)


if __name__ == "__main__":
    unittest.main()
