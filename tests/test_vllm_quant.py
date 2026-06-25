"""Tests for the vLLM quant scaffold's vllm-free core.

The vLLM glue (create_weights/apply) needs a vllm env to validate; these lock the parts
that don't: the module is import-safe without vllm, the shared VQLinear-from-meta builder
works, and registration fails cleanly when vllm is absent.
"""

from __future__ import annotations

import unittest


class VllmQuantScaffoldTest(unittest.TestCase):
    def test_module_import_safe_without_vllm(self):
        import orka.vllm_quant  # must not import vllm at module load
        self.assertTrue(hasattr(orka.vllm_quant, "register_orka_vllm"))

    def test_build_vq_linear_from_meta(self):
        from orka.vllm_quant import _build_vq_linear_from_meta

        meta = {
            "out_features": 64, "in_features": 64, "n_stages": 2,
            "group_size": 4, "block_size": 16, "cb_sizes": [256, 256], "has_bias": False,
        }
        vq = _build_vq_linear_from_meta(meta)
        self.assertEqual(vq.out_features, 64)
        self.assertEqual(vq.n_stages, 2)
        self.assertTrue(hasattr(vq, "indices_0"))  # cb 256 -> uint8 buffer

    def test_planed_meta_builds_planes(self):
        from orka.vllm_quant import _build_vq_linear_from_meta

        meta = {
            "out_features": 64, "in_features": 64, "n_stages": 2,
            "group_size": 4, "block_size": 16, "cb_sizes": [1024, 1024], "has_bias": False,
        }
        vq = _build_vq_linear_from_meta(meta)
        self.assertTrue(hasattr(vq, "indices_lo_0"))  # 1024 -> 10-bit -> planes

    def test_register_without_vllm_raises_cleanly(self):
        from orka.vllm_quant import register_orka_vllm

        try:
            import vllm  # noqa: F401
            self.skipTest("vllm installed; registration path covered elsewhere")
        except ImportError:
            with self.assertRaises(ImportError):
                register_orka_vllm()


if __name__ == "__main__":
    unittest.main()
