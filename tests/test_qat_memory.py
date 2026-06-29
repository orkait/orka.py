"""VRAM-fitting QAT modes: bf16 shadow + full-forward checkpoint + QLoRA recovery.

These let full-model QAT / quality recovery fit a small GPU. The invariants:
- the full-forward gradient checkpoint is bit-identical to the non-checkpointed
  path (it only trades compute for memory),
- a bf16 shadow trains (finite, nonzero grads close to fp32),
- a QLoRA adapter starts exactly at the quantized base (B init 0) and merges
  back to W_q + B@A*scaling.
"""
import unittest

import torch

from orka.qat._core import QATVQLinear


def _grads(ckpt, dtype):
    torch.manual_seed(0)
    w = torch.randn(128, 64)
    q = QATVQLinear(w, None, 8, [256], checkpoint=ckpt, shadow_dtype=dtype)
    q.train()
    torch.manual_seed(1)
    x = torch.randn(8, 64)
    out = q(x)
    (out.sum() + q._last_cb_loss).backward()
    return q.shadow.grad.float().clone(), q.codebooks[0].grad.float().clone()


class QATMemoryModeTest(unittest.TestCase):
    def test_full_forward_checkpoint_is_bit_identical(self):
        gs, gc = _grads(False, torch.float32)
        cs, cc = _grads(True, torch.float32)
        self.assertEqual(float((gs - cs).abs().max()), 0.0)
        self.assertEqual(float((gc - cc).abs().max()), 0.0)
        self.assertGreater(float(gs.abs().sum()), 0.0)

    def test_bf16_shadow_trains(self):
        bs, _ = _grads(True, torch.bfloat16)
        gs, _ = _grads(True, torch.float32)
        self.assertTrue(torch.isfinite(bs).all())
        self.assertGreater(float(bs.abs().sum()), 0.0)
        # bf16 master should track fp32 within a small relative tolerance
        self.assertLess(float((bs - gs).abs().max() / gs.abs().max()), 0.02)

    def test_shadow_dtype_default_is_fp32(self):
        q = QATVQLinear(torch.randn(64, 64), None, 8, [256])
        self.assertEqual(q.shadow.dtype, torch.float32)
        self.assertEqual(q.scales.dtype, torch.float32)


class QLoRARecoveryTest(unittest.TestCase):
    def test_adapter_starts_at_base_and_merges(self):
        from orka.qat.qlora import QLoRALinear

        torch.manual_seed(0)
        w_q = torch.randn(64, 32)
        ql = QLoRALinear(w_q, None, rank=8, alpha=16)
        x = torch.randn(4, 32)
        # B is zero-init -> initial output == frozen base, merged weight == W_q
        self.assertTrue(torch.allclose(ql(x), torch.nn.functional.linear(x, w_q), atol=1e-5))
        self.assertTrue(torch.allclose(ql.merged_weight(), w_q.float(), atol=1e-5))
        # after moving B, forward must equal linear(x, merged_weight)
        with torch.no_grad():
            ql.lora_B.add_(torch.randn_like(ql.lora_B))
        self.assertTrue(torch.allclose(
            ql(x), torch.nn.functional.linear(x, ql.merged_weight().to(x.dtype)), atol=1e-4))


if __name__ == "__main__":
    unittest.main()
