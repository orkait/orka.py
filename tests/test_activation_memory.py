"""Calibration peak RAM must scale with the sample cap, not with the prompt count.

The collector subsampled only after concatenating every captured activation, so peak memory
grew linearly in prompts: ~7 GB for 64 prompts on a 230M model, ~59 GB for 512 - enough to
drive a 30 GB machine into swap death. The cap existed; it was applied too late to protect
anything. These tests pin that peak stays bounded as prompts grow.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from orka.quant.activations import _collect_activations_hf  # noqa: E402

DIM, CAP = 64, 32


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(DIM, DIM)

    def forward(self, input_ids=None, attention_mask=None, **_):
        x = torch.randn(1, int(input_ids.shape[1]), DIM)
        return types.SimpleNamespace(logits=self.lin(x))


class _Tok:
    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        n = min(len(text.split()), max_length or 16)
        return {"input_ids": torch.ones(1, max(n, 2), dtype=torch.long)}


def _install_stub(monkeypatch, model):
    stub = types.ModuleType("transformers")
    stub.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: model)
    stub.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: _Tok())
    monkeypatch.setitem(sys.modules, "transformers", stub)


def _peak_rows(monkeypatch, n_prompts):
    """Largest row count any single buffer reaches during collection."""
    model = _Tiny()
    _install_stub(monkeypatch, model)
    seen = []
    real_cat = torch.cat

    def spy(tensors, *a, **k):
        out = real_cat(tensors, *a, **k)
        if out.dim() == 2 and out.shape[-1] == DIM:
            seen.append(out.shape[0])
        return out

    monkeypatch.setattr(torch, "cat", spy)
    acts = _collect_activations_hf(
        Path("unused"), ["word " * 20] * n_prompts, max_length=16,
        device="cpu", max_samples_per_layer=CAP)
    return acts, (max(seen) if seen else 0)


def test_peak_buffer_does_not_grow_with_prompt_count(monkeypatch):
    _, peak_small = _peak_rows(monkeypatch, 8)
    _, peak_large = _peak_rows(monkeypatch, 200)
    # 25x the prompts must not mean 25x the peak; trimming bounds it near 2 x cap
    assert peak_large <= 4 * CAP, f"peak {peak_large} rows for 200 prompts"
    assert peak_large <= max(peak_small, CAP) * 3


def test_output_still_respects_the_sample_cap(monkeypatch):
    acts, _ = _peak_rows(monkeypatch, 64)
    assert acts, "collector returned nothing"
    for name, t in acts.items():
        assert t.shape[0] <= CAP, f"{name} has {t.shape[0]} rows, cap is {CAP}"
        assert t.dtype is torch.float32


def test_few_prompts_are_not_truncated_below_what_was_captured(monkeypatch):
    acts, _ = _peak_rows(monkeypatch, 1)
    assert all(t.shape[0] > 0 for t in acts.values())
