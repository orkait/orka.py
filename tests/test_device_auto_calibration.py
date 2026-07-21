"""--device auto must reach torch as a concrete device.

cmd_pack resolves the BACKEND but passes --device through verbatim, and it defaults
to "auto". Activation calibration then called model.to("auto"), which raises; the
graceful-degradation handler in _load_awq_activations turned that into a warning and
packed UNWEIGHTED. Measured cost of losing Hessian weighting, per the code's own
note: SmolLM2-135M rvq-12-12 perplexity ratio 1.63x -> 2.19x. Silent, on the default
path, against any real model directory.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import torch

from orka.quant.activations import _collect_activations_hf


class _StubModel:
    def __init__(self):
        self.devices: list[str] = []

    def to(self, device):
        # The real nn.Module.to raises on an unparsable string; mirror that so the
        # test fails the same way production did rather than silently accepting it.
        torch.device(device)
        self.devices.append(str(device))
        return self

    def eval(self):
        return self

    def named_modules(self):
        return []

    def __call__(self, **_kwargs):
        return types.SimpleNamespace(logits=torch.zeros(1, 2, 4))


class _StubTokenizer:
    def __call__(self, _text, **_kwargs):
        return {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }


class DeviceAutoCalibrationTest(unittest.TestCase):
    def _run(self, device: str) -> _StubModel:
        stub = _StubModel()
        transformers = types.ModuleType("transformers")
        transformers.AutoModelForCausalLM = mock.Mock(
            from_pretrained=mock.Mock(return_value=stub)
        )
        transformers.AutoTokenizer = mock.Mock(
            from_pretrained=mock.Mock(return_value=_StubTokenizer())
        )
        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            _collect_activations_hf(
                model_dir="unused",
                prompts=["hello world"],
                max_length=8,
                device=device,
                max_samples_per_layer=4,
            )
        return stub

    def test_auto_is_resolved_before_reaching_torch(self) -> None:
        stub = self._run("auto")
        self.assertEqual(len(stub.devices), 1)
        self.assertNotEqual(stub.devices[0], "auto")
        self.assertIn(stub.devices[0].split(":")[0], {"cpu", "cuda"})

    def test_explicit_cpu_passes_through(self) -> None:
        stub = self._run("cpu")
        self.assertEqual(stub.devices, ["cpu"])


if __name__ == "__main__":
    unittest.main()
