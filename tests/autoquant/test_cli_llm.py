import argparse
import json

import numpy as np
from safetensors.numpy import save_file

from orka.cli import commands


def test_cmd_autoquant_uses_llm_for_escalated_tensor(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))   # isolate global cache

    # a tensor with an ambiguous/unknown role -> escalates
    model = tmp_path / "m"
    model.mkdir()
    save_file({"weird.mystery.proj.weight": np.random.randn(64, 64).astype("float32")},
              str(model / "model.safetensors"))

    def fake_make_llm_fn(*a, **k):
        def llm_fn(messages):
            return json.dumps({"method": "int8", "bits": 8, "stages": 0,
                               "normalization": "block-max", "keep_fp16": False,
                               "rationale": "llm chose int8"})
        return llm_fn

    monkeypatch.setattr("orka.autoquant.transport.make_llm_fn", fake_make_llm_fn)

    out = tmp_path / "alloc.json"
    args = argparse.Namespace(model=str(model), objective="min-bits", out=str(out),
                              no_llm=False, target=None, prompts=None)
    assert commands.cmd_autoquant(args) == 0
    m = json.loads(out.read_text())
    cfg = m["weird.mystery.proj.weight"]
    assert cfg["source"] == "llm" and cfg["method"] == "int8"
