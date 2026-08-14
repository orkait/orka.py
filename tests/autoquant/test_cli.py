import argparse
import json

import numpy as np
from safetensors.numpy import save_file

from orka.cli.commands import cmd_autoquant


def test_cmd_autoquant_writes_allocation_map(tmp_path):
    model = tmp_path / "m"
    model.mkdir()
    save_file({"embed_out.weight": np.random.randn(128, 32).astype("float32"),
               "model.layers.0.self_attn.q_proj.weight": np.random.randn(32, 32).astype("float32")},
              str(model / "model.safetensors"))
    out = tmp_path / "alloc.json"
    args = argparse.Namespace(model=str(model), objective="min-bits", out=str(out),
                              no_llm=True, target=None, prompts=None, autonomous=False)
    assert cmd_autoquant(args) == 0
    m = json.loads(out.read_text())
    assert m["embed_out.weight"]["method"] == "int8"


def test_cmd_autoquant_writes_artifact_report_and_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("ORKA_GRATIS_BASE_URL", "http://gratis")
    model = tmp_path / "m"
    model.mkdir()
    save_file({"embed_out.weight": np.random.randn(64, 16).astype("float32"),
               "model.layers.0.self_attn.q_proj.weight": np.random.randn(16, 16).astype("float32")},
              str(model / "model.safetensors"))
    names = iter(["inspect", "derive", "allocate", "pack", "pulse_check", "verify"])

    def next_action(self, messages, tools):
        return {"function": {"name": next(names), "arguments": "{}"}}

    monkeypatch.setattr("orka.autoquant.gratis.GratisClient.next_action", next_action)

    def pack_fn(*, source, out_dir, **_k):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text('{"format":"orka","tensors":[]}\n')

    monkeypatch.setattr("orka.pipeline.pack.pack_checkpoint", pack_fn)
    monkeypatch.setattr("orka.artifact.verify.verify_artifact",
                        lambda _p: {"ok": True, "passed": True})

    out = tmp_path / "out.orka"
    args = argparse.Namespace(
        model=str(model), objective="knee", out=str(out), no_llm=False,
        target=0.02, prompts=None, autonomous=True, gratis_base_url="http://gratis",
        gratis_model="gratis-auto", max_usd=0.05, max_steps=8, timeout_seconds=600,
    )
    assert cmd_autoquant(args) == 0
    assert (tmp_path / "out.orka").exists()
    assert (tmp_path / "autoquant-report.json").exists()
