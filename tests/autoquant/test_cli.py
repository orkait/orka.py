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
                              no_llm=True, target=None, prompts=None)
    assert cmd_autoquant(args) == 0
    m = json.loads(out.read_text())
    assert m["embed_out.weight"]["method"] == "int8"
