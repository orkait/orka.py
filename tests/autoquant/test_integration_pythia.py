import os
import json
import subprocess
import glob
import pytest

BASE = glob.glob(os.path.expanduser(
    "~/ai-models/hf-cache/hub/models--EleutherAI--pythia-160m/snapshots/*"))


@pytest.mark.skipif(not BASE, reason="pythia-160m not cached")
def test_autoquant_derives_int8_head(tmp_path):
    out = tmp_path / "alloc.json"
    r = subprocess.run([".venv/bin/python", "orka.py", "autoquant", BASE[0],
                        "--objective", "min-bits", "--no-llm", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = json.loads(out.read_text())
    head = next(v for k, v in m.items() if "embed_out" in k or "lm_head" in k)
    assert head["method"] == "int8"          # the session's hard-won prior, auto-derived
    assert any(v["method"] == "rvq" for k, v in m.items() if "mlp" in k or "attn" in k)
