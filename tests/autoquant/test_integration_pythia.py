import glob
import json
import os
import subprocess
import sys

import pytest

BASE = glob.glob(os.path.expanduser(
    "~/ai-models/hf-cache/hub/models--EleutherAI--pythia-160m/snapshots/*"))


@pytest.mark.skipif(not BASE, reason="pythia-160m not cached")
def test_autoquant_derives_int8_head(tmp_path):
    out = tmp_path / "alloc.json"
    r = subprocess.run([sys.executable, "-m", "orka", "autoquant", BASE[0],
                        "--objective", "min-bits", "--no-llm", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = json.loads(out.read_text())
    head = next(v for k, v in m.items() if "embed_out" in k or "lm_head" in k)
    assert head["method"] == "int8"          # the session's hard-won prior, auto-derived
    assert any(v["method"] == "rvq" for k, v in m.items() if "mlp" in k or "attn" in k)


def test_autonomous_fails_without_gratis_budget(tmp_path):
    model = tmp_path / "m"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"")
    r = subprocess.run(
        [sys.executable, "-m", "orka", "autoquant", str(model),
         "--autonomous", "--objective", "knee", "--out", str(tmp_path / "out.orka")],
        capture_output=True, text=True, env={**os.environ, "ORKA_GRATIS_BASE_URL": ""},
    )
    assert r.returncode != 0
    assert "max-usd" in (r.stdout + r.stderr).lower() or "gratis" in (r.stdout + r.stderr).lower()
