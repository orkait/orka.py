import httpx
from fastapi.testclient import TestClient

from ui.backend import app as appmod
from ui.backend.app import app

CONFIG = {"vocab_size": 32, "tie_word_embeddings": False, "torch_dtype": "bfloat16"}
SHAPES = {"lm_head.weight": (32, 8), "model.layers.0.mlp.down_proj.weight": (64, 64)}


def test_analyze_ok(monkeypatch):
    monkeypatch.setattr(appmod.journey, "fetch_config", lambda m, token=None: CONFIG)
    monkeypatch.setattr(appmod.journey, "fetch_shapes", lambda m, token=None: SHAPES)
    c = TestClient(app)
    r = c.get("/analyze", params={"model": "x/y", "bpw": 3.0})
    assert r.status_code == 200
    body = r.json()
    assert body["model"]["name"] == "x/y"
    assert body["result"]["source"] == "estimated"


def test_analyze_404(monkeypatch):
    def boom(m, token=None):
        raise httpx.HTTPStatusError("nf", request=httpx.Request("GET", "http://x"),
                                    response=httpx.Response(404))
    monkeypatch.setattr(appmod.journey, "fetch_config", boom)
    c = TestClient(app)
    r = c.get("/analyze", params={"model": "no/such"})
    assert r.status_code == 404
