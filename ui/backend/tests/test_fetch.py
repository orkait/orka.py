import json
import struct

import httpx

from ui.backend import fetch


def _st_bytes(header: dict) -> bytes:
    blob = json.dumps(header).encode()
    return struct.pack("<Q", len(blob)) + blob


def test_parse_header_from_bytes(monkeypatch):
    header = {"__metadata__": {"x": "y"},
              "lm_head.weight": {"dtype": "BF16", "shape": [32, 8], "data_offsets": [0, 512]},
              "model.layers.0.mlp.down_proj.weight": {"dtype": "BF16", "shape": [8, 16], "data_offsets": [512, 768]}}
    raw = _st_bytes(header)

    def fake_get(url, headers=None, **kw):
        rng = headers.get("Range", "")
        start, end = rng.replace("bytes=", "").split("-")
        body = raw[int(start): int(end) + 1]
        return httpx.Response(200, content=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(fetch.httpx, "get", fake_get)
    shapes = fetch._st_header("http://x/model.safetensors", token=None)
    assert shapes == {"lm_head.weight": (32, 8),
                      "model.layers.0.mlp.down_proj.weight": (8, 16)}
