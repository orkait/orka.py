"""Fetch an HF model's config + tensor shapes WITHOUT downloading weights.

Shapes come from the safetensors header (first 8 bytes = uint64 header length, then that
many bytes of JSON mapping tensor -> {dtype, shape, data_offsets}) via HTTP range GET - KB,
not GB. Sharded models are enumerated through model.safetensors.index.json."""
from __future__ import annotations

import json
import struct

import httpx

HF_BASE = "https://huggingface.co"


def _auth(token: str | None) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def fetch_config(model: str, token: str | None = None) -> dict:
    url = f"{HF_BASE}/{model}/resolve/main/config.json"
    r = httpx.get(url, headers=_auth(token), follow_redirects=True, timeout=30)
    r.raise_for_status()
    return r.json()


def _st_header(url: str, token: str | None) -> dict:
    """tensor name -> shape tuple from one safetensors file's header (2 range GETs)."""
    a = _auth(token)
    head = httpx.get(url, headers={**a, "Range": "bytes=0-7"}, follow_redirects=True, timeout=30)
    head.raise_for_status()
    n = struct.unpack("<Q", head.content[:8])[0]
    body = httpx.get(url, headers={**a, "Range": f"bytes=8-{8 + n - 1}"},
                     follow_redirects=True, timeout=30)
    body.raise_for_status()
    header = json.loads(body.content)
    return {k: tuple(v["shape"]) for k, v in header.items()
            if k != "__metadata__" and isinstance(v, dict) and "shape" in v}


def fetch_shapes(model: str, token: str | None = None) -> dict:
    """All tensor shapes. Single-file model.safetensors, else the sharded index."""
    base = f"{HF_BASE}/{model}/resolve/main"
    try:
        return _st_header(f"{base}/model.safetensors", token)
    except httpx.HTTPStatusError:
        pass
    idx = httpx.get(f"{base}/model.safetensors.index.json",
                    headers=_auth(token), follow_redirects=True, timeout=30)
    idx.raise_for_status()
    shapes: dict = {}
    for shard in sorted(set(idx.json()["weight_map"].values())):
        shapes.update(_st_header(f"{base}/{shard}", token))
    return shapes
