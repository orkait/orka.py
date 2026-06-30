"""Fetch an HF model's config + tensor shapes WITHOUT downloading weights.

Shapes come from the safetensors header (first 8 bytes = uint64 header length, then that
many bytes of JSON mapping tensor -> {dtype, shape, data_offsets}) via HTTP range GET - KB,
not GB. Sharded models are enumerated through model.safetensors.index.json."""
from __future__ import annotations

import json
import struct

import httpx
import numpy as np

HF_BASE = "https://huggingface.co"

# safetensors dtype -> (numpy decode code, bytes/elem). BF16 has no native numpy dtype:
# it is the high 16 bits of fp32, so widen u16<<16 then view as f4.
_DT = {"F64": ("<f8", 8), "F32": ("<f4", 4), "F16": ("<f2", 2), "BF16": ("<u2", 2)}


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


def _header_with_len(url: str, token: str | None) -> tuple[int, dict]:
    """(header_len, full_header_json) for one safetensors file."""
    a = _auth(token)
    head = httpx.get(url, headers={**a, "Range": "bytes=0-7"}, follow_redirects=True, timeout=30)
    head.raise_for_status()
    n = struct.unpack("<Q", head.content[:8])[0]
    body = httpx.get(url, headers={**a, "Range": f"bytes=8-{8 + n - 1}"},
                     follow_redirects=True, timeout=30)
    body.raise_for_status()
    return n, json.loads(body.content)


def _resolve_file(model: str, name: str, token: str | None) -> tuple[str, int, dict]:
    """(url, header_len, header_entry) for the safetensors file holding `name`."""
    base = f"{HF_BASE}/{model}/resolve/main"
    try:
        hl, hdr = _header_with_len(f"{base}/model.safetensors", token)
        if name in hdr:
            return f"{base}/model.safetensors", hl, hdr[name]
    except httpx.HTTPStatusError:
        pass
    idx = httpx.get(f"{base}/model.safetensors.index.json",
                    headers=_auth(token), follow_redirects=True, timeout=30)
    idx.raise_for_status()
    shard = idx.json()["weight_map"][name]
    hl, hdr = _header_with_len(f"{base}/{shard}", token)
    return f"{base}/{shard}", hl, hdr[name]


def fetch_tensor_block(model: str, name: str, max_elems: int = 16384,
                       token: str | None = None) -> tuple[np.ndarray, tuple, str]:
    """Range-fetch the FIRST `max_elems` weights of one tensor and decode to float32.

    No full download: read the header for the tensor's byte range + dtype, then GET only
    the leading slice. Returns (flat float32 array, full shape, dtype string)."""
    url, hl, entry = _resolve_file(model, name, token)
    dtype = entry["dtype"]
    start, end = entry["data_offsets"]
    code, bpe = _DT.get(dtype, ("<f2", 2))
    nbytes = min(end - start, max_elems * bpe)
    nbytes -= nbytes % bpe
    abs0 = 8 + hl + start
    r = httpx.get(url, headers={**_auth(token), "Range": f"bytes={abs0}-{abs0 + nbytes - 1}"},
                  follow_redirects=True, timeout=60)
    r.raise_for_status()
    raw = r.content
    if dtype == "BF16":
        u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        arr = u.view(np.float32)
    else:
        arr = np.frombuffer(raw, dtype=code).astype(np.float32)
    return arr, tuple(entry["shape"]), dtype
