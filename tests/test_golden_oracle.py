"""Structural gate for pack_checkpoint.

Codebook bytes are not reproducible under threaded BLAS, so this hashes a
config-derived fingerprint of each manifest instead. A change that moves
COMBINED_HASH changed pack behaviour.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from orka.pipeline.pack import pack_checkpoint

COMBINED_HASH = "d73e0b19fc38f099"

PER_CONFIG_HASHES = {
    "default": "5ad81ae4d56c38c4",
    "per-tensor": "5ad81ae4d56c38c4",
    "global": "e6370ee0fedd3cbc",
    "family": "6c1b65b21e0ed8f5",
    "blockmax": "0b77032fe42654ae",
    "slrq": "2a849fadae28809a",
    "chan-blockmax": "cfeade5b47afbe16",
    "multistage": "bcab98f95b147b12",
    "emaq": "bcab98f95b147b12",
    "mse-scale": "2e5c86dbe3daf9e8",
    "outliers": "5ad81ae4d56c38c4",
    "hadamard": "045b4f3c250701a8",
}

CONFIGS = {
    "default": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy"),
    "per-tensor": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy", codebook_mode="per-tensor"),
    "global": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy", codebook_mode="global"),
    "family": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy", codebook_mode="family"),
    "blockmax": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy", normalization="block-max"),
    "slrq": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy", normalization="slrq-block"),
    "chan-blockmax": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy", normalization="channel-block-max"),
    "multistage": dict(group_size=8, codebook_sizes=[16, 16], iterations=3, backend="numpy"),
    "emaq": dict(group_size=8, codebook_sizes=[16, 16], iterations=3, backend="numpy", em_aq_passes=2),
    "mse-scale": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy", normalization="block-max", mse_scale=True),
    "outliers": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy", outlier_frac=0.05),
    "hadamard": dict(group_size=8, codebook_size=16, iterations=3, backend="numpy", rotation="hadamard", rotation_seed=7),
}


def _make_source(path: Path) -> None:
    random.seed(0)

    def rows(r: int, c: int):
        return [[random.gauss(0, 1) for _ in range(c)] for _ in range(r)]

    path.write_text(json.dumps({"tensors": {
        "model.embed_tokens.weight": rows(64, 32),
        "model.layers.0.self_attn.q_proj.weight": rows(32, 32),
        "model.layers.0.self_attn.k_proj.weight": rows(16, 32),
        "model.layers.0.mlp.up_proj.weight": rows(48, 32),
        "model.layers.0.mlp.down_proj.weight": rows(32, 48),
    }}))


def _fingerprint(artifact: Path) -> str:
    man = json.loads((artifact / "manifest.json").read_text())
    struct = {
        "n_stages": man.get("n_stages"),
        "codebook_mode": man.get("codebook_mode"),
        "normalization": man.get("normalization"),
        "tensor_count": man.get("tensor_count"),
        "mse_scale": man.get("mse_scale"),
        "rotation": man.get("rotation"),
        "tensors": sorted(
            [
                {
                    "name": t["name"], "group_size": t.get("group_size"), "shape": t.get("shape"),
                    "n_stages": t.get("n_stages"), "normalization": t.get("normalization"),
                    "index_bits": t.get("index_bits"), "has_outliers": bool(t.get("outlier_count")),
                    "has_salient": bool(t.get("salient_count")), "scale_count": t.get("scale_count"),
                }
                for t in man.get("tensors", [])
            ],
            key=lambda x: x["name"],
        ),
    }
    return hashlib.sha256(json.dumps(struct, sort_keys=True).encode()).hexdigest()[:16]


def _pack(name: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "m.json"
        _make_source(src)
        old = sys.stderr
        sys.stderr = io.StringIO()
        try:
            random.seed(12345)
            np.random.seed(12345)
            pack_checkpoint(src, root / "a.orka", **CONFIGS[name])
            return _fingerprint(root / "a.orka")
        finally:
            sys.stderr = old


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_config_fingerprint_unchanged(name: str) -> None:
    assert _pack(name) == PER_CONFIG_HASHES[name], (
        f"pack behaviour changed for config {name!r}. If deliberate, update "
        f"PER_CONFIG_HASHES and COMBINED_HASH and say why in the commit message."
    )


def test_combined_hash_unchanged() -> None:
    results = {name: _pack(name) for name in CONFIGS}
    combined = hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()[:16]
    assert combined == COMBINED_HASH
