"""Container round-trip must be byte-exact, and the result must be readable as safetensors."""
from __future__ import annotations

import json
import struct

import pytest

from orka.artifact.container import (
    container_info,
    pack_container,
    unpack_container,
)


def _artifact(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "tensors").mkdir()
    files = {
        "manifest.json": json.dumps({"tensors": [{"name": "w", "shape": [4, 2]}]}).encode(),
        "passthrough.safetensors": bytes(range(256)) * 3,
        "tensors/w.s0.indices": bytes([7, 9, 11, 13]),
        "tensors/w.s0.codebook.f32": b"\x00\x01\x02\x03" * 9,
        "tensors/w.block_max_scale.f32": b"\xff" * 17,
    }
    for rel, data in files.items():
        (root / rel).write_bytes(data)
    return files


def test_round_trip_is_byte_exact(tmp_path):
    src = tmp_path / "a.orka"
    files = _artifact(src)
    info = pack_container(src, tmp_path / "a.container")
    assert info["files"] == len(files)

    out = tmp_path / "restored"
    unpack_container(tmp_path / "a.container", out)
    restored = {p.relative_to(out).as_posix(): p.read_bytes()
                for p in out.rglob("*") if p.is_file()}
    assert restored == files


def test_container_is_valid_safetensors(tmp_path):
    """The point of reusing the layout: standard tooling reads it without orka."""
    safetensors = pytest.importorskip("safetensors")
    src = tmp_path / "a.orka"
    files = _artifact(src)
    path = tmp_path / "a.container"
    pack_container(src, path)

    with safetensors.safe_open(str(path), framework="np") as f:
        assert set(f.keys()) == set(files)
        assert bytes(f.get_tensor("tensors/w.s0.indices")) == files["tensors/w.s0.indices"]
        assert f.metadata()["orka_container"] == "1"


def test_empty_files_survive(tmp_path):
    src = tmp_path / "a.orka"
    _artifact(src)
    (src / "tensors" / "blank.bin").write_bytes(b"")
    pack_container(src, tmp_path / "a.container")
    out = tmp_path / "restored"
    unpack_container(tmp_path / "a.container", out)
    assert (out / "tensors" / "blank.bin").read_bytes() == b""


def test_truncated_container_is_detected(tmp_path):
    """A partial directory is indistinguishable from a whole one; a partial container is not."""
    src = tmp_path / "a.orka"
    _artifact(src)
    path = tmp_path / "a.container"
    pack_container(src, path)

    full = container_info(path)
    assert full["complete"] is True

    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 32])
    assert container_info(path)["complete"] is False
    with pytest.raises(EOFError):
        unpack_container(path, tmp_path / "partial")


def test_rejects_foreign_safetensors(tmp_path):
    header = json.dumps({"x": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode()
    path = tmp_path / "foreign.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00")
    with pytest.raises(ValueError, match="not an orka container"):
        unpack_container(path, tmp_path / "out")


def test_no_partial_file_left_on_success(tmp_path):
    src = tmp_path / "a.orka"
    _artifact(src)
    path = tmp_path / "a.container"
    pack_container(src, path)
    assert not list(tmp_path.glob("*.partial"))
