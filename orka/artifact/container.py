"""Single-file transport for .orka artifacts.

The container is a valid safetensors file: u64 header length, JSON header mapping each
artifact-relative path to a U8 tensor with explicit ``data_offsets``, then one contiguous
blob. Both directions stream in chunks; ``save_file`` would need the whole artifact resident.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

CONTAINER_VERSION = "1"
_HEADER_ALIGN = 8
_CHUNK = 8 << 20


def _header_of(path: Path) -> tuple[dict, dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {})
    return header, meta, 8 + n


def pack_container(artifact_dir: Path, out_file: Path, *, chunk_bytes: int = _CHUNK) -> dict:
    """Fold an artifact directory into one file. Returns a summary of what was written."""
    artifact_dir, out_file = Path(artifact_dir), Path(out_file)
    if not artifact_dir.is_dir():
        raise NotADirectoryError(f"not an artifact directory: {artifact_dir}")

    files = sorted(p for p in artifact_dir.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"no files under {artifact_dir}")

    entries: dict[str, dict] = {}
    empty: list[str] = []
    offset = 0
    for p in files:
        rel = p.relative_to(artifact_dir).as_posix()
        size = p.stat().st_size
        if size == 0:
            empty.append(rel)
            continue
        entries[rel] = {"dtype": "U8", "shape": [size], "data_offsets": [offset, offset + size]}
        offset += size

    header = dict(entries)
    header["__metadata__"] = {
        "orka_container": CONTAINER_VERSION,
        "artifact_name": artifact_dir.name,
        "file_count": str(len(entries) + len(empty)),
        "empty_files": json.dumps(empty),
    }
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((-len(blob)) % _HEADER_ALIGN)

    digest = hashlib.sha256()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(out_file.suffix + ".partial")
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for rel in entries:
            with open(artifact_dir / rel, "rb") as src:
                while chunk := src.read(chunk_bytes):
                    out.write(chunk)
                    digest.update(chunk)
    tmp.rename(out_file)

    return {
        "container": str(out_file),
        "files": len(entries) + len(empty),
        "payload_bytes": offset,
        "container_bytes": out_file.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def unpack_container(container: Path, out_dir: Path, *, chunk_bytes: int = _CHUNK) -> dict:
    """Expand a container back into an artifact directory."""
    container, out_dir = Path(container), Path(out_dir)
    header, meta, base = _header_of(container)
    if meta.get("orka_container") is None:
        raise ValueError(f"{container} is not an orka container")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(container, "rb") as f:
        for rel, spec in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            begin, end = spec["data_offsets"]
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            f.seek(base + begin)
            remaining = end - begin
            with open(dst, "wb") as o:
                while remaining:
                    chunk = f.read(min(chunk_bytes, remaining))
                    if not chunk:
                        raise EOFError(f"{container} truncated inside {rel}")
                    o.write(chunk)
                    remaining -= len(chunk)
            written += 1

    for rel in json.loads(meta.get("empty_files", "[]")):
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"")
        written += 1
    return {"out_dir": str(out_dir), "files": written}


def container_info(container: Path) -> dict:
    """Header-only inspection: no payload is read."""
    header, meta, base = _header_of(Path(container))
    payload = max((s["data_offsets"][1] for s in header.values()), default=0)
    actual = Path(container).stat().st_size - base
    return {
        "version": meta.get("orka_container"),
        "artifact_name": meta.get("artifact_name"),
        "files": int(meta.get("file_count", len(header))),
        "payload_bytes": payload,
        "complete": actual >= payload,
        "names": sorted(header),
    }
