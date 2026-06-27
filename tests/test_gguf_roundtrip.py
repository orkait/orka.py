"""Orka -> GGUF round-trip regression test.

Locks the fix for the GGUF tool decode bugs:
  - indices must be read honoring the v2 bit-pack + zlib encoding (a plain
    np.fromfile produced garbage on 12-bit / zlib stages - the default config);
  - 16-bit index values >= 2^15 must round-trip (signed vs unsigned read);
  - fp16 codebooks / fp16 scales must be read with the canonical readers.

Skips automatically when the vendored gguf-py (llama.cpp/gguf-py) is absent, so
it is CI-safe where that checkout does not exist.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))

# Skip the whole module if the vendored gguf writer/reader is unavailable.
gguf = pytest.importorskip("gguf")
sys.path.insert(0, str(ROOT / "tools"))

from safetensors.numpy import save_file  # noqa: E402

from orka.core._format import _read_indices  # noqa: E402
from orka.pipeline.decode import _decode_tensor  # noqa: E402
from orka.pipeline.pack import pack_checkpoint  # noqa: E402


def _build_synth_artifact(tmp_path: Path) -> Path:
    """Pack a tiny model whose stages exercise the tricky v2 paths:
    weights are sized so stage 0 reaches k=4096 -> 12-bit, bit-packed indices."""
    rng = np.random.default_rng(0)
    weights = {
        "model.layers.0.self_attn.q_proj.weight": rng.standard_normal((256, 256)).astype("float32"),
        "model.layers.0.mlp.gate_proj.weight": rng.standard_normal((512, 128)).astype("float32"),
        "model.norm.weight": rng.standard_normal((256,)).astype("float32"),  # passthrough
    }
    src = tmp_path / "synth.safetensors"
    save_file(weights, str(src))
    orka_dir = tmp_path / "synth.orka"
    pack_checkpoint(
        source=src,
        out_dir=orka_dir,
        group_size=8,
        codebook_size=4096,
        codebook_sizes=[4096, 256],  # rvq-12-8: stage0 12-bit (packed), stage1 8-bit
        codebook_mode="per-tensor",
        normalization="slrq-block",  # adds fp16 scales + salient sidecars
        backend="numpy",
        sample_vectors=4096,
        iterations=4,
        outlier_frac=0.02,  # adds the outlier sidecar - guards against the GGUF dropping it
    )
    return orka_dir


def test_gguf_roundtrip(tmp_path):
    orka_dir = _build_synth_artifact(tmp_path)
    manifest = json.loads((orka_dir / "manifest.json").read_text())
    tensors = manifest["tensors"]

    # The test is only meaningful if the artifact actually uses bit-packed stages.
    assert any(s.get("packed") for t in tensors for s in t["stages"]), "expected bit-packed stages"

    gguf_path = tmp_path / "synth.gguf"
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "orka_to_gguf.py"), str(orka_dir), "-o", str(gguf_path)],
        check=True,
        capture_output=True,
    )

    from gguf import GGUFReader
    import verify_gguf

    reader = GGUFReader(str(gguf_path))
    gguf_tensors = {t.name: t for t in reader.tensors}

    for tm in tensors:
        # (a) indices must round-trip bit-exact: this is the core of the bug,
        #     isolated from the lossy Q8_0 codebook storage.
        for stage in tm["stages"]:
            s_group = int(stage.get("group_size", tm["group_size"]))
            count = math.ceil(int(tm["padded_values"]) / s_group)
            ref_idx = _read_indices(
                orka_dir / stage["indices"],
                int(stage["index_bits"]),
                count,
                packed=bool(stage.get("packed", False)),
                encoding=stage.get("encoding", "raw"),
            )
            gt = gguf_tensors[f"{tm['name']}.orka.s{stage['stage']}.indices"]
            udt = np.uint16 if int(stage["index_bits"]) > 8 else np.uint8
            got_idx = gt.data.view(udt).astype(np.int64)
            assert np.array_equal(np.asarray(ref_idx).reshape(-1), got_idx), (
                f"index mismatch in {tm['name']} stage {stage['stage']}"
            )

        # (b) full decode must match the reference within Q8_0 codebook noise.
        ref = np.asarray(_decode_tensor(orka_dir, tm), dtype=np.float32).reshape(tm["shape"])
        got = verify_gguf.decompress_gguf_tensor(tm, gguf_tensors, reader)
        denom = float(np.max(np.abs(ref))) or 1.0
        rel = float(np.max(np.abs(ref - got))) / denom
        assert rel < 0.05, f"{tm['name']} decode rel diff {rel:.4f} exceeds Q8_0 noise floor"
