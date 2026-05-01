"""On-disk caching of stage-0 codebooks. Strict-zero quality loss (binary identical)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

from orka._tensor import _is_numpy_array, _is_torch_tensor


def _codebook_cache_key(parts: Sequence[object]) -> str:
    import hashlib

    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()

def _codebook_cache_load(cache_dir: Path | None, key: str):
    if cache_dir is None:
        return None
    path = cache_dir / f"{key}.npy"
    if not path.exists():
        return None
    try:
        import numpy as np

        return np.load(str(path), allow_pickle=False)
    except Exception:
        return None


def _codebook_cache_save(cache_dir: Path | None, key: str, codebook) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npy"
    import numpy as np

    if _is_torch_tensor(codebook):
        cb_np = codebook.detach().cpu().to(dtype=__import__("torch").float32).numpy()
    elif _is_numpy_array(codebook):
        cb_np = np.asarray(codebook, dtype=np.float32)
    else:
        cb_np = np.asarray([list(row) for row in codebook], dtype=np.float32)
    tmp = path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, cb_np, allow_pickle=False)
    tmp.replace(path)

