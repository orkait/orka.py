#!/usr/bin/env python3
"""Orka compiler prototype.

The CLI is intentionally dependency-light. `calc` and `selftest` run on the
standard library. `inspect` and `pack` use optional checkpoint libraries only
when the selected input format needs them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import tempfile
import threading
import queue
import time
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Callable, Iterable, Sequence


ORKA_VERSION = 1


@dataclass(frozen=True)
class PayloadEstimate:
    params: int
    group_size: int
    codebook_size: int
    index_bits: int
    vector_count: int
    index_bytes: int
    scale_block_vectors: int
    scale_bytes: int
    bits_per_weight: float
    total_payload_bytes: int


class BackgroundWriter:
    def __init__(self):
        self.queue = queue.Queue(maxsize=128)
        self.errors: list[tuple[str, str]] = []
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            task = self.queue.get()
            if task is None:
                self.queue.task_done()
                break
            fn, args = task
            try:
                fn(*args)
            except Exception as e:
                self.errors.append((fn.__name__, repr(e)))
            finally:
                self.queue.task_done()

    def submit(self, fn, *args):
        self.queue.put((fn, args))

    def wait(self):
        self.queue.join()
        if self.errors:
            detail = "; ".join(f"{name}: {err}" for name, err in self.errors)
            raise RuntimeError(
                f"background writes failed ({len(self.errors)} error(s)): {detail}"
            )

    def stop(self):
        self.queue.put(None)
        if self.thread.is_alive():
            self.thread.join()

_BG_WRITER = BackgroundWriter()


def estimate_payload(
    params: int,
    group_size: int,
    codebook_size: int,
    scale_block_vectors: int = 64,
    scale_bits: int = 0,
) -> PayloadEstimate:
    if params <= 0:
        raise ValueError("params must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if codebook_size <= 1:
        raise ValueError("codebook_size must be greater than 1")
    if scale_block_vectors <= 0:
        raise ValueError("scale_block_vectors must be positive")
    if scale_bits < 0:
        raise ValueError("scale_bits must be non-negative")

    index_bits = math.ceil(math.log2(codebook_size))
    vector_count = math.ceil(params / group_size)
    index_bytes = math.ceil(vector_count * index_bits / 8)
    scale_count = math.ceil(vector_count / scale_block_vectors)
    scale_bytes = math.ceil(scale_count * scale_bits / 8)
    return PayloadEstimate(
        params=params,
        group_size=group_size,
        codebook_size=codebook_size,
        index_bits=index_bits,
        vector_count=vector_count,
        index_bytes=index_bytes,
        scale_block_vectors=scale_block_vectors,
        scale_bytes=scale_bytes,
        bits_per_weight=(index_bytes + scale_bytes) * 8 / params,
        total_payload_bytes=index_bytes + scale_bytes,
    )


def _codebook_cache_key(parts: Sequence[object]) -> str:
    import hashlib

    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _derive_seed(parts: Sequence[object]) -> int:
    import hashlib

    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


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


def _report_progress(path: Path | None, message: str):
    if path:
        try:
            with path.open("w") as f:
                f.write(message + "\n")
        except Exception:
            pass
    print(message)


def _source_signature(source: Path) -> str:
    try:
        st = Path(source).resolve().stat()
        return f"{st.st_size}-{st.st_mtime_ns}"
    except OSError:
        return str(source)


def _collect_activations_hf(
    model_dir: Path,
    prompts: Sequence[str],
    max_length: int,
    device: str,
    max_samples_per_layer: int = 4096,
) -> dict:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("activation calibration requires torch and transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True)
    model.to(device)
    model.eval()
    activations: dict[str, list] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            captured_name = name

            def hook(_mod, inputs, _outputs, _name=captured_name):
                x = inputs[0]
                if x.dim() > 2:
                    x = x.reshape(-1, x.shape[-1])
                activations.setdefault(_name, []).append(x.detach().cpu())

            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=max_length
            )
            ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            if attn is None:
                attn = torch.ones_like(ids)
            attn = attn.to(device)
            model(input_ids=ids, attention_mask=attn)
    for h in handles:
        h.remove()
    out: dict[str, "torch.Tensor"] = {}
    for name, xs in activations.items():
        full = xs[0] if len(xs) == 1 else __import__("torch").cat(xs, dim=0)
        if full.shape[0] > max_samples_per_layer:
            import torch as _t

            idx = _t.randperm(full.shape[0])[:max_samples_per_layer]
            full = full[idx]
        out[name + ".weight"] = full.to(dtype=__import__("torch").float32)
    return out


def _fwht_torch(x):
    import torch

    x = x.contiguous().clone()
    n = int(x.shape[-1])
    if n & (n - 1) != 0:
        raise ValueError(f"FWHT requires power-of-2 last dim, got {n}")
    leading_shape = list(x.shape[:-1])
    h = 1
    while h < n:
        view = x.view(*leading_shape, n // (2 * h), 2, h)
        a = view[..., 0, :].clone()
        b = view[..., 1, :].clone()
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        x = view.reshape(*leading_shape, n)
        h *= 2
    return x * (1.0 / math.sqrt(n))


def _fwht_numpy(x):
    import numpy as np

    x = np.array(x, dtype=np.float32, copy=True)
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError(f"FWHT requires power-of-2 last dim, got {n}")
    leading_shape = list(x.shape[:-1])
    h = 1
    while h < n:
        view = x.reshape(*leading_shape, n // (2 * h), 2, h)
        a = view[..., 0, :].copy()
        b = view[..., 1, :].copy()
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        x = view.reshape(*leading_shape, n)
        h *= 2
    return x / math.sqrt(n)


def _tensor_rotation_seed(global_seed: int, name: str) -> int:
    import hashlib

    h = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(h, "little") ^ int(global_seed)) & ((1 << 63) - 1)


def _generate_orthogonal_torch(n: int, seed: int, device, dtype):
    import torch

    q_np = _generate_orthogonal_numpy(n, seed)
    return torch.from_numpy(q_np).to(device=device, dtype=dtype)


def _generate_orthogonal_numpy(n: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)
    a = rng.standard_normal((n, n)).astype(np.float32)
    q, _ = np.linalg.qr(a)
    return q.astype(np.float32)


def _rotate_tensor_to_2d(
    tensor, name: str, rotation: str, rotation_seed: int, backend: str, device: str
):
    if rotation == "none":
        return tensor, None
    if rotation not in {"orthogonal", "hadamard"}:
        raise ValueError(f"unknown rotation mode: {rotation}")

    if rotation == "hadamard":
        if backend == "torch":
            _, t = _torch_f32(tensor, device)
            shape = list(t.shape)
            rows, cols = shape[0], 1
            for s in shape[1:]:
                cols *= int(s)
            if cols & (cols - 1) != 0:
                raise ValueError(
                    f"hadamard rotation requires power-of-2 last-dim product, tensor {name} has {cols}"
                )
            return _fwht_torch(t.reshape(rows, cols)).reshape(shape), 0

        arr = _numpy_float32_array(tensor)
        shape = [int(x) for x in arr.shape]
        rows, cols = shape[0], 1
        for s in shape[1:]:
            cols *= int(s)
        if cols & (cols - 1) != 0:
            raise ValueError(
                f"hadamard rotation requires power-of-2 last-dim product, tensor {name} has {cols}"
            )
        return _fwht_numpy(arr.reshape((rows, cols))).reshape(shape), 0

    seed = _tensor_rotation_seed(rotation_seed, name)
    if backend == "torch":
        import torch

        resolved, t = _torch_f32(tensor, device)
        shape = list(t.shape)
        rows, cols = shape[0], 1
        for s in shape[1:]:
            cols *= int(s)
        q = _generate_orthogonal_torch(cols, seed, resolved, torch.float32)
        return (t.reshape(rows, cols) @ q).reshape(shape), seed
    arr = _numpy_float32_array(tensor)
    shape = [int(x) for x in arr.shape]
    rows, cols = shape[0], 1
    for s in shape[1:]:
        cols *= int(s)
    q = _generate_orthogonal_numpy(cols, seed)
    return (arr.reshape(rows, cols) @ q).reshape(shape), seed


def _unrotate_flat(flat, shape, rotation: str, seed: int):
    import numpy as np

    rows = int(shape[0])
    cols = 1
    for s in shape[1:]:
        cols *= int(s)
    arr = np.asarray(flat, dtype=np.float32)[: rows * cols].reshape(rows, cols)
    if rotation == "hadamard":
        unrotated = _fwht_numpy(arr)
    else:
        q = _generate_orthogonal_numpy(cols, seed)
        unrotated = arr @ q.T
    return unrotated.reshape(-1).tolist()


def _normalize_tensor_awq_block_max_torch(tensor, X, alpha, block_size, device):
    import torch

    resolved, arr = _torch_f32(tensor, device)
    X_t = X.to(device=resolved, dtype=torch.float32)
    shape = [int(x) for x in arr.shape]
    rows = arr.reshape(shape[0], -1)
    if X_t.shape[1] != rows.shape[1]:
        norm_tensor, block_scales, src_flat = _normalize_tensor_block_max_torch(
            tensor, block_size, device
        )
        return norm_tensor, block_scales, src_flat, None

    act_mag = X_t.abs().mean(dim=0).clamp(min=1e-6)
    s = act_mag**alpha
    awq_scales = 1.0 / s
    scaled_rows = rows / awq_scales[None, :]

    flat = scaled_rows.reshape(-1)
    n = int(flat.shape[0])
    pad = (-n) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(-1, block_size)
    scales = blocks.abs().amax(dim=1)
    safe = torch.where(scales == 0, torch.ones_like(scales), scales)
    normalized = (blocks / safe[:, None]).reshape(-1)
    if pad:
        normalized = normalized[:n]

    return (
        normalized.reshape(arr.shape),
        scales.detach().cpu(),
        arr.reshape(-1).detach().cpu(),
        awq_scales.detach().cpu(),
    )


def _normalize_tensor_block_max_torch(tensor, block_size: int, device):
    import torch

    _, arr = _torch_f32(tensor, device)
    flat = arr.reshape(-1)
    n = int(flat.shape[0])
    pad = (-n) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(-1, block_size)
    scales = blocks.abs().amax(dim=1)
    safe = torch.where(scales == 0, torch.ones_like(scales), scales)
    normalized = (blocks / safe[:, None]).reshape(-1)
    if pad:
        normalized = normalized[:n]
    return (
        normalized.reshape(arr.shape),
        scales.detach().cpu(),
        arr.reshape(-1).detach().cpu(),
    )


def _normalize_tensor_block_max_numpy(tensor, block_size: int):
    import numpy as np

    arr = _numpy_float32_array(tensor)
    flat = arr.reshape(-1)
    n = int(flat.shape[0])
    pad = (-n) % block_size
    if pad:
        flat = np.pad(flat, (0, pad), mode="constant")
    blocks = flat.reshape(-1, block_size)
    scales = np.abs(blocks).max(axis=1).astype(np.float32)
    safe = np.where(scales == 0, 1.0, scales).astype(np.float32)
    normalized = (blocks / safe[:, None]).reshape(-1)
    if pad:
        normalized = normalized[:n]
    return normalized.reshape(arr.shape), scales, arr.reshape(-1)


def _normalize_tensor_slrq_block_torch(tensor, block_size: int, device):
    import torch

    _, arr = _torch_f32(tensor, device)
    flat = arr.reshape(-1)
    n = int(flat.shape[0])
    pad = (-n) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(-1, block_size)
    
    # Salient protection
    abs_blocks = blocks.abs()
    salient_indices = abs_blocks.argmax(dim=1)
    row_indices = torch.arange(blocks.shape[0], device=device)
    salient_weights = blocks[row_indices, salient_indices].clone()
    
    blocks[row_indices, salient_indices] = 0.0
    max_rem = blocks.abs().amax(dim=1)
    safe = torch.where(max_rem == 0, torch.ones_like(max_rem), max_rem)
    safe = torch.exp2(torch.ceil(torch.log2(safe)))
    
    normalized = (blocks / safe[:, None]).reshape(-1)
    if pad:
        normalized = normalized[:n]
        
    return (
        normalized.reshape(arr.shape),
        safe.detach().cpu(),
        salient_weights.detach().cpu(),
        salient_indices.detach().cpu(),
        arr.reshape(-1).detach().cpu(),
    )


def _normalize_tensor_slrq_block_numpy(tensor, block_size: int):
    import numpy as np

    arr = _numpy_float32_array(tensor)
    flat = arr.reshape(-1)
    n = int(flat.shape[0])
    pad = (-n) % block_size
    if pad:
        flat = np.pad(flat, (0, pad), mode="constant")
    blocks = flat.reshape(-1, block_size)
    
    # Salient protection
    abs_blocks = np.abs(blocks)
    salient_indices = np.argmax(abs_blocks, axis=1)
    row_indices = np.arange(blocks.shape[0])
    salient_weights = blocks[row_indices, salient_indices].copy()
    
    blocks[row_indices, salient_indices] = 0.0
    max_rem = np.abs(blocks).max(axis=1)
    safe = np.where(max_rem == 0, 1.0, max_rem).astype(np.float32)
    safe = np.exp2(np.ceil(np.log2(safe))).astype(np.float32)
    
    normalized = (blocks / safe[:, np.newaxis]).reshape(-1)
    if pad:
        normalized = normalized[:n]
        
    return (
        normalized.reshape(arr.shape),
        safe,
        salient_weights,
        salient_indices,
        arr.reshape(-1),
    )


def _apply_block_max_scales(flat, scales, block_size: int):
    out = []
    n = len(flat)
    block_idx = 0
    for i in range(0, n, block_size):
        end = min(i + block_size, n)
        scale = float(scales[block_idx]) if block_idx < len(scales) else 1.0
        out.extend(float(flat[j]) * scale for j in range(i, end))
        block_idx += 1
    return out


def _apply_block_max_scales_numpy(flat, scales, block_size: int):
    import numpy as np

    arr = np.asarray(flat, dtype=np.float32)
    n = arr.shape[0]
    pad = (-n) % block_size
    if pad:
        arr = np.pad(arr, (0, pad), mode="constant")
    blocks = arr.reshape(-1, block_size)
    scale_arr = np.asarray(scales, dtype=np.float32)
    if scale_arr.shape[0] != blocks.shape[0]:
        raise ValueError(
            f"block scale count {scale_arr.shape[0]} != block count {blocks.shape[0]}"
        )
    out = (blocks * scale_arr[:, None]).reshape(-1)
    if pad:
        out = out[:n]
    return out


def _normalize_tensor_row_l2_numpy(tensor):
    import numpy as np

    arr = _numpy_float32_array(tensor)
    shape = [int(x) for x in arr.shape]
    rows = arr.reshape(shape[0], -1)
    scales = np.linalg.norm(rows, axis=1).astype(np.float32)
    safe = np.where(scales == 0, 1.0, scales).astype(np.float32)
    normalized = (rows / safe[:, None]).reshape(arr.shape)
    return normalized, scales, arr.reshape(-1)


def _normalize_tensor_col_l2_numpy(tensor):
    import numpy as np

    arr = _numpy_float32_array(tensor)
    shape = [int(x) for x in arr.shape]
    rows = arr.reshape(shape[0], -1)
    scales = np.linalg.norm(rows, axis=0).astype(np.float32)
    safe = np.where(scales == 0, 1.0, scales).astype(np.float32)
    normalized = (rows / safe[None, :]).reshape(arr.shape)
    return normalized, scales, arr.reshape(-1)


def _normalize_tensor_row_l2_torch(tensor, device):
    import torch

    _, arr = _torch_f32(tensor, device)
    rows = arr.reshape(arr.shape[0], -1)
    scales = torch.linalg.vector_norm(rows, ord=2, dim=1).to(dtype=torch.float32)
    safe = torch.where(scales == 0, torch.ones_like(scales), scales)
    normalized = (rows / safe[:, None]).reshape(arr.shape)
    return normalized, scales.detach().cpu(), arr.reshape(-1).detach().cpu()


def _normalize_tensor_awq_torch(tensor, X, alpha, device):
    import torch

    resolved, arr = _torch_f32(tensor, device)
    X_t = X.to(device=resolved, dtype=torch.float32)
    shape = [int(x) for x in arr.shape]
    rows = arr.reshape(shape[0], -1)
    if X_t.shape[1] != rows.shape[1]:
        return _normalize_tensor_col_l2_torch(tensor, device)
    act_mag = X_t.abs().mean(dim=0).clamp(min=1e-6)
    s = act_mag**alpha
    scales = 1.0 / s
    normalized = (rows / scales[None, :]).reshape(arr.shape)
    return normalized, scales.detach().cpu(), arr.reshape(-1).detach().cpu()


def _normalize_tensor_awq_numpy(tensor, X, alpha):
    import numpy as np

    arr = _numpy_float32_array(tensor)
    X_arr = _numpy_float32_array(X)
    shape = [int(x) for x in arr.shape]
    rows = arr.reshape(shape[0], -1)
    if X_arr.shape[1] != rows.shape[1]:
        return _normalize_tensor_col_l2_numpy(tensor)
    act_mag = np.mean(np.abs(X_arr), axis=0)
    act_mag = np.maximum(act_mag, 1e-6)
    s = act_mag**alpha
    scales = 1.0 / s
    normalized = (rows / scales[None, :]).reshape(arr.shape)
    return normalized, scales.astype(np.float32), arr.reshape(-1)


def _normalize_tensor_col_l2_torch(tensor, device):
    import torch

    _, arr = _torch_f32(tensor, device)
    rows = arr.reshape(arr.shape[0], -1)
    scales = torch.linalg.vector_norm(rows, ord=2, dim=0).to(dtype=torch.float32)
    safe = torch.where(scales == 0, torch.ones_like(scales), scales)
    normalized = (rows / safe[None, :]).reshape(arr.shape)
    return normalized, scales.detach().cpu(), arr.reshape(-1).detach().cpu()


def _apply_normalization(
    tensor, name, normalization, awq_activations, awq_alpha,
    block_scale_size, backend, device, awq_fallbacks,
):
    is_torch = backend == "torch"
    has_awq = awq_activations is not None and name in awq_activations
    awq_col_scales = None

    def _row_l2():
        return (_normalize_tensor_row_l2_torch(tensor, device) if is_torch
                else _normalize_tensor_row_l2_numpy(tensor))

    def _col_l2():
        return (_normalize_tensor_col_l2_torch(tensor, device) if is_torch
                else _normalize_tensor_col_l2_numpy(tensor))

    def _block_max():
        return (_normalize_tensor_block_max_torch(tensor, block_scale_size, device) if is_torch
                else _normalize_tensor_block_max_numpy(tensor, block_scale_size))

    def _slrq_block():
        # returns tensor, scales, salient_weights, salient_indices, source_flat
        return (_normalize_tensor_slrq_block_torch(tensor, block_scale_size, device) if is_torch
                else _normalize_tensor_slrq_block_numpy(tensor, block_scale_size))

    salient_weights = None
    salient_indices = None

    if normalization == "row-l2":
        tensor, row_scales, source_flat = _row_l2()
    elif normalization == "col-l2":
        tensor, row_scales, source_flat = _col_l2()
    elif normalization == "slrq-block":
        tensor, row_scales, salient_weights, salient_indices, source_flat = _slrq_block()
    elif normalization == "awq":
        if not has_awq:
            awq_fallbacks.append(name)
            tensor, row_scales, source_flat = _col_l2()
        elif is_torch:
            tensor, row_scales, source_flat = _normalize_tensor_awq_torch(
                tensor, awq_activations[name], awq_alpha, device)
        else:
            tensor, row_scales, source_flat = _normalize_tensor_awq_numpy(
                tensor, awq_activations[name], awq_alpha)
    elif normalization == "awq-block-max":
        if not is_torch:
            raise RuntimeError("awq-block-max requires --backend torch")
        if not has_awq:
            awq_fallbacks.append(name)
            tensor, row_scales, source_flat = _block_max()
        else:
            tensor, row_scales, source_flat, awq_col_scales = (
                _normalize_tensor_awq_block_max_torch(
                    tensor, awq_activations[name], awq_alpha, block_scale_size, device))
    else:
        tensor, row_scales, source_flat = _block_max()
    return tensor, row_scales, source_flat, awq_col_scales, salient_weights, salient_indices


def _apply_col_l2_scales(flat, shape, scales):
    try:
        import numpy as np
        return _apply_col_l2_scales_numpy(flat, shape, scales).tolist()
    except ImportError:
        pass

    rows = int(shape[0])
    cols = 1
    for s in shape[1:]:
        cols *= int(s)
    if len(scales) != cols:
        raise ValueError("col scale count does not match tensor cols")
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(float(flat[r * cols + c]) * float(scales[c]))
    return out


def _apply_col_l2_scales_numpy(flat, shape, scales):
    import numpy as np

    rows = int(shape[0])
    cols = 1
    for s in shape[1:]:
        cols *= int(s)
    arr = np.asarray(flat, dtype=np.float32)[: rows * cols].reshape(rows, cols)
    col_scales = np.asarray(scales, dtype=np.float32)
    if col_scales.shape[0] != cols:
        raise ValueError("col scale count does not match tensor cols")
    return (arr * col_scales[None, :]).reshape(-1)


def _extract_outliers(vectors, outlier_frac: float, packed_values: int):
    if outlier_frac is None or outlier_frac <= 0:
        return None, None, vectors
    n = int(packed_values)
    k = int(outlier_frac * n)
    if k <= 0:
        return None, None, vectors
    if _is_torch_tensor(vectors):
        import torch

        flat = vectors.reshape(-1).clone()
        relevant = flat[:n]
        _, topk_idx = torch.topk(relevant.abs(), k)
        positions = topk_idx.detach().cpu().to(torch.int64).numpy()
        values = flat[topk_idx].detach().cpu().to(torch.float32).numpy()
        flat[topk_idx] = 0
        return positions, values, flat.reshape(vectors.shape)
    if _is_numpy_array(vectors):
        import numpy as np

        flat = vectors.reshape(-1).copy()
        relevant = flat[:n]
        order = np.argpartition(np.abs(relevant), -k)[-k:]
        positions = order.astype(np.int64)
        values = flat[positions].astype(np.float32)
        flat[positions] = 0
        return positions, values, flat.reshape(vectors.shape)
    flat = []
    for row in vectors:
        flat.extend(float(v) for v in row)
    pairs = [(abs(flat[i]), i) for i in range(n)]
    pairs.sort(reverse=True)
    positions = [p for _, p in pairs[:k]]
    values = [flat[p] for p in positions]
    for p in positions:
        flat[p] = 0.0
    g = len(vectors[0])
    new_vectors = [flat[i * g : (i + 1) * g] for i in range(len(vectors))]
    return positions, values, new_vectors


_FP16_MAX = 65504.0


def _write_outliers(idx_path: Path, val_path: Path, positions, values) -> None:
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("outlier writing requires numpy") from exc
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    val_arr = np.asarray(values, dtype=np.float32)
    over = np.abs(val_arr) > _FP16_MAX
    if bool(over.any()):
        n_over = int(over.sum())
        max_abs = float(np.max(np.abs(val_arr)))
        print(
            f"WARN: {n_over} outlier value(s) exceed fp16 range ({max_abs:.1f} > {_FP16_MAX}); "
            f"will be clipped to ±inf at {val_path.name}",
            file=os.sys.stderr,
        )
    np.asarray(positions, dtype="<u4").tofile(str(idx_path))
    val_arr.astype("<f2").tofile(str(val_path))


def _read_outliers(idx_path: Path, val_path: Path) -> tuple[list[int], list[float]]:
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("outlier reading requires numpy") from exc
    positions = np.fromfile(str(idx_path), dtype="<u4")
    values = np.fromfile(str(val_path), dtype="<f2").astype(np.float32)
    if len(positions) != len(values):
        raise ValueError(f"outlier count mismatch: {len(positions)} != {len(values)}")
    return positions.tolist(), values.tolist()


def _kmeans_pp_init_torch(
    rows, k: int, seed: int | None = None, oversample_factor: float = 2.0
):
    import torch

    n = int(rows.shape[0])
    d = int(rows.shape[1])
    if k >= n:
        return rows.clone()
    gen = torch.Generator(device=rows.device)
    if seed is not None:
        gen.manual_seed(int(seed) & ((1 << 63) - 1))

    # K-Means|| (Scalable K-Means++)
    first = int(torch.randint(n, (1,), generator=gen, device=rows.device).item())
    centroids = [rows[first]]
    min_d2 = torch.sum((rows - rows[first]) ** 2, dim=1)

    # We sample ~ l points per step. l = oversample_factor * k
    # We do this log(n) times, but usually a small constant like 5 is enough
    for _ in range(5):
        if len(centroids) >= k:
            break
        sum_d2 = min_d2.sum().item()
        if sum_d2 == 0:
            break
        probs = min_d2 / sum_d2
        # Sample l points
        l = int(oversample_factor * k)
        rand_vals = torch.rand(n, generator=gen, device=rows.device)
        chosen = torch.where(rand_vals < probs * l)[0]

        if chosen.numel() == 0:
            break

        new_centers = rows[chosen]
        for c in new_centers:
            centroids.append(c)
            
        # Update min_d2 efficiently using GEMM
        # Pre-calculate squared norms for new centers
        c_norm_sq = torch.sum(new_centers * new_centers, dim=1, keepdim=True).T
        
        batch_size = max(1024, (1 << 28) // max(int(new_centers.shape[0]), 1))
        for i in range(0, n, batch_size):
            batch_rows = rows[i : i + batch_size]
            r_norm_sq = torch.sum(batch_rows * batch_rows, dim=1, keepdim=True)
            
            dists = torch.addmm(
                (r_norm_sq + c_norm_sq),
                batch_rows,
                new_centers.T,
                alpha=-2.0,
                beta=1.0
            )
            
            min_d2[i : i + batch_size] = torch.minimum(
                min_d2[i : i + batch_size], dists.min(dim=1)[0]
            )
        del dists, batch_rows, new_centers


    centroids = torch.stack(centroids)
    if centroids.shape[0] > k:
        # If we oversampled, run K-Means++ on the sampled set to reduce to k
        subset = centroids
        final_centers = [subset[0]]
        sub_d2 = torch.sum((subset - subset[0]) ** 2, dim=1)
        for _ in range(1, k):
            sum_d2 = sub_d2.sum().item()
            if sum_d2 == 0:
                break
            probs = sub_d2 / sum_d2
            cumprobs = torch.cumsum(probs, dim=0)
            r = torch.rand(1, generator=gen, device=rows.device).item()
            chosen_idx = int(torch.searchsorted(cumprobs, r).item())
            chosen_idx = min(chosen_idx, subset.shape[0] - 1)
            final_centers.append(subset[chosen_idx])
            d2 = torch.sum((subset - subset[chosen_idx]) ** 2, dim=1)
            sub_d2 = torch.minimum(sub_d2, d2)
        centroids = torch.stack(final_centers)

    # Pad if we have less than k
    while centroids.shape[0] < k:
        centroids = torch.cat(
            [centroids, rows[torch.randint(n, (1,), generator=gen, device=rows.device)]]
        )

    return centroids[:k]


def _kmeans_pp_init_numpy(rows, k: int, seed: int | None = None):
    import numpy as np

    n, d = rows.shape
    if k >= n:
        return rows.copy()
    centroids = np.empty((k, d), dtype=np.float32)
    rng = (
        np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)
        if seed is not None
        else np.random.default_rng()
    )
    centroids[0] = rows[rng.integers(n)]
    min_d2 = np.full(n, np.inf, dtype=np.float64)
    for i in range(1, k):
        diff = rows - centroids[i - 1]
        d2 = np.sum(diff * diff, axis=1, dtype=np.float64)
        np.minimum(min_d2, d2, out=min_d2)
        total = float(min_d2.sum())
        if not math.isfinite(total) or total <= 0:
            idx = int(rng.integers(n))
        else:
            probs = min_d2 / total
            idx = int(rng.choice(n, p=probs))
        centroids[i] = rows[idx]
    return centroids


def _sample_vector_rows(vectors, sample_vectors: int | None):
    if sample_vectors is None or sample_vectors <= 0 or sample_vectors >= len(vectors):
        return vectors
    if _is_torch_tensor(vectors):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch vector sampling requires torch") from exc
        positions = (
            torch.linspace(
                0,
                len(vectors) - 1,
                steps=sample_vectors,
                device=vectors.device,
                dtype=torch.float64,
            )
            .round()
            .to(dtype=torch.long)
            .clamp_(max=len(vectors) - 1)
        )
        return vectors.index_select(0, positions)
    if hasattr(vectors, "shape") and hasattr(vectors, "__getitem__"):
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("NumPy vector sampling requires numpy") from exc
        positions = np.linspace(0, len(vectors) - 1, sample_vectors, dtype=np.int64)
        return vectors[positions]

    if sample_vectors == 1:
        return [vectors[len(vectors) // 2]]
    last = len(vectors) - 1
    return [
        vectors[round(i * last / (sample_vectors - 1))] for i in range(sample_vectors)
    ]


def _is_numpy_array(value: object) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "reshape")
        and hasattr(value, "astype")
        and hasattr(value, "tolist")
    )


def _is_torch_tensor(value: object) -> bool:
    return hasattr(value, "detach") and hasattr(value, "to")


def _supports_numpy_backend(value: object) -> bool:
    return _is_numpy_array(value) or hasattr(value, "detach")


def _resolve_torch_device(device: str):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch backend requires torch") from exc

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "torch backend requested CUDA, but torch.cuda.is_available() is false"
        )
    return resolved


def _cuda_sm_major(device_index: int = 0) -> int | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability(device_index)[0]
    except Exception:
        return None


def _maybe_fallback_cuda_to_cpu(device: str, backend: str) -> str:
    """Fall back to CPU if the CUDA device is sm_60 (P100 etc.) — PyTorch requires sm_70+."""
    if backend != "torch":
        return device
    dev_lower = device.lower()
    if "cuda" not in dev_lower and dev_lower != "auto":
        return device
    sm = _cuda_sm_major()
    if sm is not None and sm < 7:
        import sys
        print(
            f"WARNING: CUDA device is SM {sm}.x (< 7.0) — not supported by this PyTorch build. "
            f"Falling back to CPU automatically.",
            file=sys.stderr,
            flush=True,
        )
        return "cpu"
    return device


def _torch_f32(tensor, device):
    import torch

    resolved = _resolve_torch_device(device)
    if _is_torch_tensor(tensor):
        return resolved, tensor.detach().to(device=resolved, dtype=torch.float32)
    return resolved, torch.as_tensor(tensor, dtype=torch.float32, device=resolved)


def _numpy_assign(vectors, codebook, chunk_size: int = 65536):
    import numpy as np

    rows = np.asarray(vectors, dtype=np.float32)
    centroids = np.asarray(codebook, dtype=np.float32)
    indices = np.empty(rows.shape[0], dtype=np.int64)
    total = 0.0
    width = rows.shape[1]
    
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
    c_norm_sq = np.sum(centroids * centroids, axis=1)

    for start in range(0, rows.shape[0], chunk_size):
        end = min(start + chunk_size, rows.shape[0])
        chunk = rows[start:end]
        r_norm_sq = np.sum(chunk * chunk, axis=1)
        
        # GEMM for the cross term
        # dists = r_norm_sq[:, None] + c_norm_sq[None, :] - 2 * (chunk @ centroids.T)
        dists = r_norm_sq[:, None] + c_norm_sq[None, :] - 2 * np.dot(chunk, centroids.T)
        
        chosen = np.argmin(dists, axis=1)
        indices[start:end] = chosen
        total += float(dists[np.arange(chosen.shape[0]), chosen].sum())

    return indices, total / (rows.shape[0] * width)


def _torch_float32_matrix(values, device: str):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch backend requires torch") from exc

    resolved = _resolve_torch_device(device)
    if _is_torch_tensor(values):
        rows = values.detach().to(device=resolved, dtype=torch.float32)
    else:
        rows = torch.as_tensor(values, dtype=torch.float32, device=resolved)
    if rows.ndim != 2:
        raise ValueError("torch VQ expects a 2D vector matrix")
    return rows.contiguous()


def _torch_assign(vectors, codebook, device: str, chunk_size: int = 65536):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch backend requires torch") from exc

    resolved = _resolve_torch_device(device)
    # Detect if we can use half precision (FP16 is generally safe for distance ranking)
    use_half = resolved.type == "cuda"
    dtype = torch.float16 if use_half else torch.float32

    rows = _torch_float32_matrix(vectors, device).to(dtype)
    centroids = _torch_float32_matrix(codebook, device).to(dtype)
    
    indices_parts = []
    total = 0.0
    width = int(rows.shape[1])
    k = int(centroids.shape[0])
    
    # Pre-calculate squared norms for centroids: ||b||^2
    c_norm_sq = torch.sum(centroids.to(torch.float32) * centroids.to(torch.float32), dim=1, keepdim=True).T.to(dtype)
    
    effective_chunk = max(256, min(chunk_size, (1 << 28) // max(k, 1)))

    with torch.no_grad():
        for start in range(0, int(rows.shape[0]), effective_chunk):
            end = min(start + effective_chunk, int(rows.shape[0]))
            chunk = rows[start:end]
            
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
            r_norm_sq = torch.sum(chunk.to(torch.float32) * chunk.to(torch.float32), dim=1, keepdim=True).to(dtype)
            
            dists = torch.addmm(
                (r_norm_sq + c_norm_sq),
                chunk,
                centroids.T,
                alpha=-2.0,
                beta=1.0
            )
            
            chosen = torch.argmin(dists, dim=1)
            indices_parts.append(chosen.detach().cpu())
            
            # Accumulate error in float32 for precision
            total += float(
                dists[torch.arange(chosen.shape[0], device=rows.device), chosen]
                .to(torch.float32)
                .sum()
                .detach()
                .cpu()
                .item()
            )

    indices = (
        torch.cat(indices_parts).to(dtype=torch.int64)
        if indices_parts
        else torch.empty(0, dtype=torch.int64)
    )
    return indices, total / (int(rows.shape[0]) * width)


def _quality_from_totals(
    value_count: int,
    sse: float,
    abs_error_sum: float,
    max_abs_error: float,
    source_l2_sq: float,
    reconstructed_l2_sq: float,
    dot: float,
) -> dict:
    mse = sse / value_count if value_count else 0.0
    rmse = math.sqrt(mse)
    if source_l2_sq > 0:
        relative_rmse = math.sqrt(sse / source_l2_sq)
    else:
        relative_rmse = 0.0 if sse == 0 else float("inf")

    denom = math.sqrt(source_l2_sq) * math.sqrt(reconstructed_l2_sq)
    if denom > 0:
        cosine = dot / denom
    else:
        cosine = 1.0 if source_l2_sq == 0 and reconstructed_l2_sq == 0 else 0.0

    return {
        "value_count": value_count,
        "sse": sse,
        "mse": mse,
        "rmse": rmse,
        "mae": abs_error_sum / value_count if value_count else 0.0,
        "max_abs_error": max_abs_error,
        "source_l2_sq": source_l2_sq,
        "reconstructed_l2_sq": reconstructed_l2_sq,
        "dot": dot,
        "relative_rmse": relative_rmse,
        "cosine_similarity": cosine,
    }


def quality_metrics_from_flat(
    source: Sequence[float], reconstructed: Sequence[float]
) -> dict:
    if len(source) != len(reconstructed):
        raise ValueError("source and reconstructed values must have the same length")

    try:
        import numpy as np
        use_numpy = True
    except ImportError:
        use_numpy = False

    if use_numpy:
        return _quality_metrics_for_numpy_flat(source, reconstructed)

    sse = 0.0
    abs_error_sum = 0.0
    max_abs_error = 0.0
    source_l2_sq = 0.0
    reconstructed_l2_sq = 0.0
    dot = 0.0

    for src, rec in zip(source, reconstructed):
        err = float(src) - float(rec)
        abs_err = abs(err)
        sse += err * err
        abs_error_sum += abs_err
        max_abs_error = max(max_abs_error, abs_err)
        source_l2_sq += float(src) * float(src)
        reconstructed_l2_sq += float(rec) * float(rec)
        dot += float(src) * float(rec)

    return _quality_from_totals(
        value_count=len(source),
        sse=sse,
        abs_error_sum=abs_error_sum,
        max_abs_error=max_abs_error,
        source_l2_sq=source_l2_sq,
        reconstructed_l2_sq=reconstructed_l2_sq,
        dot=dot,
    )


def _quality_metrics_for_torch_vectors(
    vectors, codebook, indices, device: str, chunk_size: int = 65536
) -> dict:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch backend requires torch") from exc

    rows = _torch_float32_matrix(vectors, device)
    centroids = _torch_float32_matrix(codebook, device)
    assigned = (
        indices.detach().to(device=rows.device, dtype=torch.long)
        if _is_torch_tensor(indices)
        else torch.as_tensor(indices, dtype=torch.long, device=rows.device)
    )

    sse = 0.0
    abs_error_sum = 0.0
    max_abs_error = 0.0
    source_l2_sq = 0.0
    reconstructed_l2_sq = 0.0
    dot = 0.0

    with torch.no_grad():
        for start in range(0, int(rows.shape[0]), chunk_size):
            end = min(start + chunk_size, int(rows.shape[0]))
            source = rows[start:end]
            reconstructed = centroids[assigned[start:end]]
            diff = source - reconstructed
            abs_diff = diff.abs()
            sse += float((diff * diff).sum().detach().cpu().item())
            abs_error_sum += float(abs_diff.sum().detach().cpu().item())
            max_abs_error = max(
                max_abs_error,
                float(abs_diff.max().detach().cpu().item())
                if abs_diff.numel()
                else 0.0,
            )
            source_l2_sq += float((source * source).sum().detach().cpu().item())
            reconstructed_l2_sq += float(
                (reconstructed * reconstructed).sum().detach().cpu().item()
            )
            dot += float((source * reconstructed).sum().detach().cpu().item())

    return _quality_from_totals(
        value_count=int(rows.numel()),
        sse=sse,
        abs_error_sum=abs_error_sum,
        max_abs_error=max_abs_error,
        source_l2_sq=source_l2_sq,
        reconstructed_l2_sq=reconstructed_l2_sq,
        dot=dot,
    )


def _quality_metrics_for_numpy_vectors(
    vectors, codebook, indices, chunk_size: int = 65536
) -> dict:
    import numpy as np

    rows = np.asarray(vectors, dtype=np.float32)
    centroids = np.asarray(codebook, dtype=np.float32)
    assigned = np.asarray(indices, dtype=np.int64)

    sse = 0.0
    abs_error_sum = 0.0
    max_abs_error = 0.0
    source_l2_sq = 0.0
    reconstructed_l2_sq = 0.0
    dot = 0.0

    for start in range(0, rows.shape[0], chunk_size):
        end = min(start + chunk_size, rows.shape[0])
        source = rows[start:end]
        reconstructed = centroids[assigned[start:end]]
        diff = source - reconstructed
        abs_diff = np.abs(diff)
        sse += float(np.sum(diff * diff))
        abs_error_sum += float(np.sum(abs_diff))
        max_abs_error = max(
            max_abs_error, float(np.max(abs_diff)) if abs_diff.size else 0.0
        )
        source_l2_sq += float(np.sum(source * source))
        reconstructed_l2_sq += float(np.sum(reconstructed * reconstructed))
        dot += float(np.sum(source * reconstructed))

    return _quality_from_totals(
        value_count=int(rows.size),
        sse=sse,
        abs_error_sum=abs_error_sum,
        max_abs_error=max_abs_error,
        source_l2_sq=source_l2_sq,
        reconstructed_l2_sq=reconstructed_l2_sq,
        dot=dot,
    )


def _quality_metrics_for_numpy_flat(
    source, reconstructed, chunk_size: int = 1_000_000
) -> dict:
    import numpy as np

    src = np.asarray(source, dtype=np.float32).reshape(-1)
    rec = np.asarray(reconstructed, dtype=np.float32).reshape(-1)
    if src.shape[0] != rec.shape[0]:
        raise ValueError("source and reconstructed arrays must have the same size")

    sse = 0.0
    abs_error_sum = 0.0
    max_abs_error = 0.0
    source_l2_sq = 0.0
    reconstructed_l2_sq = 0.0
    dot = 0.0

    for start in range(0, src.shape[0], chunk_size):
        end = min(start + chunk_size, src.shape[0])
        s = src[start:end]
        r = rec[start:end]
        diff = s - r
        abs_diff = np.abs(diff)
        sse += float(np.sum(diff * diff))
        abs_error_sum += float(np.sum(abs_diff))
        max_abs_error = max(
            max_abs_error, float(np.max(abs_diff)) if abs_diff.size else 0.0
        )
        source_l2_sq += float(np.sum(s * s))
        reconstructed_l2_sq += float(np.sum(r * r))
        dot += float(np.sum(s * r))

    return _quality_from_totals(
        value_count=int(src.shape[0]),
        sse=sse,
        abs_error_sum=abs_error_sum,
        max_abs_error=max_abs_error,
        source_l2_sq=source_l2_sq,
        reconstructed_l2_sq=reconstructed_l2_sq,
        dot=dot,
    )


def _learn_codebook_numpy(
    vectors, codebook_size: int, iterations: int, seed: int | None = None,
    initial_codebook=None,
):
    import numpy as np

    rows = np.asarray(vectors, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError("NumPy VQ expects a 2D vector matrix")
    if rows.shape[0] == 0:
        raise ValueError("at least one vector is required")
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    n = rows.shape[0]
    k = min(codebook_size, n)
    effective_iters = min(iterations, 3) if k >= int(n * 0.9) else iterations
    if initial_codebook is not None:
        codebook = np.asarray(initial_codebook, dtype=np.float32)[:k].copy()
    elif k == 1:
        codebook = rows[[n // 2]].copy()
    else:
        codebook = _kmeans_pp_init_numpy(rows, k, seed=seed)

    for _ in range(effective_iters):
        indices, _ = _numpy_assign(rows, codebook)
        sums = np.zeros_like(codebook)
        counts = np.bincount(indices, minlength=k).astype(np.float32)
        np.add.at(sums, indices, rows)
        nonzero = counts > 0
        codebook[nonzero] = sums[nonzero] / counts[nonzero, None]

    indices, mse = _numpy_assign(rows, codebook)
    return codebook, indices, float(mse)


def _learn_codebook_torch(
    vectors,
    codebook_size: int,
    iterations: int,
    device: str,
    vector_weights=None,
    seed: int | None = None,
    initial_codebook=None,
):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch backend requires torch") from exc

    rows = _torch_float32_matrix(vectors, device)
    if rows.shape[0] == 0:
        raise ValueError("at least one vector is required")
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    n = int(rows.shape[0])
    k = min(int(codebook_size), n)
    # When k >= n/2 each centroid has ~1 sample on average; centroids barely move after init.
    effective_iters = min(iterations, 3) if k >= int(n * 0.9) else iterations
    with torch.no_grad():
        if initial_codebook is not None:
            codebook = _torch_float32_matrix(initial_codebook, str(rows.device))[:k].clone()
        elif k == 1:
            codebook = rows[[n // 2]].clone()
        else:
            codebook = _kmeans_pp_init_torch(rows, k, seed=seed)

        sums = torch.zeros_like(codebook)
        counts = torch.zeros(k, dtype=torch.float32, device=rows.device)

        for _ in range(effective_iters):
            if vector_weights is not None:
                W = torch.as_tensor(
                    vector_weights, dtype=torch.float32, device=rows.device
                )
                weighted_rows = rows * torch.sqrt(W)
                weighted_cb = codebook * torch.sqrt(W)
                indices, _ = _torch_assign(weighted_rows, weighted_cb, str(rows.device))
            else:
                indices, _ = _torch_assign(rows, codebook, str(rows.device))
            
            chosen = indices.to(device=rows.device, dtype=torch.long)
            
            # Reuse buffers
            sums.zero_()
            counts.zero_()
            
            sums.index_add_(0, chosen, rows)
            counts.index_put_((chosen,), torch.ones(len(chosen), device=rows.device), accumulate=True)
            
            nonzero = counts > 0
            codebook[nonzero] = sums[nonzero] / counts[nonzero, None]

        indices, mse = _torch_assign(rows, codebook, str(rows.device))
    return codebook.detach().cpu(), indices, float(mse)


def learn_codebook_auto(
    vectors,
    codebook_size: int,
    iterations: int,
    backend: str,
    device: str = "cpu",
    vector_weights=None,
    seed: int | None = None,
    initial_codebook=None,
):
    if backend not in {"auto", "numpy", "torch"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'torch'")
    if backend == "torch":
        return _learn_codebook_torch(
            vectors,
            codebook_size,
            iterations,
            device,
            vector_weights=vector_weights,
            seed=seed,
            initial_codebook=initial_codebook,
        )
    if not _is_numpy_array(vectors):
        raise RuntimeError("NumPy backend requires NumPy array tensors")
    return _learn_codebook_numpy(vectors, codebook_size, iterations, seed=seed,
                                 initial_codebook=initial_codebook)


def quantize_vectors_auto(vectors, codebook, backend: str, device: str = "cpu"):
    if backend not in {"auto", "numpy", "torch"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'torch'")
    if backend == "torch":
        return _torch_assign(vectors, codebook, device)
    if not _is_numpy_array(vectors):
        raise RuntimeError("NumPy backend requires NumPy array tensors")
    return _numpy_assign(vectors, codebook)


def _index_bits_for_size(codebook_size: int) -> int:
    return math.ceil(math.log2(codebook_size)) if codebook_size > 1 else 1


def _safe_tensor_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return safe.strip("._") or "tensor"


def classify_tensor_family(name: str) -> str:
    lowered = name.lower()
    if any(marker in lowered for marker in ("embed", "embedding", "wte", "wpe")):
        return "embedding"
    if any(
        marker in lowered
        for marker in (".mlp.", "mlp", "gate_proj", "up_proj", "down_proj", "c_fc")
    ):
        return "mlp"
    if any(
        marker in lowered
        for marker in (
            "attn",
            "attention",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "c_attn",
        )
    ):
        return "attention"
    return "other"


_INDEX_BIT_SPECS = (
    (8, "<u1", "<B"),
    (16, "<u2", "<H"),
    (32, "<u4", "<I"),
    (64, "<u8", "<Q"),
)


def _index_bit_spec(index_bits: int) -> tuple[int, str, str]:
    for ceiling, np_dtype, struct_fmt in _INDEX_BIT_SPECS:
        if index_bits <= ceiling:
            return ceiling, np_dtype, struct_fmt
    raise ValueError(f"index_bits above 64 are not supported: got {index_bits}")


def _write_indices(path: Path, indices: Sequence[int], index_bits: int) -> None:
    ceiling, np_dtype, struct_fmt = _index_bit_spec(index_bits)
    if _is_torch_tensor(indices):
        indices = indices.detach().cpu().tolist()
    if hasattr(indices, "astype") and hasattr(indices, "tobytes"):
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("NumPy index writing requires numpy") from exc
        path.write_bytes(np.asarray(indices, dtype=np_dtype).tobytes())
        return
    if ceiling == 8:
        path.write_bytes(bytes(int(i) & 0xFF for i in indices))
        return
    with path.open("wb") as f:
        for index in indices:
            f.write(struct.pack(struct_fmt, int(index)))


def _write_codebook(path: Path, codebook: Sequence[Sequence[float]]) -> None:
    if _is_torch_tensor(codebook):
        codebook = codebook.detach().cpu().tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for row in codebook:
            for value in row:
                f.write(struct.pack("<f", float(value)))


def _write_f32_vector(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_torch_tensor(values):
        values = values.detach().cpu().tolist()
    if hasattr(values, "astype") and hasattr(values, "tobytes"):
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("NumPy scale writing requires numpy") from exc
        path.write_bytes(np.asarray(values, dtype="<f4").tobytes())
        return

    with path.open("wb") as f:
        for value in values:
            f.write(struct.pack("<f", float(value)))


def _read_f32_vector(path: Path, expected_count: int) -> list[float]:
    data = path.read_bytes()
    expected = expected_count * 4
    if len(data) != expected:
        raise ValueError(
            f"f32 vector size mismatch for {path}: expected {expected}, got {len(data)}"
        )
    return [value[0] for value in struct.iter_unpack("<f", data)]


def _read_codebook(
    path: Path, codebook_size: int, group_size: int
) -> list[list[float]]:
    data = path.read_bytes()
    expected = codebook_size * group_size * 4
    if len(data) != expected:
        raise ValueError(
            f"codebook size mismatch for {path}: expected {expected}, got {len(data)}"
        )
    floats = [value[0] for value in struct.iter_unpack("<f", data)]
    return [floats[i : i + group_size] for i in range(0, len(floats), group_size)]


def _read_indices(path: Path, index_bits: int, expected_count: int) -> list[int]:
    ceiling, _, struct_fmt = _index_bit_spec(index_bits)
    data = path.read_bytes()
    bytes_per = ceiling // 8
    if ceiling == 8:
        indices = list(data)
    else:
        if len(data) % bytes_per:
            raise ValueError(
                f"index file size not multiple of {bytes_per} bytes: {path}"
            )
        indices = [value[0] for value in struct.iter_unpack(struct_fmt, data)]
    if len(indices) != expected_count:
        raise ValueError(
            f"index count mismatch for {path}: expected {expected_count}, got {len(indices)}"
        )
    return indices


def _load_tensors(path: Path) -> Iterable[tuple[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open() as f:
            loaded = json.load(f)
        tensors = loaded.get("tensors", loaded) if isinstance(loaded, dict) else loaded
        if not isinstance(tensors, dict):
            raise ValueError("JSON input must be an object or contain a tensors object")
        for name, tensor in tensors.items():
            yield name, tensor
        return

    if suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except Exception as exc:
            raise RuntimeError(
                "safetensors input requires the safetensors package"
            ) from exc
        try:
            import torch  # noqa: F401
        except Exception:
            framework = "np"
        else:
            framework = "pt"
        with safe_open(str(path), framework=framework) as handle:
            for name in handle.keys():
                try:
                    yield name, handle.get_tensor(name)
                except TypeError as exc:
                    if framework == "np":
                        raise RuntimeError(
                            f"safetensors tensor {name} uses a dtype that requires torch loading"
                        ) from exc
                    raise
        return

    if suffix in {".pt", ".pth", ".bin"}:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch checkpoint input requires torch") from exc
        loaded = torch.load(path, map_location="cpu")
        state = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else loaded
        if not isinstance(state, dict):
            raise ValueError(
                "checkpoint must load to a tensor dictionary or contain state_dict"
            )
        for name, tensor in state.items():
            yield name, tensor
        return

    raise ValueError(f"unsupported input format: {path.suffix}")


def _infer_shape(value: object) -> list[int]:
    if isinstance(value, (list, tuple)):
        if not value:
            return [0]
        first = _infer_shape(value[0])
        for item in value[1:]:
            if _infer_shape(item) != first:
                raise ValueError(
                    "nested JSON tensor values must have a rectangular shape"
                )
        return [len(value)] + first
    return []


def _flatten_nested(value: object) -> list[float]:
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_nested(item))
        return out
    return [float(value)]


def _flatten_float_values(tensor: object, limit: int | None = None) -> list[float]:
    if hasattr(tensor, "detach"):
        flat = tensor.detach().float().cpu().reshape(-1)
        if limit is not None:
            flat = flat[:limit]
        return [float(x) for x in flat.tolist()]

    if hasattr(tensor, "reshape") and hasattr(tensor, "tolist"):
        flat = tensor.reshape(-1)
        if hasattr(flat, "astype"):
            flat = flat.astype("float32")
        values = [float(x) for x in flat.tolist()]
        return values[:limit] if limit is not None else values

    if isinstance(tensor, (list, tuple)):
        values = _flatten_nested(tensor)
        return values[:limit] if limit is not None else values

    raise TypeError("expected a tensor-like object")


def _numpy_float32_array(tensor: object):
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("NumPy backend requires numpy") from exc

    if hasattr(tensor, "detach"):
        return tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(tensor, dtype=np.float32)


def _numpy_vectors_from_tensor(tensor: object, group_size: int, limit: int | None):
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("NumPy backend requires numpy") from exc
    flat = _numpy_float32_array(tensor).reshape(-1)
    if limit is not None:
        flat = flat[:limit]
    original_len = int(flat.shape[0])
    remainder = original_len % group_size
    if remainder:
        flat = np.pad(flat, (0, group_size - remainder), mode="constant")
    return original_len, int(flat.shape[0]), flat.reshape(-1, group_size)


def _numpy_vectors_from_tensor_row_l2(
    tensor: object, group_size: int, limit: int | None
):
    if limit is not None:
        raise ValueError("row-l2 normalization does not support max_values_per_tensor")
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("NumPy backend requires numpy") from exc
    arr = _numpy_float32_array(tensor)
    shape = [int(x) for x in arr.shape]
    rows = arr.reshape(shape[0], -1)
    scales = np.linalg.norm(rows, axis=1).astype(np.float32)
    safe_scales = np.where(scales == 0, 1.0, scales).astype(np.float32)
    normalized = rows / safe_scales[:, None]
    flat = normalized.reshape(-1)
    original_len = int(flat.shape[0])
    remainder = original_len % group_size
    if remainder:
        flat = np.pad(flat, (0, group_size - remainder), mode="constant")
    return (
        original_len,
        int(flat.shape[0]),
        flat.reshape(-1, group_size),
        scales,
        arr.reshape(-1),
    )


def _torch_vectors_from_tensor(
    tensor: object, group_size: int, limit: int | None, device: str
):
    import torch

    _, arr = _torch_f32(tensor, device)
    flat = arr.reshape(-1)
    if limit is not None:
        flat = flat[:limit]
    original_len = int(flat.shape[0])
    remainder = original_len % group_size
    if remainder:
        flat = torch.nn.functional.pad(flat, (0, group_size - remainder))
    return original_len, int(flat.shape[0]), flat.reshape(-1, group_size)


def _torch_vectors_from_tensor_row_l2(
    tensor: object, group_size: int, limit: int | None, device: str
):
    if limit is not None:
        raise ValueError("row-l2 normalization does not support max_values_per_tensor")
    import torch

    _, arr = _torch_f32(tensor, device)
    rows = arr.reshape(arr.shape[0], -1)
    scales = torch.linalg.vector_norm(rows, ord=2, dim=1).to(dtype=torch.float32)
    safe = torch.where(scales == 0, torch.ones_like(scales), scales)
    flat = (rows / safe[:, None]).reshape(-1)
    original_len = int(flat.shape[0])
    remainder = original_len % group_size
    if remainder:
        flat = torch.nn.functional.pad(flat, (0, group_size - remainder))
    return (
        original_len,
        int(flat.shape[0]),
        flat.reshape(-1, group_size),
        scales.detach().cpu(),
        arr.reshape(-1).detach().cpu(),
    )


def _tensor_shape(tensor: object) -> list[int]:
    shape = getattr(tensor, "shape", None)
    if shape is not None:
        return [int(x) for x in shape]
    return _infer_shape(tensor) if isinstance(tensor, (list, tuple)) else []


def _tensor_numel(tensor: object) -> int:
    if hasattr(tensor, "numel"):
        return int(tensor.numel())
    size = getattr(tensor, "size", None)
    if isinstance(size, int):
        return size
    if isinstance(tensor, (list, tuple)):
        shape = _infer_shape(tensor)
        total = 1
        for dim in shape:
            total *= dim
        return total
    return 0


def inspect_checkpoint(path: Path) -> dict:
    tensors = []
    total_params = 0
    for name, tensor in _load_tensors(path):
        numel = _tensor_numel(tensor)
        shape = _tensor_shape(tensor)
        if numel <= 0:
            continue
        total_params += numel
        tensors.append(
            {
                "name": name,
                "shape": shape,
                "numel": numel,
                "candidate": len(shape) >= 2,
            }
        )
    return {
        "source": str(path),
        "tensor_count": len(tensors),
        "total_params": total_params,
        "tensors": tensors,
    }


def _concat_vector_parts(parts: Sequence[object]):
    if not parts:
        raise ValueError("cannot concatenate empty vector group")
    if _is_torch_tensor(parts[0]):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch vector concatenation requires torch") from exc
        return torch.cat(list(parts), dim=0)
    if _is_numpy_array(parts[0]):
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("NumPy vector concatenation requires numpy") from exc
        return np.concatenate(parts, axis=0)

    out = []
    for part in parts:
        out.extend(part)
    return out


QUANT_SPEC_MAX_PER_STAGE_BITS = 64
QUANT_SPEC_MAX_TOTAL_BITS = 64

RVQ_MIXED_FAMILY_BITS = {
    "embedding": [16, 16, 16],
    "attention": [16, 8],
    "mlp": [16, 8],
    "other": [16],
}


def rvq_mixed_family_stages() -> dict[str, list[int]]:
    return {fam: [1 << b for b in bits] for fam, bits in RVQ_MIXED_FAMILY_BITS.items()}


def is_rvq_mixed_spec(spec: str | None) -> bool:
    return spec == "rvq-mixed"


def parse_quant_spec(spec: str) -> list[int]:
    if not isinstance(spec, str):
        raise ValueError(f"quant spec must be a string: {spec!r}")
    if spec.startswith("rvq-"):
        body = spec[4:]
        prefix = "rvq-"
    elif spec.startswith("vq-"):
        body = spec[3:]
        prefix = "vq-"
    else:
        raise ValueError(f"quant spec must start with 'vq-' or 'rvq-': {spec!r}")
    parts = body.split("-")
    if not parts or not all(p for p in parts):
        raise ValueError(f"empty stage in quant spec: {spec!r}")
    bits = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"non-integer bits in quant spec: {p!r}")
        b = int(p)
        if b < 1 or b > QUANT_SPEC_MAX_PER_STAGE_BITS:
            raise ValueError(
                f"per-stage bits must be 1..{QUANT_SPEC_MAX_PER_STAGE_BITS}: got {b}"
            )
        bits.append(b)
    total = sum(bits)
    if total > QUANT_SPEC_MAX_TOTAL_BITS:
        raise ValueError(
            f"total bits per vector must be ≤ {QUANT_SPEC_MAX_TOTAL_BITS}: got {total}"
        )
    if prefix == "vq-" and len(bits) > 1:
        raise ValueError(
            f"'vq-' is single-stage; use 'rvq-' for {len(bits)} stages: {spec!r}"
        )
    if prefix == "rvq-" and len(bits) < 2:
        raise ValueError(f"'rvq-' requires ≥2 stages; got {len(bits)}: {spec!r}")
    return [1 << b for b in bits]


def quant_spec_from_sizes(sizes: Sequence[int]) -> str:
    parts = [str(_index_bits_for_size(int(k))) for k in sizes]
    prefix = "vq-" if len(parts) == 1 else "rvq-"
    return prefix + "-".join(parts)


def _resolve_quant_stages(
    quant_mode: str | None,
    codebook_sizes: Sequence[int] | None,
    codebook_size: int,
) -> list[int]:
    if codebook_sizes:
        return [int(x) for x in codebook_sizes]
    if quant_mode:
        return parse_quant_spec(quant_mode)
    return [int(codebook_size)]


def _decode_to_vectors_format(
    vectors_template, codebook, indices, backend: str, device: str
):
    if _is_torch_tensor(vectors_template):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch decode requires torch") from exc
        cb = _torch_float32_matrix(codebook, str(vectors_template.device))
        if _is_torch_tensor(indices):
            idx = indices.detach().to(device=cb.device, dtype=torch.long)
        else:
            idx = torch.as_tensor(indices, dtype=torch.long, device=cb.device)
        return cb[idx]
    if _is_numpy_array(vectors_template):
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("NumPy decode requires numpy") from exc
        cb = np.asarray(codebook, dtype=np.float32)
        idx = np.asarray(indices, dtype=np.int64)
        return cb[idx]
    return [list(codebook[int(i)]) for i in indices]


def _vectors_subtract(a, b):
    if _is_torch_tensor(a) or _is_torch_tensor(b):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch subtract requires torch") from exc
        ta = a if _is_torch_tensor(a) else torch.as_tensor(a, dtype=torch.float32)
        tb = (
            b
            if _is_torch_tensor(b)
            else torch.as_tensor(b, dtype=torch.float32, device=ta.device)
        )
        if tb.device != ta.device:
            tb = tb.to(ta.device)
        return ta - tb
    if _is_numpy_array(a) or _is_numpy_array(b):
        import numpy as np

        return np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return [[float(x) - float(y) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def _decode_vectors_to_flat(vectors, codebook, indices, backend: str):
    if backend == "torch":
        centroids = _torch_float32_matrix(codebook, "auto")
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch decode requires torch") from exc
        assigned = (
            indices.detach().to(device=centroids.device, dtype=torch.long)
            if _is_torch_tensor(indices)
            else torch.as_tensor(indices, dtype=torch.long, device=centroids.device)
        )
        return centroids[assigned].reshape(-1).detach().cpu()
    if backend in {"auto", "numpy"} and _is_numpy_array(vectors):
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("NumPy decode requires numpy") from exc
        centroids = np.asarray(codebook, dtype=np.float32)
        assigned = np.asarray(indices, dtype=np.int64)
        return centroids[assigned].reshape(-1)
    if backend == "numpy":
        raise RuntimeError("NumPy backend requires NumPy array tensors")

    decoded = []
    for index in indices:
        decoded.extend(float(v) for v in codebook[int(index)])
    return decoded


def _apply_row_l2_scales(
    flat: Sequence[float], shape: Sequence[int], scales: Sequence[float]
) -> list[float]:
    try:
        import numpy as np
        return _apply_row_l2_scales_numpy(flat, shape, scales).tolist()
    except ImportError:
        pass

    row_count = int(shape[0])
    row_width = _product([int(x) for x in shape[1:]])
    if len(scales) != row_count:
        raise ValueError("row scale count does not match tensor rows")
    out = []
    for row_i in range(row_count):
        scale = float(scales[row_i])
        start = row_i * row_width
        out.extend(float(v) * scale for v in flat[start : start + row_width])
    return out


def _apply_row_l2_scales_numpy(flat, shape: Sequence[int], scales):
    import numpy as np

    row_count = int(shape[0])
    row_width = _product([int(x) for x in shape[1:]])
    values = np.asarray(flat, dtype=np.float32)[: row_count * row_width].reshape(
        row_count, row_width
    )
    row_scales = np.asarray(scales, dtype=np.float32)
    if row_scales.shape[0] != row_count:
        raise ValueError("row scale count does not match tensor rows")
    return (values * row_scales[:, None]).reshape(-1)


def _write_passthrough_tensors(path: Path, tensors: dict) -> None:
    """Save non-quantized tensors (1D norms, sensitivity-skipped 2D) preserving original dtype."""
    has_torch_tensor = any(_is_torch_tensor(t) for t in tensors.values())
    if has_torch_tensor:
        try:
            import torch  # noqa: F401
            from safetensors.torch import save_file as save_torch
            torch_dict = {}
            for name, tensor in tensors.items():
                if _is_torch_tensor(tensor):
                    torch_dict[name] = tensor.detach().cpu().contiguous()
                else:
                    import numpy as np
                    torch_dict[name] = torch.from_numpy(np.asarray(tensor, dtype=np.float32))
            save_torch(torch_dict, str(path))
            return
        except Exception as exc:
            raise RuntimeError(f"failed to write passthrough tensors (torch): {exc}") from exc
    # Pure numpy path
    import numpy as np
    arrays = {name: np.asarray(t) for name, t in tensors.items()}
    try:
        from safetensors.numpy import save_file
        save_file(arrays, str(path))
    except Exception as exc:
        raise RuntimeError(f"failed to write passthrough tensors (numpy): {exc}") from exc


def pack_checkpoint(
    source: Path,
    out_dir: Path,
    group_size: int,
    codebook_size: int = 256,
    iterations: int = 12,
    max_values_per_tensor: int | None = None,
    codebook_mode: str = "per-tensor",
    sample_vectors: int | None = None,
    backend: str = "auto",
    normalization: str = "none",
    device: str = "cpu",
    codebook_sizes: Sequence[int] | None = None,
    family_stages_map: dict[str, Sequence[int]] | None = None,
    outlier_frac: float = 0.0,
    rotation: str = "none",
    rotation_seed: int | None = None,
    awq_activations: dict | None = None,
    awq_alpha: float = 0.5,
    max_tensors: int | None = None,
    progress_file: Path | None = None,
    sensitivity_map: dict | None = None,
    codebook_cache_dir: Path | None = None,
    block_scale_size: int = 32,
) -> dict:
    if codebook_mode not in {"per-tensor", "global", "family"}:
        raise ValueError(
            "codebook_mode must be 'per-tensor', 'global', or 'family'"
        )
    if backend not in {"auto", "numpy", "torch"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'torch'")
    if normalization not in {
        "none",
        "row-l2",
        "col-l2",
        "block-max",
        "awq",
        "awq-block-max",
        "slrq-block",
    }:
        raise ValueError(
            "normalization must be 'none', 'row-l2', 'col-l2', 'block-max', 'awq', or 'awq-block-max'"
        )
    if rotation not in {"none", "orthogonal", "hadamard"}:
        raise ValueError("rotation must be 'none', 'orthogonal', or 'hadamard'")
    if backend == "torch":
        device = _maybe_fallback_cuda_to_cpu(device, backend)
        resolved_device = str(_resolve_torch_device(device))
    else:
        resolved_device = "cpu"

    if rotation == "orthogonal" and rotation_seed is None:
        rotation_seed = int.from_bytes(os.urandom(8), "little")

    # Mixed-Precision Sensitivity Logic
    skipped_tensors = set()
    if sensitivity_map is not None:
        for entry in sensitivity_map.get("layers", []):
            if (
                entry["loss_delta"] > 1.5
                or "embed" in entry["layer"]
                or "lm_head" in entry["layer"]
            ):
                skipped_tensors.add(entry["layer"])

    src_sig = _source_signature(source)

    if family_stages_map is not None:
        if codebook_mode != "per-tensor":
            raise ValueError(
                "family_stages_map (mixed mode) requires codebook_mode='per-tensor'"
            )
        family_stages_resolved = {
            fam: [int(k) for k in stages] for fam, stages in family_stages_map.items()
        }
        stages_spec = []
        n_stages = max(len(s) for s in family_stages_resolved.values())
    else:
        family_stages_resolved = None
        stages_spec = list(codebook_sizes) if codebook_sizes else [codebook_size]
        n_stages = len(stages_spec)
        if n_stages < 1:
            raise ValueError("at least one codebook stage is required")

    tensor_dir = out_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "orka",
        "version": ORKA_VERSION,
        "source": str(source),
        "group_size": group_size,
        "requested_codebook_size": stages_spec[0]
        if (family_stages_resolved is None and n_stages == 1)
        else None,
        "codebook_sizes": list(stages_spec) if family_stages_resolved is None else None,
        "family_stages_map": family_stages_resolved,
        "n_stages": n_stages,
        "codebook_mode": codebook_mode,
        "sample_vectors": sample_vectors,
        "backend": backend,
        "device": resolved_device,
        "normalization": normalization,
        "outlier_frac": outlier_frac,
        "rotation": rotation,
        "rotation_seed": rotation_seed,
        "awq_enabled": awq_activations is not None,
        "tensors": [],
    }

    def _offload(t):
        if _is_torch_tensor(t):
            return t.detach().cpu()
        return t

    def _onload(t, device):
        if _is_torch_tensor(t):
            return t.to(device=device)
        return t

    candidates = []
    awq_fallbacks: list[str] = []
    _passthrough: dict[str, object] = {}

    # Prefetch Queue for Concurrency
    prefetch_queue = queue.Queue(maxsize=4)
    prefetch_done = threading.Event()
    _prefetch_exc: list[BaseException] = []

    def _prefetch_worker():
        try:
            for i, (name, tensor) in enumerate(_load_tensors(source)):
                if max_tensors is not None and prefetch_queue.qsize() + len(candidates) >= max_tensors:
                    break
                shape = _tensor_shape(tensor)
                if len(shape) < 2:
                    _passthrough[name] = tensor
                    continue
                
                # Skipped tensors stay FP16 in the artifact (passthrough), not quantized.
                if name.replace(".weight", "") in skipped_tensors or name in skipped_tensors:
                    _passthrough[name] = tensor
                    continue

                row_scales = None
                source_flat = None
                awq_col_scales = None
                salient_weights = None
                salient_indices = None
                if normalization in {"row-l2", "col-l2", "block-max", "awq", "awq-block-max", "slrq-block"}:
                    (
                        tensor, row_scales, source_flat, awq_col_scales,
                        salient_weights, salient_indices
                    ) = _apply_normalization(
                        tensor, name, normalization, awq_activations, awq_alpha,
                        block_scale_size, backend, resolved_device, awq_fallbacks,
                    )

                tensor_seed = None
                if rotation == "orthogonal":
                    tensor, tensor_seed = _rotate_tensor_to_2d(
                        tensor, name, rotation, rotation_seed, backend, resolved_device
                    )

                if backend == "torch":
                    packed_values, padded_values, vectors = _torch_vectors_from_tensor(
                        tensor, group_size, max_values_per_tensor, resolved_device
                    )
                else:
                    packed_values, padded_values, vectors = _numpy_vectors_from_tensor(
                        tensor, group_size, max_values_per_tensor
                    )
                
                vw = None
                if (awq_activations is not None and name in awq_activations and shape[-1] % group_size == 0):
                    import torch
                    H_diag = torch.as_tensor(awq_activations[name], dtype=torch.float32).pow(2).mean(dim=0)
                    vw = H_diag.reshape(-1, group_size).mean(dim=0).clamp(min=1e-6).tolist()

                prefetch_queue.put({
                    "name": name, "shape": shape, "source_flat": source_flat,
                    "packed_values": packed_values, "padded_values": padded_values,
                    "vectors": vectors, "row_scales": row_scales, "awq_col_scales": awq_col_scales,
                    "salient_weights": salient_weights, "salient_indices": salient_indices,
                    "normalization": normalization, "block_scale_size": block_scale_size if normalization in ("block-max", "awq-block-max", "slrq-block") else None,
                    "family": classify_tensor_family(name), "rotation_seed": tensor_seed,
                    "vector_weights": vw, "stages_data": {},
                })
        except BaseException as exc:
            _prefetch_exc.append(exc)
        finally:
            prefetch_done.set()

    prefetch_thread = threading.Thread(target=_prefetch_worker, daemon=True)
    prefetch_thread.start()

    while not prefetch_done.is_set() or not prefetch_queue.empty():
        if _prefetch_exc:
            break
        try:
            c = prefetch_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        _report_progress(progress_file, f"Prepared {c['name']} {c['shape']} (Ready for Quantization)")
        
        positions, values, new_vectors = _extract_outliers(c["vectors"], outlier_frac, c["packed_values"])
        c["outlier_positions"] = positions
        c["outlier_values"] = values
        c["vectors"] = _offload(new_vectors)
        c["vectors_orig"] = c["vectors"]
        c["vectors_residual"] = c["vectors"]
        c["decoded_sum"] = None
        c["stages_meta"] = []
        candidates.append(c)
        prefetch_queue.task_done()

    prefetch_thread.join()

    if _prefetch_exc:
        raise RuntimeError(f"prefetch worker failed: {_prefetch_exc[0]}") from _prefetch_exc[0]
    if not candidates:
        raise RuntimeError(
            "prefetch worker produced 0 candidates - no quantizable tensors found "
            "(check model path, tensor shapes, and device errors above)"
        )

    if _passthrough:
        passthrough_path = out_dir / "passthrough.safetensors"
        _write_passthrough_tensors(passthrough_path, _passthrough)
        manifest["passthrough_count"] = len(_passthrough)

    total_index_bytes = 0

    for stage_i in range(n_stages):
        _report_progress(
            progress_file, f"--- Starting Stage {stage_i + 1}/{n_stages} ---"
        )
        stage_codebooks = {}
        if (
            family_stages_resolved is None
            and codebook_mode in {"global", "family"}
            and candidates
        ):
            k = stages_spec[stage_i]
            vector_groups = {}
            for c in candidates:
                key = "global" if codebook_mode == "global" else c["family"]
                vector_groups.setdefault(key, []).append(c["vectors_residual"])
            for key, parts in vector_groups.items():
                cache_key = (
                    _codebook_cache_key(
                        [
                            "shared",
                            src_sig,
                            codebook_mode,
                            key,
                            group_size,
                            k,
                            sample_vectors,
                            iterations,
                            backend,
                            normalization,
                            rotation,
                            rotation_seed,
                            outlier_frac,
                            max_tensors,
                            stage_i,
                            "awq-weighted" if awq_activations else "unweighted",
                        ]
                    )
                    if stage_i == 0
                    else None
                )
                cached = (
                    _codebook_cache_load(codebook_cache_dir, cache_key)
                    if cache_key
                    else None
                )
                if cached is not None:
                    cb = cached
                    training_count = (
                        int(sample_vectors)
                        if sample_vectors
                        else sum(len(p) for p in parts)
                    )
                else:
                    if sample_vectors is None or sample_vectors <= 0:
                        sampled_parts = parts
                    else:
                        total_count = sum(len(p) for p in parts)
                        if total_count <= sample_vectors:
                            sampled_parts = parts
                        else:
                            sampled_parts = []
                            remaining_budget = int(sample_vectors)
                            for idx, p in enumerate(parts):
                                share = max(
                                    1, int(round(sample_vectors * len(p) / total_count))
                                )
                                share = min(share, len(p), remaining_budget)
                                if idx == len(parts) - 1:
                                    share = min(remaining_budget, len(p))
                                sampled_parts.append(_sample_vector_rows(p, share))
                                remaining_budget -= share
                                if remaining_budget <= 0:
                                    break
                    training = _concat_vector_parts(sampled_parts)
                    vw = None

                    cb_seed = _derive_seed(
                        ["shared", src_sig, codebook_mode, key, group_size, k, stage_i]
                    )
                    cb, _, _ = learn_codebook_auto(
                        training,
                        min(k, len(training)),
                        iterations,
                        backend,
                        resolved_device,
                        vector_weights=vw,
                        seed=cb_seed,
                    )
                    if cache_key:
                        _codebook_cache_save(codebook_cache_dir, cache_key, cb)
                if n_stages == 1:
                    cb_path = out_dir / "codebooks" / f"{key}.codebook.f32"
                else:
                    cb_path = out_dir / "codebooks" / f"{key}.s{stage_i}.codebook.f32"
                _write_codebook(cb_path, cb)
                stage_codebooks[key] = (cb, cb_path)

        for i, c in enumerate(candidates):
            base_name = c["name"].replace(".weight", "")
            if base_name in skipped_tensors or c["name"] in skipped_tensors:
                continue
            _report_progress(
                progress_file,
                f"Quantizing {c['name']} ({i + 1}/{len(candidates)}) | Stage {stage_i + 1}/{n_stages}",
            )
            safe = _safe_tensor_name(c["name"])
            if backend == "torch":
                c["vectors_orig"] = _onload(c["vectors_orig"], resolved_device)
                c["vectors_residual"] = _onload(c["vectors_residual"], resolved_device)
                if c["decoded_sum"] is not None:
                    c["decoded_sum"] = _onload(c["decoded_sum"], resolved_device)
            if family_stages_resolved is not None:
                stages_for_c = family_stages_resolved[c["family"]]
                if stage_i >= len(stages_for_c):
                    continue
                k = stages_for_c[stage_i]
                training = _sample_vector_rows(c["vectors_residual"], sample_vectors)
                cb_seed = _derive_seed(
                    ["family-mixed", src_sig, c["name"], group_size, k, stage_i]
                )
                cb, _, _ = learn_codebook_auto(
                    training,
                    min(k, len(training)),
                    iterations,
                    backend,
                    resolved_device,
                    seed=cb_seed,
                )
                training_count = len(training)
                cb_path = tensor_dir / f"{safe}.s{stage_i}.codebook.f32"
                _write_codebook(cb_path, cb)
            elif codebook_mode in {"global", "family"}:
                k = stages_spec[stage_i]
                key = "global" if codebook_mode == "global" else c["family"]
                cb, cb_path = stage_codebooks[key]
                training_count = sample_vectors or len(c["vectors_residual"])
            else:
                k = stages_spec[stage_i]
                cache_key = (
                    _codebook_cache_key(
                        [
                            "per-tensor",
                            src_sig,
                            c["name"],
                            group_size,
                            k,
                            sample_vectors,
                            iterations,
                            backend,
                            normalization,
                            rotation,
                            rotation_seed,
                            outlier_frac,
                            max_tensors,
                            stage_i,
                            "awq-weighted" if awq_activations else "unweighted",
                        ]
                    )
                    if stage_i == 0
                    else None
                )
                cached = (
                    _codebook_cache_load(codebook_cache_dir, cache_key)
                    if cache_key
                    else None
                )
                if cached is not None:
                    cb = cached
                    training_count = sample_vectors or len(c["vectors_residual"])
                else:
                    training = _sample_vector_rows(
                        c["vectors_residual"], sample_vectors
                    )
                    vw = c.get("vector_weights")

                    cb_seed = _derive_seed(
                        ["per-tensor", src_sig, c["name"], group_size, k, stage_i]
                    )
                    cb, _, _ = learn_codebook_auto(
                        training,
                        min(k, len(training)),
                        iterations,
                        backend,
                        resolved_device,
                        vector_weights=vw,
                        seed=cb_seed,
                    )
                    training_count = len(training)
                    if cache_key:
                        _codebook_cache_save(codebook_cache_dir, cache_key, cb)
                if n_stages == 1:
                    cb_path = tensor_dir / f"{safe}.codebook.f32"
                else:
                    cb_path = tensor_dir / f"{safe}.s{stage_i}.codebook.f32"
                _write_codebook(cb_path, cb)

            indices, _ = quantize_vectors_auto(
                c["vectors_residual"], cb, backend, resolved_device
            )
            
            # Cache for joint refinement
            c["stages_data"][stage_i] = {
                "cb": cb,
                "indices": indices
            }
            index_bits = _index_bits_for_size(len(cb))
            if n_stages == 1:
                idx_path = tensor_dir / f"{safe}.indices"
            else:
                idx_path = tensor_dir / f"{safe}.s{stage_i}.indices"
            _write_indices(idx_path, indices, index_bits)
            stage_bytes = idx_path.stat().st_size
            total_index_bytes += stage_bytes

            decoded = _decode_to_vectors_format(
                c["vectors_orig"], cb, indices, backend, resolved_device
            )
            if c["decoded_sum"] is None:
                c["decoded_sum"] = decoded
            else:
                if _is_torch_tensor(c["decoded_sum"]):
                    c["decoded_sum"] = c["decoded_sum"] + decoded
                elif _is_numpy_array(c["decoded_sum"]):
                    c["decoded_sum"] = c["decoded_sum"] + decoded
                else:
                    c["decoded_sum"] = [
                        [a + b for a, b in zip(ra, rb)]
                        for ra, rb in zip(c["decoded_sum"], decoded)
                    ]
            c["vectors_residual"] = _vectors_subtract(
                c["vectors_orig"], c["decoded_sum"]
            )

            if backend == "torch":
                c["vectors_residual"] = _offload(c["vectors_residual"])
                c["decoded_sum"] = _offload(c["decoded_sum"])
                c["vectors_orig"] = _offload(c["vectors_orig"])
                try:
                    import torch as _t

                    _t.cuda.empty_cache()
                except Exception:
                    pass

            c["stages_meta"].append(
                {
                    "stage": stage_i,
                    "codebook": str(cb_path.relative_to(out_dir)),
                    "codebook_size": len(cb),
                    "index_bits": index_bits,
                    "indices": str(idx_path.relative_to(out_dir)),
                    "index_bytes": stage_bytes,
                    "training_vector_count": training_count,
                    "codebook_family": c["family"],
                }
            )

    # Joint-Optimized Additive Quantization (AQLM EM-style)
    # We run the greedy stage-by-stage first, then do a joint refinement pass.
    if n_stages > 1 and codebook_mode == "per-tensor":
        _report_progress(progress_file, "--- Starting Joint Optimization (EM-AQ) ---")
        joint_iterations = 3
        
        # Pre-calculate full sum once
        for i, c in enumerate(candidates):
            base_name = c["name"].replace(".weight", "")
            if base_name in skipped_tensors or c["name"] in skipped_tensors:
                continue
            
            full_sum = None
            for stage_i in range(n_stages):
                sd = c["stages_data"][stage_i]
                dec = _decode_to_vectors_format(
                    c["vectors_orig"], sd["cb"], sd["indices"], backend, resolved_device
                )
                full_sum = dec if full_sum is None else full_sum + dec
            c["current_full_sum"] = full_sum

        for joint_iter in range(joint_iterations):
            _report_progress(
                progress_file,
                f"Joint Refinement Pass {joint_iter + 1}/{joint_iterations}",
            )
            for stage_i in range(n_stages):
                _report_progress(progress_file, f"    Refining stage {stage_i + 1}/{n_stages}...")
                for i, c in enumerate(candidates):
                    base_name = c["name"].replace(".weight", "")
                    if base_name in skipped_tensors or c["name"] in skipped_tensors:
                        continue

                    k = c["stages_meta"][stage_i]["codebook_size"]

                    # When k >= sample_vectors the codebook already memorizes the sample;
                    # joint refinement gives near-zero quality gain but dominates runtime.
                    if k >= sample_vectors:
                        continue

                    # O(1) Residual Update: target = orig - (full_sum - current_stage_dec)
                    sd = c["stages_data"][stage_i]
                    old_dec = _decode_to_vectors_format(
                        c["vectors_orig"], sd["cb"], sd["indices"], backend, resolved_device
                    )
                    
                    target = _vectors_subtract(c["vectors_orig"], (c["current_full_sum"] - old_dec))
                    training = _sample_vector_rows(target, sample_vectors)
                    vw = c.get("vector_weights")

                    cb, _, _ = learn_codebook_auto(
                        training, min(k, len(training)), 2, backend, resolved_device,
                        vector_weights=vw, initial_codebook=sd["cb"],
                    )

                    indices, _ = quantize_vectors_auto(target, cb, backend, resolved_device)
                    
                    # Update full sum: subtract old dec, add new dec
                    new_dec = _decode_to_vectors_format(c["vectors_orig"], cb, indices, backend, resolved_device)
                    c["current_full_sum"] = (c["current_full_sum"] - old_dec) + new_dec
                    
                    # Update cache
                    c["stages_data"][stage_i] = {"cb": cb, "indices": indices}

                    safe = _safe_tensor_name(c["name"])
                    cb_path = tensor_dir / f"{safe}.s{stage_i}.codebook.f32"
                    _BG_WRITER.submit(_write_codebook, cb_path, cb)

                    idx_path = tensor_dir / f"{safe}.s{stage_i}.indices"
                    _BG_WRITER.submit(_write_indices, idx_path, indices.cpu() if hasattr(indices, "cpu") else indices, c["stages_meta"][stage_i]["index_bits"])
                    c["decoded_sum"] = None

        # Sync decoded_sum after all passes
        for c in candidates:
            if "current_full_sum" in c:
                c["decoded_sum"] = _offload(c["current_full_sum"])
                del c["current_full_sum"]
            
            # Update metrics in manifest with joint-refined values
            refined_metrics = _stage_quality_metrics(c, backend)
            c["refined_metrics"] = refined_metrics

    _BG_WRITER.wait()

    _report_progress(progress_file, "--- Writing packed tensors & generating manifest ---")
    for i, c in enumerate(candidates):
        base_name = c["name"].replace(".weight", "")
        if base_name in skipped_tensors or c["name"] in skipped_tensors:
            continue
        _report_progress(progress_file, f"  Writing {c['name']} ({i+1}/{len(candidates)})...")
        safe = _safe_tensor_name(c["name"])
        scale_path = None
        scale_bytes = 0
        scale_count = 0
        if c["normalization"] == "row-l2":
            scale_path = tensor_dir / f"{safe}.row_l2_scale.f32"
            _write_f32_vector(scale_path, c["row_scales"])
            scale_bytes = scale_path.stat().st_size
            scale_count = len(c["row_scales"])
        elif c["normalization"] in ("col-l2", "awq"):
            scale_path = tensor_dir / f"{safe}.col_l2_scale.f32"
            _write_f32_vector(scale_path, c["row_scales"])
            scale_bytes = scale_path.stat().st_size
            scale_count = len(c["row_scales"])
        elif c["normalization"] in ("block-max", "slrq-block"):
            scale_path = tensor_dir / f"{safe}.block_max_scale.f32"
            _write_f32_vector(scale_path, c["row_scales"])
            scale_bytes = scale_path.stat().st_size
            scale_count = len(c["row_scales"])
        elif c["normalization"] == "awq-block-max":
            scale_path = tensor_dir / f"{safe}.block_max_scale.f32"
            _write_f32_vector(scale_path, c["row_scales"])
            scale_bytes = scale_path.stat().st_size
            scale_count = len(c["row_scales"])

        awq_col_meta = None
        if (
            c["normalization"] == "awq-block-max"
            and c.get("awq_col_scales") is not None
        ):
            awq_col_path = tensor_dir / f"{safe}.awq_col_scale.f32"
            _write_f32_vector(awq_col_path, c["awq_col_scales"])
            awq_col_meta = {
                "path": str(awq_col_path.relative_to(out_dir)),
                "count": len(c["awq_col_scales"]),
                "bytes": awq_col_path.stat().st_size,
            }

        outlier_meta = None
        if c.get("outlier_positions") is not None and len(c["outlier_positions"]) > 0:
            out_idx_path = tensor_dir / f"{safe}.outliers.idx"
            out_val_path = tensor_dir / f"{safe}.outliers.val"
            _write_outliers(
                out_idx_path, out_val_path, c["outlier_positions"], c["outlier_values"]
            )
            outlier_meta = {
                "count": int(len(c["outlier_positions"])),
                "positions": str(out_idx_path.relative_to(out_dir)),
                "values": str(out_val_path.relative_to(out_dir)),
                "positions_bytes": out_idx_path.stat().st_size,
                "values_bytes": out_val_path.stat().st_size,
            }

        salient_meta = None
        if c.get("salient_indices") is not None:
            s_idx_path = tensor_dir / f"{safe}.salient.idx"
            s_val_path = tensor_dir / f"{safe}.salient.val"
            
            sw = c["salient_weights"].numpy() if hasattr(c["salient_weights"], "numpy") else c["salient_weights"]
            si = c["salient_indices"].numpy() if hasattr(c["salient_indices"], "numpy") else c["salient_indices"]
            
            sw.astype("<f4").tofile(str(s_val_path))
            si.astype("<u4").tofile(str(s_idx_path))
            
            salient_meta = {
                "count": int(len(sw)),
                "indices": str(s_idx_path.relative_to(out_dir)),
                "weights": str(s_val_path.relative_to(out_dir)),
                "indices_bytes": s_idx_path.stat().st_size,
                "weights_bytes": s_val_path.stat().st_size,
            }

        metrics = c.get("refined_metrics") or _stage_quality_metrics(c, backend)
        first = c["stages_meta"][0]
        last_idx_path = tensor_dir / (
            f"{safe}.indices" if n_stages == 1 else f"{safe}.s0.indices"
        )
        index_bytes_total = sum(s["index_bytes"] for s in c["stages_meta"])
        manifest["tensors"].append(
            {
                "name": c["name"],
                "shape": c["shape"],
                "packed_values": c["packed_values"],
                "padded_values": c["padded_values"],
                "vector_count": len(c["vectors_orig"]),
                "training_vector_count": first["training_vector_count"],
                "group_size": group_size,
                "codebook_size": first["codebook_size"],
                "index_bits": first["index_bits"],
                "index_bytes": index_bytes_total,
                "n_stages": n_stages,
                "stages": c["stages_meta"],
                "total_bits_per_vector": sum(s["index_bits"] for s in c["stages_meta"]),
                "mse": metrics["mse"],
                "sse": metrics["sse"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "max_abs_error": metrics["max_abs_error"],
                "source_l2_sq": metrics["source_l2_sq"],
                "reconstructed_l2_sq": metrics["reconstructed_l2_sq"],
                "dot": metrics["dot"],
                "relative_rmse": metrics["relative_rmse"],
                "cosine_similarity": metrics["cosine_similarity"],
                "indices": str(last_idx_path.relative_to(out_dir)),
                "codebook": c["stages_meta"][0]["codebook"],
                "codebook_family": c["family"],
                "normalization": c["normalization"],
                "scales": str(scale_path.relative_to(out_dir)) if scale_path else None,
                "scale_count": scale_count,
                "scale_bytes": scale_bytes,
                "block_scale_size": block_scale_size
                if c["normalization"] in ("block-max", "awq-block-max", "slrq-block")
                else None,
                "awq_col_scales": awq_col_meta,
                "outliers": outlier_meta,
                "salient": salient_meta,
                "rotation_seed": c.get("rotation_seed"),
                "rotation": rotation if c.get("rotation_seed") is not None else "none",
            }
        )

    manifest["total_index_bytes"] = total_index_bytes
    manifest["tensor_count"] = len(manifest["tensors"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    return manifest


def _stage_quality_metrics(candidate: dict, backend: str) -> dict:
    decoded_sum = candidate["decoded_sum"]
    orig = candidate["vectors_orig"]
    if _is_torch_tensor(decoded_sum):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch metrics requires torch") from exc
        diff = orig - decoded_sum
        sse = float((diff * diff).sum().detach().cpu().item())
        abs_diff = diff.abs()
        abs_error_sum = float(abs_diff.sum().detach().cpu().item())
        max_abs_error = (
            float(abs_diff.max().detach().cpu().item()) if abs_diff.numel() else 0.0
        )
        source_l2_sq = float((orig * orig).sum().detach().cpu().item())
        reconstructed_l2_sq = float(
            (decoded_sum * decoded_sum).sum().detach().cpu().item()
        )
        dot = float((orig * decoded_sum).sum().detach().cpu().item())
        value_count = int(orig.numel())
    elif _is_numpy_array(decoded_sum):
        import numpy as np

        diff = orig - decoded_sum
        abs_diff = np.abs(diff)
        sse = float(np.sum(diff * diff))
        abs_error_sum = float(np.sum(abs_diff))
        max_abs_error = float(np.max(abs_diff)) if abs_diff.size else 0.0
        source_l2_sq = float(np.sum(orig * orig))
        reconstructed_l2_sq = float(np.sum(decoded_sum * decoded_sum))
        dot = float(np.sum(orig * decoded_sum))
        value_count = int(orig.size)
    else:
        flat_src = []
        flat_rec = []
        for ro, rd in zip(orig, decoded_sum):
            flat_src.extend(float(v) for v in ro)
            flat_rec.extend(float(v) for v in rd)
        return _denorm_metrics_from_flat(candidate, flat_src, flat_rec)

    metrics = _quality_from_totals(
        value_count=value_count,
        sse=sse,
        abs_error_sum=abs_error_sum,
        max_abs_error=max_abs_error,
        source_l2_sq=source_l2_sq,
        reconstructed_l2_sq=reconstructed_l2_sq,
        dot=dot,
    )
    norm = candidate.get("normalization", "none")
    rot_seed = candidate.get("rotation_seed")
    has_rotation = rot_seed is not None
    outlier_positions = candidate.get("outlier_positions")
    outlier_values = candidate.get("outlier_values")
    has_outliers = outlier_positions is not None and len(outlier_positions) > 0
    if norm == "none" and not has_rotation and not has_outliers:
        return metrics

    import numpy as np
    if _is_torch_tensor(decoded_sum):
        flat_decoded = (
            decoded_sum.reshape(-1).detach().cpu().numpy()[: candidate["packed_values"]]
        ).copy()
    elif _is_numpy_array(decoded_sum):
        flat_decoded = decoded_sum.reshape(-1)[: candidate["packed_values"]].copy()
    else:
        flat_decoded = np.asarray(
            list(
                decoded_sum.reshape(-1)
                if hasattr(decoded_sum, "reshape")
                else [v for row in decoded_sum for v in row]
            )[: candidate["packed_values"]],
            dtype=np.float32,
        )
    # Patch outliers (stored in normalized+rotated space, written by _decode_tensor before un-rotate).
    if has_outliers:
        positions = np.asarray(list(outlier_positions), dtype=np.int64)
        values = np.asarray(list(outlier_values), dtype=np.float32)
        flat_decoded[positions] = values
    # Un-rotate first (matching decode order: un-rotate → un-normalize).
    if has_rotation:
        flat_decoded = np.asarray(
            _unrotate_flat(flat_decoded.tolist(), candidate["shape"], "orthogonal", int(rot_seed)),
            dtype=np.float32,
        )
    if norm == "none":
        return _quality_metrics_for_numpy_flat(candidate["source_flat"], flat_decoded)
    return _denorm_metrics_from_flat(candidate, candidate["source_flat"], flat_decoded)


def _denorm_metrics_from_flat(candidate: dict, source_flat, decoded_flat) -> dict:
    norm = candidate.get("normalization", "none")
    if norm == "none":
        return (
            _quality_metrics_for_numpy_flat(source_flat, decoded_flat)
            if _is_numpy_array(decoded_flat)
            else quality_metrics_from_flat(source_flat, decoded_flat)
        )
    if norm == "row-l2":
        if _is_numpy_array(decoded_flat):
            denorm = _apply_row_l2_scales_numpy(
                decoded_flat, candidate["shape"], candidate["row_scales"]
            )
            return _quality_metrics_for_numpy_flat(candidate["source_flat"], denorm)
        denorm = _apply_row_l2_scales(
            decoded_flat, candidate["shape"], candidate["row_scales"]
        )
        return quality_metrics_from_flat(candidate["source_flat"], denorm)
    if norm in ("col-l2", "awq"):
        if _is_numpy_array(decoded_flat):
            denorm = _apply_col_l2_scales_numpy(
                decoded_flat, candidate["shape"], candidate["row_scales"]
            )
            return _quality_metrics_for_numpy_flat(candidate["source_flat"], denorm)
        denorm = _apply_col_l2_scales(
            decoded_flat, candidate["shape"], candidate["row_scales"]
        )
        return quality_metrics_from_flat(candidate["source_flat"], denorm)
    if norm in ("block-max", "slrq-block"):
        block_size = candidate.get("block_scale_size") or 32
        if _is_numpy_array(decoded_flat):
            denorm = _apply_block_max_scales_numpy(
                decoded_flat, candidate["row_scales"], block_size
            )
            return _quality_metrics_for_numpy_flat(candidate["source_flat"], denorm)
        denorm = _apply_block_max_scales(
            decoded_flat, candidate["row_scales"], block_size
        )
        return quality_metrics_from_flat(candidate["source_flat"], denorm)
    if norm == "awq-block-max":
        block_size = candidate.get("block_scale_size") or 32
        awq_scales = candidate.get("awq_col_scales")
        block_scales = candidate["row_scales"]
        if _is_numpy_array(decoded_flat):
            stage = _apply_block_max_scales_numpy(
                decoded_flat, block_scales, block_size
            )
            if awq_scales is not None:
                stage = _apply_col_l2_scales_numpy(
                    stage, candidate["shape"], awq_scales
                )
            return _quality_metrics_for_numpy_flat(candidate["source_flat"], stage)
        stage = _apply_block_max_scales(decoded_flat, block_scales, block_size)
        if awq_scales is not None:
            stage = _apply_col_l2_scales(stage, candidate["shape"], awq_scales)
        return quality_metrics_from_flat(candidate["source_flat"], stage)
    raise ValueError(f"unknown normalization: {norm}")


def _product(values: Sequence[int]) -> int:
    total = 1
    for value in values:
        total *= value
    return total


def _reshape_flat(values: Sequence[float], shape: Sequence[int]) -> object:
    if not shape:
        if len(values) != 1:
            raise ValueError("scalar reshape requires exactly one value")
        return float(values[0])
    if len(shape) == 1:
        width = int(shape[0])
        return [float(v) for v in values[:width]]

    step = _product(shape[1:])
    return [
        _reshape_flat(values[i : i + step], shape[1:])
        for i in range(0, int(shape[0]) * step, step)
    ]


def _dir_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def report_artifact(out_dir: Path) -> dict:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Orka manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    tensors = manifest.get("tensors", [])
    total_index_bytes = sum(int(t.get("index_bytes", 0)) for t in tensors)
    total_codebook_bytes = 0
    total_scale_bytes = sum(int(t.get("scale_bytes", 0)) for t in tensors)
    total_outlier_bytes = sum(
        int((t.get("outliers") or {}).get("positions_bytes", 0))
        + int((t.get("outliers") or {}).get("values_bytes", 0))
        for t in tensors
    )
    total_outlier_count = sum(
        int((t.get("outliers") or {}).get("count", 0)) for t in tensors
    )
    original_fp16_bytes = 0
    weighted_error = 0.0
    weighted_values = 0
    sse = 0.0
    abs_error_sum = 0.0
    max_abs_error = 0.0
    source_l2_sq = 0.0
    reconstructed_l2_sq = 0.0
    dot = 0.0
    counted_codebooks = set()

    for tensor in tensors:
        cb_paths = [out_dir / s["codebook"] for s in tensor.get("stages", [])] or [
            out_dir / tensor["codebook"]
        ]
        for codebook_path in cb_paths:
            if codebook_path.exists() and codebook_path not in counted_codebooks:
                total_codebook_bytes += codebook_path.stat().st_size
                counted_codebooks.add(codebook_path)
        shape = [int(x) for x in tensor.get("shape", [])]
        if shape:
            original_fp16_bytes += _product(shape) * 2
        value_count = int(tensor.get("packed_values", 0))
        weighted_error += float(tensor.get("mse", 0.0)) * value_count
        weighted_values += value_count
        sse += float(tensor.get("sse", float(tensor.get("mse", 0.0)) * value_count))
        abs_error_sum += float(tensor.get("mae", 0.0)) * value_count
        max_abs_error = max(max_abs_error, float(tensor.get("max_abs_error", 0.0)))
        source_l2_sq += float(tensor.get("source_l2_sq", 0.0))
        reconstructed_l2_sq += float(tensor.get("reconstructed_l2_sq", 0.0))
        dot += float(tensor.get("dot", 0.0))

    artifact_bytes = _dir_size(out_dir)
    worst_tensors = sorted(
        (
            {
                "name": tensor.get("name"),
                "shape": tensor.get("shape"),
                "mse": tensor.get("mse", 0.0),
                "relative_rmse": tensor.get("relative_rmse"),
                "cosine_similarity": tensor.get("cosine_similarity"),
                "index_bytes": tensor.get("index_bytes", 0),
            }
            for tensor in tensors
        ),
        key=lambda item: float(item["mse"]),
        reverse=True,
    )[:10]

    compression_ratio = (
        original_fp16_bytes / artifact_bytes if artifact_bytes > 0 else 0.0
    )
    aggregate_metrics = _quality_from_totals(
        value_count=weighted_values,
        sse=sse,
        abs_error_sum=abs_error_sum,
        max_abs_error=max_abs_error,
        source_l2_sq=source_l2_sq,
        reconstructed_l2_sq=reconstructed_l2_sq,
        dot=dot,
    )
    return {
        "format": manifest.get("format"),
        "version": manifest.get("version"),
        "source": manifest.get("source"),
        "tensor_count": len(tensors),
        "group_size": manifest.get("group_size"),
        "requested_codebook_size": manifest.get("requested_codebook_size"),
        "codebook_mode": manifest.get("codebook_mode", "per-tensor"),
        "normalization": manifest.get("normalization", "none"),
        "total_index_bytes": total_index_bytes,
        "total_codebook_bytes": total_codebook_bytes,
        "total_scale_bytes": total_scale_bytes,
        "total_outlier_bytes": total_outlier_bytes,
        "total_outlier_count": total_outlier_count,
        "artifact_bytes": artifact_bytes,
        "original_fp16_bytes": original_fp16_bytes,
        "compression_ratio_fp16_to_artifact": compression_ratio,
        "weighted_mse": weighted_error / weighted_values if weighted_values else 0.0,
        "rmse": aggregate_metrics["rmse"],
        "mae": aggregate_metrics["mae"],
        "max_abs_error": aggregate_metrics["max_abs_error"],
        "relative_rmse": aggregate_metrics["relative_rmse"],
        "cosine_similarity": aggregate_metrics["cosine_similarity"],
        "source_norm": math.sqrt(source_l2_sq),
        "reconstructed_norm": math.sqrt(reconstructed_l2_sq),
        "worst_tensors": worst_tensors,
    }


def _decode_tensor(out_dir: Path, tensor_meta: dict) -> list[float]:
    group_size = int(tensor_meta["group_size"])
    padded_values = int(tensor_meta["padded_values"])
    index_count = math.ceil(padded_values / group_size)
    stages = tensor_meta.get("stages")
    if not stages:
        stages = [
            {
                "codebook": tensor_meta["codebook"],
                "codebook_size": int(tensor_meta["codebook_size"]),
                "index_bits": int(tensor_meta["index_bits"]),
                "indices": tensor_meta["indices"],
            }
        ]

    try:
        import numpy as np
        use_numpy = True
    except ImportError:
        use_numpy = False

    if use_numpy:
        decoded_np = np.zeros(index_count * group_size, dtype=np.float32)
        for stage in stages:
            cb = np.fromfile(str(out_dir / stage["codebook"]), dtype="<f4").reshape(-1, group_size)
            idxs = np.asarray(_read_indices(out_dir / stage["indices"], int(stage["index_bits"]), index_count), dtype=np.int64)
            decoded_np += cb[idxs].reshape(-1)
        decoded = decoded_np[: int(tensor_meta["packed_values"])].tolist()
    else:
        decoded = [0.0] * (index_count * group_size)
        for stage in stages:
            cb = _read_codebook(
                out_dir / stage["codebook"], int(stage["codebook_size"]), group_size
            )
            idxs = _read_indices(
                out_dir / stage["indices"], int(stage["index_bits"]), index_count
            )
            offset = 0
            for index in idxs:
                row = cb[index]
                for j in range(group_size):
                    decoded[offset + j] += row[j]
                offset += group_size
        decoded = decoded[: int(tensor_meta["packed_values"])]
    outl = tensor_meta.get("outliers")
    if outl:
        positions, values = _read_outliers(
            out_dir / outl["positions"], out_dir / outl["values"]
        )
        for pos, val in zip(positions, values):
            decoded[int(pos)] = float(val)
            
    rotation = tensor_meta.get("rotation", "none")
    if rotation in {"orthogonal", "hadamard"}:
        seed = int(tensor_meta.get("rotation_seed") or 0)
        decoded = _unrotate_flat(decoded, tensor_meta["shape"], rotation, seed)
    norm = tensor_meta.get("normalization", "none")
    if norm == "row-l2":
        scales = _read_f32_vector(
            out_dir / tensor_meta["scales"], int(tensor_meta["scale_count"])
        )
        decoded = _apply_row_l2_scales(decoded, tensor_meta["shape"], scales)
    elif norm in ("col-l2", "awq"):
        scales = _read_f32_vector(
            out_dir / tensor_meta["scales"], int(tensor_meta["scale_count"])
        )
        decoded = _apply_col_l2_scales(decoded, tensor_meta["shape"], scales)
    elif norm in ("block-max", "slrq-block"):
        scales = _read_f32_vector(
            out_dir / tensor_meta["scales"], int(tensor_meta["scale_count"])
        )
        block_size = int(tensor_meta.get("block_scale_size") or 32)
        decoded = _apply_block_max_scales(decoded, scales, block_size)

    elif norm == "awq-block-max":
        block_scales = _read_f32_vector(
            out_dir / tensor_meta["scales"], int(tensor_meta["scale_count"])
        )
        block_size = int(tensor_meta.get("block_scale_size") or 32)
        decoded = _apply_block_max_scales(decoded, block_scales, block_size)
        awq_meta = tensor_meta.get("awq_col_scales")
        if awq_meta:
            awq_scales = _read_f32_vector(
                out_dir / awq_meta["path"], int(awq_meta["count"])
            )
            decoded = _apply_col_l2_scales(decoded, tensor_meta["shape"], awq_scales)

    salient = tensor_meta.get("salient")
    if salient:
        s_idx = np.fromfile(str(out_dir / salient["indices"]), dtype="<u4")
        s_val = np.fromfile(str(out_dir / salient["weights"]), dtype="<f4")
        
        # SLRQ: re-inject salient weights AFTER scaling to avoid double-scaling.
        for b_idx, (local_idx, weight) in enumerate(zip(s_idx, s_val)):
            flat_idx = b_idx * int(tensor_meta.get("block_scale_size", 16)) + int(local_idx)
            if flat_idx < len(decoded):
                decoded[flat_idx] = float(weight)

    return decoded


def verify_artifact(out_dir: Path) -> dict:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Orka manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest["source"])
    source_tensors = {name: tensor for name, tensor in _load_tensors(source)}
    verified = 0
    weighted_error = 0.0
    weighted_values = 0
    sse = 0.0
    abs_error_sum = 0.0
    max_abs_error = 0.0
    source_l2_sq = 0.0
    reconstructed_l2_sq = 0.0
    dot = 0.0
    max_mse_delta = 0.0
    worst_tensors = []

    tensors_list = manifest.get("tensors", [])
    for i, tensor_meta in enumerate(tensors_list):
        name = tensor_meta["name"]
        print(f"  Validating tensor {name} ({i+1}/{len(tensors_list)})...", flush=True)
        if name not in source_tensors:
            raise KeyError(f"source tensor missing during verification: {name}")
        original = _flatten_float_values(
            source_tensors[name], int(tensor_meta["packed_values"])
        )
        decoded = _decode_tensor(out_dir, tensor_meta)
        if len(original) != len(decoded):
            raise ValueError(f"decoded value count mismatch for {name}")

        metrics = quality_metrics_from_flat(original, decoded)
        mse = metrics["mse"]
        mse_delta = abs(mse - float(tensor_meta.get("mse", 0.0)))
        max_mse_delta = max(max_mse_delta, mse_delta)
        weighted_error += mse * len(original)
        weighted_values += len(original)
        sse += metrics["sse"]
        abs_error_sum += metrics["mae"] * metrics["value_count"]
        max_abs_error = max(max_abs_error, metrics["max_abs_error"])
        source_l2_sq += metrics["source_l2_sq"]
        reconstructed_l2_sq += metrics["reconstructed_l2_sq"]
        dot += metrics["dot"]
        verified += 1
        worst_tensors.append(
            {
                "name": name,
                "mse": mse,
                "relative_rmse": metrics["relative_rmse"],
                "cosine_similarity": metrics["cosine_similarity"],
                "manifest_mse": tensor_meta.get("mse", 0.0),
                "mse_delta": mse_delta,
            }
        )

    worst_tensors.sort(key=lambda item: item["mse"], reverse=True)
    aggregate_metrics = _quality_from_totals(
        value_count=weighted_values,
        sse=sse,
        abs_error_sum=abs_error_sum,
        max_abs_error=max_abs_error,
        source_l2_sq=source_l2_sq,
        reconstructed_l2_sq=reconstructed_l2_sq,
        dot=dot,
    )
    return {
        "artifact": str(out_dir),
        "source": str(source),
        "verified_tensors": verified,
        "weighted_mse": weighted_error / weighted_values if weighted_values else 0.0,
        "rmse": aggregate_metrics["rmse"],
        "mae": aggregate_metrics["mae"],
        "max_abs_error": aggregate_metrics["max_abs_error"],
        "relative_rmse": aggregate_metrics["relative_rmse"],
        "cosine_similarity": aggregate_metrics["cosine_similarity"],
        "max_mse_delta": max_mse_delta,
        "worst_tensors": worst_tensors[:10],
    }


def _decoded_tensor_map(out_dir: Path, manifest: dict) -> dict:
    tensors = {}
    for tensor_meta in manifest.get("tensors", []):
        decoded = _decode_tensor(out_dir, tensor_meta)
        shape = [int(x) for x in tensor_meta.get("shape", [])]
        tensors[tensor_meta["name"]] = {
            "shape": shape,
            "flat": decoded,
            "values": _reshape_flat(decoded, shape),
        }
    return tensors


def _complete_decoded_tensor_map(out_dir: Path, manifest: dict) -> dict:
    tensors = {}
    packed_names = {t["name"] for t in manifest.get("tensors", [])}

    # Load passthrough tensors from artifact (self-contained, no source needed).
    passthrough_path = out_dir / "passthrough.safetensors"
    if passthrough_path.exists():
        for name, tensor in _load_tensors(passthrough_path):
            shape = _tensor_shape(tensor)
            flat = _flatten_float_values(tensor)
            tensors[name] = {"shape": shape, "flat": flat, "values": _reshape_flat(flat, shape)}

    # Fall back to source for anything still missing (backward compat, sensitivity-map skips).
    source = Path(manifest["source"])
    if source.exists():
        for name, tensor in _load_tensors(source):
            if name in packed_names or name in tensors:
                continue
            shape = _tensor_shape(tensor)
            flat = _flatten_float_values(tensor)
            tensors[name] = {"shape": shape, "flat": flat, "values": _reshape_flat(flat, shape)}

    tensors.update(_decoded_tensor_map(out_dir, manifest))
    return tensors


def _write_json_reconstruction(
    out_dir: Path, output_path: Path, manifest: dict, tensors: dict
) -> None:
    output = {
        "format": "orka-reconstruction",
        "version": ORKA_VERSION,
        "source_artifact": str(out_dir),
        "source_checkpoint": manifest.get("source"),
        "tensor_count": len(tensors),
        "tensors": {
            name: {
                "shape": tensor["shape"],
                "values": tensor["values"],
            }
            for name, tensor in tensors.items()
        },
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n")


def _write_safetensors_reconstruction(output_path: Path, tensors: dict) -> None:
    try:
        import numpy as np
        from safetensors.numpy import save_file
    except Exception as exc:
        raise RuntimeError(
            "safetensors reconstruction requires numpy and safetensors"
        ) from exc

    arrays = {}
    for name, tensor in tensors.items():
        arrays[name] = np.asarray(tensor["flat"], dtype=np.float32).reshape(
            tensor["shape"]
        )
    save_file(arrays, str(output_path))


def _decode_tensor_torch(out_dir: Path, tm: dict, device: str):
    """Decode a single quantized tensor on GPU, return torch tensor in original shape."""
    import torch
    import numpy as np

    group_size = int(tm["group_size"])
    padded_values = int(tm["padded_values"])
    packed_values = int(tm["packed_values"])
    index_count = math.ceil(padded_values / group_size)
    shape = [int(x) for x in tm["shape"]]

    stages = tm.get("stages") or [{
        "codebook": tm["codebook"],
        "codebook_size": int(tm["codebook_size"]),
        "index_bits": int(tm["index_bits"]),
        "indices": tm["indices"],
    }]

    decoded = torch.zeros(index_count * group_size, dtype=torch.float32, device=device)
    for stage in stages:
        cb_np = np.fromfile(str(out_dir / stage["codebook"]), dtype="<f4").reshape(-1, group_size)
        idxs_np = np.frombuffer(
            (out_dir / stage["indices"]).read_bytes(),
            dtype=_index_bit_spec(int(stage["index_bits"]))[1],
        ).astype(np.int64)
        cb = torch.from_numpy(cb_np).to(device)
        idxs = torch.from_numpy(idxs_np).to(device)
        decoded.add_(cb[idxs].reshape(-1))
    decoded = decoded[:packed_values]

    outl = tm.get("outliers")
    if outl:
        positions, values = _read_outliers(out_dir / outl["positions"], out_dir / outl["values"])
        if positions:
            pos_t = torch.tensor(list(positions), dtype=torch.long, device=device)
            val_t = torch.tensor(list(values), dtype=torch.float32, device=device)
            decoded[pos_t] = val_t

    rotation = tm.get("rotation", "none")
    if rotation in {"orthogonal", "hadamard"}:
        seed = int(tm.get("rotation_seed") or 0)
        rows = shape[0]
        cols = 1
        for s in shape[1:]:
            cols *= int(s)
        arr = decoded[:rows * cols].reshape(rows, cols)
        if rotation == "hadamard":
            unrotated = torch.from_numpy(_fwht_numpy(arr.cpu().numpy())).to(device)
        else:
            q = torch.from_numpy(_generate_orthogonal_numpy(cols, seed)).to(device)
            unrotated = arr @ q.T
        decoded = unrotated.reshape(-1)

    norm = tm.get("normalization", "none")
    if norm in ("block-max", "awq-block-max", "slrq-block"):
        scales = np.fromfile(
            str(out_dir / tm["scales"]), dtype="<f4", count=int(tm["scale_count"])
        )
        block_size = int(tm.get("block_scale_size") or 32)
        scales_t = torch.from_numpy(scales).to(device)
        n = decoded.numel()
        pad = (-n) % block_size
        if pad:
            decoded = torch.cat([decoded, torch.zeros(pad, dtype=torch.float32, device=device)])
        decoded = (decoded.reshape(-1, block_size) * scales_t[:decoded.numel() // block_size, None]).reshape(-1)
        if pad:
            decoded = decoded[:n]
        if norm == "awq-block-max":
            awq_meta = tm.get("awq_col_scales")
            if awq_meta:
                awq_scales = np.fromfile(
                    str(out_dir / awq_meta["path"]), dtype="<f4", count=int(awq_meta["count"])
                )
                awq_t = torch.from_numpy(awq_scales).to(device)
                cols = shape[-1]
                rows = decoded.numel() // cols
                decoded = (decoded[:rows * cols].reshape(rows, cols) * awq_t[None, :]).reshape(-1)
    elif norm == "row-l2":
        scales = np.fromfile(
            str(out_dir / tm["scales"]), dtype="<f4", count=int(tm["scale_count"])
        )
        scales_t = torch.from_numpy(scales).to(device)
        cols = decoded.numel() // scales_t.numel()
        decoded = (decoded.reshape(-1, cols) * scales_t[:, None]).reshape(-1)
    elif norm in ("col-l2", "awq"):
        scales = np.fromfile(
            str(out_dir / tm["scales"]), dtype="<f4", count=int(tm["scale_count"])
        )
        scales_t = torch.from_numpy(scales).to(device)
        cols = scales_t.numel()
        rows = decoded.numel() // cols
        decoded = (decoded[:rows * cols].reshape(rows, cols) * scales_t[None, :]).reshape(-1)

    salient = tm.get("salient")
    if salient:
        s_idx_np = np.fromfile(str(out_dir / salient["indices"]), dtype="<u4")
        s_val_np = np.fromfile(str(out_dir / salient["weights"]), dtype="<f4")
        
        s_idx = torch.from_numpy(s_idx_np.astype(np.int64)).to(device)
        s_val = torch.from_numpy(s_val_np).to(device)
        
        # SLRQ: re-inject salient weights AFTER scaling to avoid double-scaling.
        block_size = int(tm.get("block_scale_size", 16))
        b_count = len(s_idx)
        b_indices = torch.arange(b_count, device=device)
        flat_indices = b_indices * block_size + s_idx
        
        # Guard against padding
        mask = flat_indices < decoded.numel()
        decoded[flat_indices[mask]] = s_val[mask]

    return decoded.reshape(shape)


def _write_complete_safetensors_reconstruction(
    out_dir: Path, output_path: Path, manifest: dict, device: str | None = None
) -> dict:
    """Reconstruct full model. Uses GPU streaming path when device='cuda' to avoid Python list bloat."""
    if device is not None and "cuda" in str(device).lower():
        try:
            import torch
            if torch.cuda.is_available():
                from safetensors.torch import save_file as save_torch
                from safetensors import safe_open
                arrays: dict = {}
                packed_names = {t["name"] for t in manifest.get("tensors", [])}
                # Passthrough first
                pp = out_dir / "passthrough.safetensors"
                if pp.exists():
                    with safe_open(str(pp), framework="pt") as f:
                        for name in f.keys():
                            arrays[name] = f.get_tensor(name).contiguous()
                # Source fallback for anything missing
                source = Path(manifest["source"])
                if source.exists():
                    with safe_open(str(source), framework="pt") as f:
                        for name in f.keys():
                            if name in packed_names or name in arrays:
                                continue
                            arrays[name] = f.get_tensor(name).contiguous()
                # GPU decode quantized tensors, move to CPU immediately to free GPU memory
                for tm in manifest.get("tensors", []):
                    dec_gpu = _decode_tensor_torch(out_dir, tm, device)
                    arrays[tm["name"]] = dec_gpu.cpu().contiguous()
                    del dec_gpu
                    torch.cuda.empty_cache()
                save_torch(arrays, str(output_path))
                return {"out": str(output_path), "tensor_count": len(arrays), "format": "safetensors"}
        except Exception as exc:
            print(f"GPU reconstruction failed ({exc}); falling back to numpy path", flush=True)
    # CPU/numpy fallback (the slow path)
    tensors = _complete_decoded_tensor_map(out_dir, manifest)
    _write_safetensors_reconstruction(output_path, tensors)
    return {
        "out": str(output_path),
        "tensor_count": len(tensors),
        "format": "safetensors",
    }


def reconstruct_artifact(
    out_dir: Path, output_path: Path, output_format: str = "json"
) -> dict:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Orka manifest: {manifest_path}")
    if output_format not in {"json", "safetensors"}:
        raise ValueError("output_format must be 'json' or 'safetensors'")

    manifest = json.loads(manifest_path.read_text())
    tensors = _decoded_tensor_map(out_dir, manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        _write_json_reconstruction(out_dir, output_path, manifest, tensors)
    else:
        _write_safetensors_reconstruction(output_path, tensors)

    return {
        "out": str(output_path),
        "tensor_count": len(tensors),
        "format": output_format,
    }


def _require_non_empty(name: str, values: Sequence[object]) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")


def _sweep_artifact_root(out_path: Path) -> Path:
    if out_path.suffix:
        return out_path.with_name(f"{out_path.stem}.artifacts")
    return out_path.parent / f"{out_path.name}.artifacts"


def _sweep_artifact_name(
    group_size: int,
    stages: Sequence[int],
    codebook_mode: str,
    normalization: str,
    label: str | None = None,
) -> str:
    if label:
        stage_part = label
    elif len(stages) == 1:
        stage_part = f"k{stages[0]}"
    else:
        stage_part = "rvq" + "+".join(f"k{k}" for k in stages)
    return (
        f"g{group_size}-{stage_part}-"
        f"{_safe_tensor_name(codebook_mode)}-{_safe_tensor_name(normalization)}.orka"
    )


def _reset_sweep_run_dir(path: Path, artifact_root: Path) -> None:
    if not path.exists():
        return
    root = artifact_root.resolve()
    target = path.resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"refusing to remove sweep artifact outside root: {path}")
    if not path.name.endswith(".orka"):
        raise ValueError(f"refusing to remove non-Orka sweep artifact: {path}")
    shutil.rmtree(path)


def _cosine_per_mb(report: dict) -> float:
    artifact_mb = float(report["artifact_bytes"]) / 1_000_000.0
    if artifact_mb <= 0:
        return 0.0
    return float(report["cosine_similarity"]) / artifact_mb


def _best_run(runs: Sequence[dict], key: str, reverse: bool) -> dict | None:
    if not runs:
        return None
    return dict(sorted(runs, key=lambda run: float(run[key]), reverse=reverse)[0])


def _sweep_run_summary(
    artifact_dir: Path,
    group_size: int,
    codebook_size: int,
    codebook_mode: str,
    normalization: str,
    report: dict,
) -> dict:
    return {
        "artifact": str(artifact_dir),
        "group_size": group_size,
        "codebook_size": codebook_size,
        "codebook_mode": codebook_mode,
        "normalization": normalization,
        "tensor_count": report["tensor_count"],
        "artifact_bytes": report["artifact_bytes"],
        "artifact_size": _human_bytes(report["artifact_bytes"]),
        "original_fp16_bytes": report["original_fp16_bytes"],
        "compression_ratio_fp16_to_artifact": report[
            "compression_ratio_fp16_to_artifact"
        ],
        "total_index_bytes": report["total_index_bytes"],
        "total_codebook_bytes": report["total_codebook_bytes"],
        "total_scale_bytes": report["total_scale_bytes"],
        "weighted_mse": report["weighted_mse"],
        "rmse": report["rmse"],
        "mae": report["mae"],
        "max_abs_error": report["max_abs_error"],
        "relative_rmse": report["relative_rmse"],
        "cosine_similarity": report["cosine_similarity"],
        "cosine_per_mb": _cosine_per_mb(report),
    }


def sweep_checkpoint(
    source: Path,
    out_path: Path,
    group_sizes: Sequence[int],
    codebook_sizes: Sequence[int],
    codebook_modes: Sequence[str],
    normalizations: Sequence[str],
    iterations: int,
    max_values_per_tensor: int | None = None,
    sample_vectors: int | None = None,
    backend: str = "auto",
    device: str = "cpu",
    verify_runs: bool = False,
    quant_modes: Sequence[str] = (),
    outlier_frac: float = 0.0,
    rotation: str = "none",
    rotation_seed: int | None = None,
    awq_activations: dict | None = None,
    awq_alpha: float = 0.5,
    awq_alphas: Sequence[float] | None = None,
    max_tensors: int | None = None,
    progress_file: Path | None = None,
    sensitivity_map: dict | None = None,
) -> dict:
    _require_non_empty("group_sizes", group_sizes)
    _require_non_empty("codebook_modes", codebook_modes)
    _require_non_empty("normalizations", normalizations)
    if not quant_modes and not codebook_sizes:
        raise ValueError("at least one of quant_modes or codebook_sizes is required")
    alpha_values = [float(a) for a in awq_alphas] if awq_alphas else [float(awq_alpha)]

    stage_specs: list[
        tuple[list[int] | None, str, int, dict[str, list[int]] | None]
    ] = []
    for k in codebook_sizes or []:
        label = quant_spec_from_sizes([int(k)])
        stage_specs.append(([int(k)], label, int(k), None))
    for mode in quant_modes:
        if is_rvq_mixed_spec(mode):
            family_map = rvq_mixed_family_stages()
            stage_specs.append((None, mode, family_map["other"][0], family_map))
        else:
            stages = parse_quant_spec(mode)
            stage_specs.append((stages, mode, stages[0], None))

    artifact_root = _sweep_artifact_root(out_path)
    artifact_root.mkdir(parents=True, exist_ok=True)
    runs = []

    for group_size in group_sizes:
        for stages, label, primary_k, family_map in stage_specs:
            for codebook_mode in codebook_modes:
                if family_map is not None and codebook_mode != "per-tensor":
                    continue
                for normalization in normalizations:
                    norm_uses_awq = (
                        normalization in {"awq", "awq-block-max"}
                        and awq_activations is not None
                    )
                    alphas_for_norm = (
                        alpha_values if norm_uses_awq else [float(awq_alpha)]
                    )
                    for cur_alpha in alphas_for_norm:
                        alpha_label = (
                            f"a{cur_alpha:.2f}".replace(".", "_")
                            if norm_uses_awq and len(alphas_for_norm) > 1
                            else None
                        )
                        base_name = _sweep_artifact_name(
                            int(group_size),
                            stages or [primary_k],
                            codebook_mode,
                            normalization,
                            label=label,
                        )
                        artifact_name = (
                            base_name
                            if alpha_label is None
                            else f"{base_name[: -len('.orka')]}-{alpha_label}.orka"
                        )
                        artifact_dir = artifact_root / artifact_name
                        _reset_sweep_run_dir(artifact_dir, artifact_root)
                        print(f"Sweep Run: Packing {artifact_name}...", flush=True)
                        pack_checkpoint(
                            source=source,
                            out_dir=artifact_dir,
                            group_size=int(group_size),
                            codebook_size=primary_k,
                            codebook_sizes=stages,
                            iterations=iterations,
                            max_values_per_tensor=max_values_per_tensor,
                            codebook_mode=codebook_mode,
                            sample_vectors=sample_vectors,
                            backend=backend,
                            device=device,
                            normalization=normalization,
                            family_stages_map=family_map,
                            outlier_frac=outlier_frac,
                            rotation=rotation,
                            rotation_seed=rotation_seed,
                            awq_activations=awq_activations,
                            awq_alpha=cur_alpha,
                            max_tensors=max_tensors,
                        )
                        report = report_artifact(artifact_dir)
                        run = _sweep_run_summary(
                            artifact_dir=artifact_dir,
                            group_size=int(group_size),
                            codebook_size=primary_k,
                            codebook_mode=codebook_mode,
                            normalization=normalization,
                            report=report,
                        )
                        run["quant_mode"] = label or "custom"
                        run["awq_alpha"] = cur_alpha if norm_uses_awq else None
                        if family_map is not None:
                            run["stages"] = None
                            run["family_stages_map"] = {
                                fam: [_index_bits_for_size(k) for k in s]
                                for fam, s in family_map.items()
                            }
                            run["bits_per_vector"] = None
                            run["bits_per_weight"] = None
                        else:
                            run["stages"] = list(stages)
                            run["bits_per_vector"] = sum(
                                _index_bits_for_size(k) for k in stages
                            )
                            run["bits_per_weight"] = run["bits_per_vector"] / int(
                                group_size
                            )
                        if verify_runs:
                            run["verify"] = verify_artifact(artifact_dir)
                        runs.append(run)

    summary = {
        "format": "orka-sweep",
        "version": ORKA_VERSION,
        "source": str(source),
        "out": str(out_path),
        "artifact_root": str(artifact_root),
        "backend": backend,
        "device": str(_resolve_torch_device(device)) if backend == "torch" else "cpu",
        "sample_vectors": sample_vectors,
        "iterations": iterations,
        "matrix": {
            "group_sizes": [int(value) for value in group_sizes],
            "codebook_sizes": [int(value) for value in codebook_sizes],
            "codebook_modes": list(codebook_modes),
            "normalizations": list(normalizations),
        },
        "run_count": len(runs),
        "best_by_cosine_similarity": _best_run(runs, "cosine_similarity", reverse=True),
        "best_by_relative_rmse": _best_run(runs, "relative_rmse", reverse=False),
        "best_by_cosine_per_mb": _best_run(runs, "cosine_per_mb", reverse=True),
        "runs": runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    return summary


def _read_prompt_file(path: Path, max_prompts: int | None = None) -> list[str]:
    if max_prompts is not None and max_prompts <= 0:
        raise ValueError("max_prompts must be positive")
    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise ValueError("prompt file must contain at least one non-empty prompt")
    return prompts


def _resolve_eval_model_dir(source: Path, model_dir: Path | None) -> Path:
    candidate = (
        model_dir
        if model_dir is not None
        else (source if source.is_dir() else source.parent)
    )
    if not (candidate / "config.json").exists():
        raise FileNotFoundError(
            f"eval requires a Hugging Face model directory with config.json: {candidate}"
        )
    return candidate


def _safe_exp(value: float) -> float:
    if value > 700:
        return float("inf")
    return math.exp(value)


def _summarize_eval_rows(rows: Sequence[dict]) -> dict:
    _require_non_empty("eval rows", rows)
    token_count = sum(int(row["token_count"]) for row in rows)
    if token_count <= 0:
        raise ValueError("eval rows must contain at least one scored token")
    original_loss = (
        sum(float(row["original_loss"]) * int(row["token_count"]) for row in rows)
        / token_count
    )
    orka_loss = (
        sum(float(row["orka_loss"]) * int(row["token_count"]) for row in rows)
        / token_count
    )
    original_perplexity = _safe_exp(original_loss)
    orka_perplexity = _safe_exp(orka_loss)
    if original_perplexity and math.isfinite(original_perplexity):
        perplexity_ratio = orka_perplexity / original_perplexity
    else:
        perplexity_ratio = float("inf")
    return {
        "prompt_count": len(rows),
        "token_count": token_count,
        "original_loss": original_loss,
        "orka_loss": orka_loss,
        "loss_delta": orka_loss - original_loss,
        "original_perplexity": original_perplexity,
        "orka_perplexity": orka_perplexity,
        "perplexity_ratio": perplexity_ratio,
    }


def _is_model_weight_sidecar(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".safetensors")
        or name.endswith(".bin")
        or name.endswith(".pt")
        or name.endswith(".pth")
        or name.endswith(".onnx")
        or name.endswith(".gguf")
        or name.endswith(".safetensors.index.json")
        or name.endswith(".bin.index.json")
    )


def _copy_hf_sidecars(source_dir: Path, target_dir: Path) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for child in sorted(source_dir.iterdir()):
        if not child.is_file() or _is_model_weight_sidecar(child):
            continue
        if child.suffix.lower() not in {".json", ".txt", ".model"}:
            continue
        shutil.copy2(child, target_dir / child.name)
        copied.append(child.name)
    if "config.json" not in copied:
        raise FileNotFoundError(f"missing config.json in model directory: {source_dir}")
    return copied


def _prepare_reconstructed_hf_dir(
    artifact_dir: Path, original_model_dir: Path, target_dir: Path,
    device: str | None = None,
) -> dict:
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(
            f"reconstructed model directory must be empty: {target_dir}"
        )
    copied = _copy_hf_sidecars(original_model_dir, target_dir)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Orka manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    reconstructed = _write_complete_safetensors_reconstruction(
        artifact_dir,
        target_dir / "model.safetensors",
        manifest,
        device=device,
    )
    return {
        "model_dir": str(target_dir),
        "copied_files": copied,
        "reconstructed": reconstructed,
    }


def _load_hf_eval_dependencies():
    try:
        import numpy  # noqa: F401
        import safetensors  # noqa: F401
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "eval requires optional dependencies: torch, transformers, numpy, and safetensors"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _hf_prompt_losses(
    model_dir: Path,
    prompts: Sequence[str],
    max_length: int,
    device: str,
    local_files_only: bool,
) -> list[dict]:
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    torch, AutoModelForCausalLM, AutoTokenizer = _load_hf_eval_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()

    rows = []
    try:
        with torch.no_grad():
            for prompt in prompts:
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                input_ids = encoded["input_ids"]
                if int(input_ids.shape[-1]) < 2:
                    continue
                model_inputs = {
                    key: value.to(device)
                    for key, value in encoded.items()
                    if key in {"input_ids", "attention_mask"}
                }
                outputs = model(**model_inputs, labels=model_inputs["input_ids"])
                rows.append(
                    {
                        "prompt": prompt,
                        "token_count": int(input_ids.shape[-1]) - 1,
                        "loss": float(outputs.loss.detach().cpu().item()),
                    }
                )
    finally:
        # Release model + tokenizer + cache to free GPU memory before next call.
        try:
            model.to("cpu")
        except Exception:
            pass
        del model, tokenizer
        if device != "cpu":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if not rows:
        raise ValueError("eval prompts produced no scored tokens")
    return rows


def _combine_eval_losses(
    original_rows: Sequence[dict], orka_rows: Sequence[dict]
) -> list[dict]:
    if len(original_rows) != len(orka_rows):
        raise ValueError("original and Orka eval row counts differ")
    rows = []
    for original, orka in zip(original_rows, orka_rows):
        if original["prompt"] != orka["prompt"]:
            raise ValueError("original and Orka prompt order differs")
        rows.append(
            {
                "prompt": original["prompt"],
                "token_count": int(original["token_count"]),
                "original_loss": float(original["loss"]),
                "orka_loss": float(orka["loss"]),
                "loss_delta": float(orka["loss"]) - float(original["loss"]),
            }
        )
    return rows


def eval_artifact(
    artifact_dir: Path,
    prompts_path: Path,
    out_path: Path,
    model_dir: Path | None = None,
    max_prompts: int | None = None,
    max_length: int = 512,
    device: str = "cpu",
    reconstructed_model_dir: Path | None = None,
    local_files_only: bool = True,
) -> dict:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Orka manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest["source"])
    original_model_dir = _resolve_eval_model_dir(source, model_dir)
    prompts = _read_prompt_file(prompts_path, max_prompts=max_prompts)
    _load_hf_eval_dependencies()

    def run_with_reconstructed_dir(target_dir: Path) -> dict:
        prepared = _prepare_reconstructed_hf_dir(
            artifact_dir, original_model_dir, target_dir, device=device
        )
        original_rows = _hf_prompt_losses(
            original_model_dir,
            prompts,
            max_length=max_length,
            device=device,
            local_files_only=local_files_only,
        )
        orka_rows = _hf_prompt_losses(
            target_dir,
            prompts,
            max_length=max_length,
            device=device,
            local_files_only=local_files_only,
        )
        rows = _combine_eval_losses(original_rows, orka_rows)
        summary = _summarize_eval_rows(rows)
        result = {
            "format": "orka-eval",
            "version": ORKA_VERSION,
            "artifact": str(artifact_dir),
            "source": str(source),
            "model_dir": str(original_model_dir),
            "prompts": str(prompts_path),
            "max_length": max_length,
            "device": device,
            "local_files_only": local_files_only,
            "reconstructed_model_dir": str(target_dir),
            "prepared": prepared,
            **summary,
            "rows": rows,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")

        return result

    if reconstructed_model_dir is not None:
        reconstructed_model_dir.mkdir(parents=True, exist_ok=True)
        return run_with_reconstructed_dir(reconstructed_model_dir)

    with tempfile.TemporaryDirectory() as tmp:
        return run_with_reconstructed_dir(Path(tmp) / "reconstructed-model")


def _eval_sweep_output_root(out_path: Path) -> Path:
    if out_path.suffix:
        return out_path.with_name(f"{out_path.stem}.evals")
    return out_path.parent / f"{out_path.name}.evals"


def _eval_run_summary(result: dict) -> dict:
    keys = [
        "prompt_count",
        "token_count",
        "original_loss",
        "orka_loss",
        "loss_delta",
        "original_perplexity",
        "orka_perplexity",
        "perplexity_ratio",
    ]
    return {key: result[key] for key in keys if key in result}


def eval_sweep(
    sweep_path: Path,
    prompts_path: Path,
    out_path: Path,
    model_dir: Path | None = None,
    max_prompts: int | None = None,
    max_length: int = 512,
    device: str = "cpu",
    local_files_only: bool = True,
    max_runs: int | None = None,
    reconstructed_model_root: Path | None = None,
    evaluator: Callable[..., dict] = eval_artifact,
) -> dict:
    if max_runs is not None and max_runs <= 0:
        raise ValueError("max_runs must be positive")

    sweep = json.loads(sweep_path.read_text())
    runs = list(sweep.get("runs", []))
    _require_non_empty("sweep runs", runs)
    if max_runs is not None:
        runs = runs[:max_runs]

    eval_root = _eval_sweep_output_root(out_path)
    eval_root.mkdir(parents=True, exist_ok=True)
    if reconstructed_model_root is not None:
        reconstructed_model_root.mkdir(parents=True, exist_ok=True)

    evaluated_runs = []
    for run_i, run in enumerate(runs):
        artifact_dir = Path(run["artifact"])
        run_name = f"{run_i:04d}-{_safe_tensor_name(artifact_dir.name)}"
        eval_path = eval_root / f"{run_name}.eval.json"
        reconstructed_model_dir = (
            reconstructed_model_root / run_name
            if reconstructed_model_root is not None
            else None
        )
        eval_result = evaluator(
            artifact_dir=artifact_dir,
            prompts_path=prompts_path,
            out_path=eval_path,
            model_dir=model_dir,
            max_prompts=max_prompts,
            max_length=max_length,
            device=device,
            reconstructed_model_dir=reconstructed_model_dir,
            local_files_only=local_files_only,
        )
        eval_summary = _eval_run_summary(eval_result)
        combined = dict(run)
        combined["eval_path"] = str(eval_path)
        combined["eval"] = eval_summary
        combined.update(eval_summary)
        evaluated_runs.append(combined)

    summary = {
        "format": "orka-eval-sweep",
        "version": ORKA_VERSION,
        "source_sweep": str(sweep_path),
        "sweep_source": sweep.get("source"),
        "prompts": str(prompts_path),
        "model_dir": str(model_dir) if model_dir is not None else None,
        "max_prompts": max_prompts,
        "max_length": max_length,
        "device": device,
        "local_files_only": local_files_only,
        "input_run_count": len(sweep.get("runs", [])),
        "run_count": len(evaluated_runs),
        "eval_root": str(eval_root),
        "reconstructed_model_root": (
            str(reconstructed_model_root)
            if reconstructed_model_root is not None
            else None
        ),
        "best_by_loss_delta": _best_run(evaluated_runs, "loss_delta", reverse=False),
        "best_by_perplexity_ratio": _best_run(
            evaluated_runs, "perplexity_ratio", reverse=False
        ),
        "best_by_artifact_bytes": _best_run(
            evaluated_runs, "artifact_bytes", reverse=False
        ),
        "runs": evaluated_runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    return summary


def _parse_params(value: str) -> int:
    text = value.strip().lower().replace("_", "")
    suffixes = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    suffix = text[-1]
    if suffix in suffixes:
        return int(Decimal(text[:-1]) * suffixes[suffix])
    return int(text)


def _human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1000 or unit == units[-1]:
            return f"{value:.3f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1000
    return f"{value:.3f} TB"


def cmd_calc(args: argparse.Namespace) -> int:
    estimate = estimate_payload(
        params=_parse_params(args.params),
        group_size=args.group_size,
        codebook_size=args.codebook_size,
        scale_block_vectors=args.scale_block_vectors,
        scale_bits=args.scale_bits,
    )
    data = asdict(estimate)
    data["index_size"] = _human_bytes(estimate.index_bytes)
    data["scale_size"] = _human_bytes(estimate.scale_bytes)
    data["total_payload_size"] = _human_bytes(estimate.total_payload_bytes)
    print(json.dumps(data, indent=2))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    report = inspect_checkpoint(Path(args.source))
    report["baseline_vq8"] = asdict(estimate_payload(report["total_params"], 8, 256))
    print(json.dumps(report, indent=2))
    return 0


class CappedOutOfMemoryError(RuntimeError):
    pass


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch
    except Exception:
        return False
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or ("cuda error" in msg and "memory" in msg)


def _wrap_capped_oom(cap_gb: float | None, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        if cap_gb and cap_gb > 0 and _is_cuda_oom(exc):
            raise CappedOutOfMemoryError(
                f"GPU memory cap exceeded ({cap_gb} GB): {exc}"
            ) from exc
        raise


CUBLAS_WORKSPACE_SLOP_BYTES = 256 * 1024 * 1024


def _apply_gpu_memory_cap(
    backend: str, device: str, max_gpu_mem_gb: float | None
) -> None:
    if not max_gpu_mem_gb or max_gpu_mem_gb <= 0:
        return
    if backend != "torch":
        return
    try:
        import torch
    except Exception:
        return
    resolved = _resolve_torch_device(device)
    if resolved.type != "cuda":
        return
    device_index = (
        resolved.index if resolved.index is not None else torch.cuda.current_device()
    )
    total_bytes = torch.cuda.get_device_properties(device_index).total_memory
    cap_bytes = int(max_gpu_mem_gb * 1024 * 1024 * 1024)
    fraction = max(0.05, min(1.0, cap_bytes / total_bytes))
    torch.cuda.set_per_process_memory_fraction(fraction, device_index)
    effective_gb = (cap_bytes - CUBLAS_WORKSPACE_SLOP_BYTES) / (1024**3)
    print(
        f"INFO: GPU memory cap = {max_gpu_mem_gb:.2f} GB on cuda:{device_index} "
        f"(fraction={fraction:.4f} of {total_bytes / (1024**3):.2f} GB). "
        f"Cap covers torch caching allocator only; cuBLAS/cuDNN workspace (~{CUBLAS_WORKSPACE_SLOP_BYTES // (1024 * 1024)} MB) "
        f"lives outside it, so plan for ≈{effective_gb:.2f} GB of usable headroom.",
        file=os.sys.stderr,
    )


def _load_awq_activations(args: argparse.Namespace):
    if not args.awq_calibration:
        return None
    prompts = _read_prompt_file(
        Path(args.awq_calibration), max_prompts=args.calibration_max_prompts
    )
    model_dir = (
        Path(args.awq_model_dir) if args.awq_model_dir else Path(args.source).parent
    )
    return _collect_activations_hf(
        model_dir,
        prompts,
        max_length=args.calibration_max_length,
        device=args.device if args.backend == "torch" else "cpu",
        max_samples_per_layer=args.calibration_max_samples,
    )


def cmd_pack(args: argparse.Namespace) -> int:
    _apply_gpu_memory_cap(args.backend, args.device, args.max_gpu_mem_gb)
    awq_activations = _load_awq_activations(args)

    if is_rvq_mixed_spec(args.quant_mode):
        family_map = rvq_mixed_family_stages()
        sizes = [family_map["other"][0]]
        codebook_mode = "per-tensor"
    else:
        family_map = None
        sizes = _resolve_quant_stages(
            args.quant_mode, args.codebook_sizes, args.codebook_size
        )
        codebook_mode = args.codebook_mode
    smap = None
    if getattr(args, "sensitivity_map", None):
        with open(args.sensitivity_map, "r") as f:
            smap = json.load(f)
    manifest = _wrap_capped_oom(
        args.max_gpu_mem_gb,
        pack_checkpoint,
        source=Path(args.source),
        out_dir=Path(args.out),
        group_size=args.group_size,
        codebook_size=sizes[0],
        iterations=args.iterations,
        max_values_per_tensor=args.max_values_per_tensor,
        codebook_mode=codebook_mode,
        sample_vectors=args.sample_vectors,
        backend=args.backend,
        normalization=args.normalization,
        device=args.device,
        codebook_sizes=sizes if family_map is None else None,
        family_stages_map=family_map,
        outlier_frac=args.outlier_frac,
        rotation=args.rotation,
        rotation_seed=args.rotation_seed,
        awq_activations=awq_activations,
        awq_alpha=args.awq_alpha,
        max_tensors=args.max_tensors,
        sensitivity_map=smap,
        progress_file=Path(args.progress_file) if args.progress_file else None,
        codebook_cache_dir=Path(args.codebook_cache).expanduser()
        if args.codebook_cache
        else None,
        block_scale_size=args.block_scale_size,
    )
    print(
        json.dumps(
            {
                "out": args.out,
                "tensor_count": manifest["tensor_count"],
                "total_index_bytes": manifest["total_index_bytes"],
            },
            indent=2,
        )
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report = report_artifact(Path(args.artifact))
    report["artifact_size"] = _human_bytes(report["artifact_bytes"])
    report["original_fp16_size"] = _human_bytes(report["original_fp16_bytes"])
    report["index_size"] = _human_bytes(report["total_index_bytes"])
    report["codebook_size"] = _human_bytes(report["total_codebook_bytes"])
    report["scale_size"] = _human_bytes(report["total_scale_bytes"])
    print(json.dumps(report, indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    result = verify_artifact(Path(args.artifact))
    print(json.dumps(result, indent=2))
    return 0


def cmd_reconstruct(args: argparse.Namespace) -> int:
    result = reconstruct_artifact(
        Path(args.artifact), Path(args.out), output_format=args.format
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    _apply_gpu_memory_cap(args.backend, args.device, args.max_gpu_mem_gb)
    awq_activations = _load_awq_activations(args)

    cb_sizes = list(args.codebook_sizes) if args.codebook_sizes else []
    qmodes = list(args.quant_modes) if args.quant_modes else []
    if not cb_sizes and not qmodes:
        cb_sizes = [256]

    smap = None
    if getattr(args, "sensitivity_map", None):
        with open(args.sensitivity_map, "r") as f:
            smap = json.load(f)

    result = _wrap_capped_oom(
        args.max_gpu_mem_gb,
        sweep_checkpoint,
        outlier_frac=args.outlier_frac,
        rotation=args.rotation,
        rotation_seed=args.rotation_seed,
        source=Path(args.source),
        out_path=Path(args.out),
        group_sizes=args.group_sizes,
        codebook_sizes=cb_sizes,
        codebook_modes=args.codebook_modes,
        normalizations=args.normalizations,
        iterations=args.iterations,
        max_values_per_tensor=args.max_values_per_tensor,
        sample_vectors=args.sample_vectors,
        backend=args.backend,
        device=args.device,
        verify_runs=args.verify,
        quant_modes=qmodes,
        awq_activations=awq_activations,
        awq_alpha=args.awq_alpha,
        awq_alphas=args.awq_alphas,
        max_tensors=args.max_tensors,
        sensitivity_map=smap,
        progress_file=Path(args.progress_file) if args.progress_file else None,
    )
    print(
        json.dumps(
            {
                "out": result["out"],
                "artifact_root": result["artifact_root"],
                "run_count": result["run_count"],
                "best_by_relative_rmse": result["best_by_relative_rmse"],
                "best_by_cosine_per_mb": result["best_by_cosine_per_mb"],
            },
            indent=2,
        )
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    try:
        result = eval_artifact(
            artifact_dir=Path(args.artifact),
            prompts_path=Path(args.prompts),
            out_path=Path(args.out),
            model_dir=Path(args.model_dir) if args.model_dir else None,
            max_prompts=args.max_prompts,
            max_length=args.max_length,
            device=args.device,
            reconstructed_model_dir=Path(args.reconstructed_model_dir)
            if args.reconstructed_model_dir
            else None,
            local_files_only=not args.allow_download,
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=os.sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "out": args.out,
                "artifact": result["artifact"],
                "prompt_count": result["prompt_count"],
                "token_count": result["token_count"],
                "original_loss": result["original_loss"],
                "orka_loss": result["orka_loss"],
                "loss_delta": result["loss_delta"],
                "original_perplexity": result["original_perplexity"],
                "orka_perplexity": result["orka_perplexity"],
                "perplexity_ratio": result["perplexity_ratio"],
            },
            indent=2,
        )
    )
    return 0


def cmd_eval_sweep(args: argparse.Namespace) -> int:
    try:
        result = eval_sweep(
            sweep_path=Path(args.sweep),
            prompts_path=Path(args.prompts),
            out_path=Path(args.out),
            model_dir=Path(args.model_dir) if args.model_dir else None,
            max_prompts=args.max_prompts,
            max_length=args.max_length,
            device=args.device,
            local_files_only=not args.allow_download,
            max_runs=args.max_runs,
            reconstructed_model_root=(
                Path(args.reconstructed_model_root)
                if args.reconstructed_model_root
                else None
            ),
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=os.sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "out": args.out,
                "eval_root": result["eval_root"],
                "run_count": result["run_count"],
                "best_by_loss_delta": result["best_by_loss_delta"],
                "best_by_perplexity_ratio": result["best_by_perplexity_ratio"],
                "best_by_artifact_bytes": result["best_by_artifact_bytes"],
            },
            indent=2,
        )
    )
    return 0


def _load_hf_token() -> str | None:
    for candidate in (
        Path("/kaggle/input/hf-token-private/hf_token.txt"),
        Path("/kaggle/input/hf-token/hf_token.txt"),
    ):
        if candidate.exists():
            tok = candidate.read_text().strip()
            if tok:
                return tok
    if Path("/kaggle/input").exists():
        for name in ("hf_token.txt", "HF_TOKEN", "token"):
            hits = list(Path("/kaggle/input").rglob(name))
            if hits:
                tok = hits[0].read_text().strip()
                if tok:
                    return tok
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        for secret in ("HF_TOKEN", "huggingface_token", "HF_HUB_TOKEN"):
            try:
                tok = client.get_secret(secret)
                if tok:
                    return tok
            except Exception:
                pass
    except ImportError:
        pass
    return os.environ.get("HF_TOKEN")


def _hf_snapshot_with_retry(
    repo_id: str,
    local_dir: Path,
    token: str | None,
    allow_patterns,
    max_retries: int = 3,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub required: pip install huggingface_hub") from exc
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                token=token,
                allow_patterns=allow_patterns,
            )
            return
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = 5 * attempt
                print(f"Download attempt {attempt} failed ({exc}); retry in {delay}s...", flush=True)
                time.sleep(delay)
    raise RuntimeError(f"Download failed after {max_retries} attempts") from last_exc


def _hf_upload_with_retry(
    api,
    folder_path: str,
    repo_id: str,
    max_retries: int = 3,
) -> None:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_folder(folder_path=folder_path, repo_id=repo_id, repo_type="model")
            return
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = 10 * attempt
                print(f"Upload attempt {attempt} failed ({exc}); retry in {delay}s...", flush=True)
                time.sleep(delay)
    raise RuntimeError(f"Upload failed after {max_retries} attempts") from last_exc


def cmd_kaggle_pack(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub required. Run: pip install huggingface_hub", file=os.sys.stderr)
        return 1

    token = _load_hf_token()
    if not token:
        print(
            "Error: HF token not found. Attach hf-token-private Kaggle dataset, "
            "add a Kaggle Secret named HF_TOKEN, or set the HF_TOKEN env var.",
            file=os.sys.stderr,
        )
        return 1

    on_kaggle = Path("/kaggle/working").exists()

    if args.out:
        out_dir = Path(args.out)
    elif on_kaggle:
        slug = args.repo_id.split("/")[-1]
        out_dir = Path("/kaggle/working") / f"{slug}.orka"
    else:
        print("Error: --out required when not running on Kaggle.", file=os.sys.stderr)
        return 1

    src_dir = (Path("/kaggle/tmp") / "orka_src_model") if on_kaggle else (
        Path(tempfile.mkdtemp()) / "orka_src_model"
    )
    src_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"--- Downloading {args.repo_id} ---", flush=True)
        _hf_snapshot_with_retry(
            repo_id=args.repo_id,
            local_dir=src_dir,
            token=token,
            allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"],
        )

        source_file = next(src_dir.glob("*.safetensors"), None)
        if not source_file:
            print(f"Error: no .safetensors found in {args.repo_id}", file=os.sys.stderr)
            return 1

        print(f"--- Packing {source_file.name} ---", flush=True)

        if is_rvq_mixed_spec(args.quant_mode):
            _kp_family_map = rvq_mixed_family_stages()
            _kp_sizes = [_kp_family_map["other"][0]]
            _kp_codebook_mode = "per-tensor"
        else:
            _kp_family_map = None
            _kp_sizes = _resolve_quant_stages(
                args.quant_mode,
                getattr(args, "codebook_sizes", None),
                args.codebook_size,
            )
            _kp_codebook_mode = args.codebook_mode

        if args.awq_calibration:
            args.awq_model_dir = str(src_dir)
        _kp_awq = _load_awq_activations(args)

        _kp_smap = None
        if getattr(args, "sensitivity_map", None):
            with open(args.sensitivity_map) as f:
                _kp_smap = json.load(f)

        _apply_gpu_memory_cap(args.backend, args.device, args.max_gpu_mem_gb)

        manifest = pack_checkpoint(
            source=source_file,
            out_dir=out_dir,
            group_size=args.group_size,
            codebook_size=_kp_sizes[0],
            codebook_sizes=_kp_sizes if _kp_family_map is None else None,
            family_stages_map=_kp_family_map,
            codebook_mode=_kp_codebook_mode,
            backend=args.backend,
            device=args.device,
            normalization=args.normalization,
            block_scale_size=args.block_scale_size,
            rotation=args.rotation,
            rotation_seed=args.rotation_seed,
            sample_vectors=args.sample_vectors,
            iterations=args.iterations,
            max_values_per_tensor=args.max_values_per_tensor,
            outlier_frac=args.outlier_frac,
            awq_activations=_kp_awq,
            awq_alpha=args.awq_alpha,
            progress_file=Path(args.progress_file) if args.progress_file else None,
            sensitivity_map=_kp_smap,
            max_tensors=args.max_tensors,
        )

        artifact_report = report_artifact(out_dir)
        pack_report = {
            "source_repo": args.repo_id,
            "upload_repo": args.upload_repo,
            "artifact": str(out_dir),
            "tensor_count": manifest["tensor_count"],
            "group_size": args.group_size,
            "codebook_mode": _kp_codebook_mode,
            "normalization": args.normalization,
            "artifact_bytes": artifact_report["artifact_bytes"],
            "artifact_size": _human_bytes(artifact_report["artifact_bytes"]),
            "original_fp16_bytes": artifact_report["original_fp16_bytes"],
            "compression_ratio_fp16_to_artifact": artifact_report[
                "compression_ratio_fp16_to_artifact"
            ],
            "weighted_mse": artifact_report["weighted_mse"],
            "relative_rmse": artifact_report["relative_rmse"],
            "cosine_similarity": artifact_report["cosine_similarity"],
        }
        report_path = (
            Path("/kaggle/working/pack_report.json") if on_kaggle
            else out_dir.parent / "pack_report.json"
        )
        report_path.write_text(json.dumps(pack_report, indent=2) + "\n")
        print(f"Pack report written to {report_path}", flush=True)

        if getattr(args, "run_eval", False):
            print("--- Running perplexity eval ---", flush=True)
            eval_prompts = (
                Path(args.eval_prompts) if args.eval_prompts
                else (Path(args.awq_calibration) if args.awq_calibration else None)
            )
            if eval_prompts is None or not eval_prompts.exists():
                print("WARNING: no eval prompts file; skipping eval", flush=True)
            else:
                eval_out = (
                    Path("/kaggle/working/eval_report.json") if on_kaggle
                    else out_dir.parent / "eval_report.json"
                )
                try:
                    eval_result = eval_artifact(
                        artifact_dir=out_dir,
                        prompts_path=eval_prompts,
                        out_path=eval_out,
                        model_dir=src_dir,
                        max_prompts=args.eval_max_prompts,
                        max_length=args.eval_max_length,
                        device=args.device if args.backend == "torch" else "cpu",
                        local_files_only=True,
                    )
                    pack_report["eval"] = {
                        "prompt_count": eval_result["prompt_count"],
                        "token_count": eval_result["token_count"],
                        "original_loss": eval_result["original_loss"],
                        "orka_loss": eval_result["orka_loss"],
                        "loss_delta": eval_result["loss_delta"],
                        "original_perplexity": eval_result["original_perplexity"],
                        "orka_perplexity": eval_result["orka_perplexity"],
                        "perplexity_ratio": eval_result["perplexity_ratio"],
                    }
                    report_path.write_text(json.dumps(pack_report, indent=2) + "\n")
                    print(f"Eval report written to {eval_out}", flush=True)
                except Exception as exc:
                    print(f"Eval failed: {exc}", flush=True)
                    pack_report["eval_error"] = str(exc)
                    report_path.write_text(json.dumps(pack_report, indent=2) + "\n")

        print("--- Cleaning up source model to free disk space ---", flush=True)
        shutil.rmtree(str(src_dir), ignore_errors=True)

        if args.upload_repo:
            print(f"--- Uploading to {args.upload_repo} ---", flush=True)
            api = HfApi(token=token)
            api.create_repo(args.upload_repo, repo_type="model", exist_ok=True)
            _hf_upload_with_retry(api, str(out_dir), args.upload_repo)
            print(f"Uploaded to {args.upload_repo}", flush=True)

        print(json.dumps(pack_report, indent=2))
        return 0

    finally:
        if src_dir.exists():
            shutil.rmtree(str(src_dir), ignore_errors=True)


# --- SLRQ EXPERIMENTAL INTEGRATION ---
def quantize_block_salient_slrq_vectorized(weights, block_size=16, bits_offset=4):
    import numpy as np
    w = weights.flatten()
    pad = (block_size - (len(w) % block_size)) % block_size
    w_padded = np.concatenate([w, np.zeros(pad)])
    blocks = w_padded.reshape(-1, block_size)
    
    abs_blocks = np.abs(blocks)
    max_indices = np.argmax(abs_blocks, axis=1)
    row_indices = np.arange(len(blocks))
    
    salient_weights = blocks[row_indices, max_indices].copy()
    blocks_no_salient = blocks.copy()
    blocks_no_salient[row_indices, max_indices] = 0.0
    
    max_rem = np.max(np.abs(blocks_no_salient), axis=1)
    anchors = 2**np.ceil(np.log2(max_rem + 1e-9))
    
    levels = 2**(bits_offset - 1) - 1
    quantized = np.round((blocks / anchors[:, np.newaxis]) * levels)
    recon_blocks = (quantized / levels) * anchors[:, np.newaxis]
    
    recon_blocks[row_indices, max_indices] = salient_weights
    return recon_blocks.flatten()[:len(w)].reshape(weights.shape)

def cmd_slrq_eval(args):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Requires torch and transformers")
        return 1

    print(f"Loading {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto")
    
    prompts = [
        "The history of artificial intelligence began in antiquity.",
        "Quantum mechanics describes physical properties of nature.",
        "Climate change refers to long-term shifts in temperatures.",
        "Machine learning algorithms build a model from data.",
        "The theory of relativity is a theory of gravitation."
    ]
    if args.prompts:
        from pathlib import Path
        prompts = [line.strip() for line in Path(args.prompts).read_text().splitlines() if line.strip()][:args.max_prompts]
        
    def eval_model(m):
        m.eval()
        total_loss = 0
        total_tokens = 0
        with torch.no_grad():
            for prompt in prompts:
                encoded = tokenizer(prompt, return_tensors="pt").to(m.device)
                if encoded["input_ids"].shape[-1] < 2: continue
                outputs = m(**encoded, labels=encoded["input_ids"])
                tokens = encoded["input_ids"].shape[-1] - 1
                total_loss += outputs.loss.item() * tokens
                total_tokens += tokens
        avg_loss = total_loss / total_tokens if total_tokens else 0
        import math
        return math.exp(avg_loss) if avg_loss < 100 else float('inf')

    print("Evaluating Baseline (FP16)...")
    ppl_base = eval_model(model)
    print(f"Baseline Perplexity: {ppl_base:.4f}")
    
    print(f"Applying Vectorized SLRQ ({args.bits}-bit, block={args.block_size}) to all Linear layers...")
    import time
    t0 = time.time()
    import numpy as np
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if "lm_head" in name or "embed" in name:
                    continue
                print(f"  Quantizing {name}...")
                w_np = module.weight.detach().cpu().numpy().astype(np.float32)
                w_recon = quantize_block_salient_slrq_vectorized(w_np, block_size=args.block_size, bits_offset=args.bits)
                module.weight.copy_(torch.from_numpy(w_recon).to(module.weight.device).to(module.weight.dtype))
    print(f"Quantization done in {time.time() - t0:.1f}s")
    
    print("Evaluating SLRQ...")
    ppl_slrq = eval_model(model)
    print(f"SLRQ Perplexity: {ppl_slrq:.4f}")
    
    print(f"Perplexity Ratio (SLRQ/Base): {ppl_slrq / ppl_base:.4f}")
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orka model compiler prototype")
    sub = parser.add_subparsers(dest="command", required=True)

    calc = sub.add_parser("calc", help="estimate Orka payload size")
    calc.add_argument(
        "--params", required=True, help="parameter count, for example 8.03b"
    )
    calc.add_argument("--group-size", type=int, default=8)
    calc.add_argument("--codebook-size", type=int, default=256)
    calc.add_argument("--scale-block-vectors", type=int, default=64)
    calc.add_argument("--scale-bits", type=int, default=16)
    calc.set_defaults(func=cmd_calc)

    inspect = sub.add_parser(
        "inspect", help="inspect a safetensors or PyTorch checkpoint"
    )
    inspect.add_argument("source")
    inspect.set_defaults(func=cmd_inspect)

    def add_pack_args(p):
        p.add_argument("--group-size", type=int, default=8)
        p.add_argument("--codebook-size", type=int, default=256)
        p.add_argument(
            "--codebook-sizes",
            type=int,
            nargs="+",
            default=None,
            help="explicit per-stage codebook sizes (overrides --codebook-size and --quant-mode)",
        )
        p.add_argument(
            "--quant-mode",
            default=None,
            help="compositional spec like vq-8 or vq-16-8 (per-stage bits, 1..16, total ≤ 64)",
        )
        p.add_argument(
            "--codebook-mode",
            choices=["per-tensor", "global", "family"],
            default="per-tensor",
        )
        p.add_argument(
            "--backend", choices=["auto", "numpy", "torch"], default="auto"
        )
        p.add_argument(
            "--device",
            default="cpu",
            help="torch backend device, for example cpu, cuda, cuda:0, or auto",
        )
        p.add_argument(
            "--normalization",
            choices=["none", "row-l2", "col-l2", "block-max", "awq", "awq-block-max", "slrq-block"],
            default="none",
        )
        p.add_argument(
            "--block-scale-size",
            type=int,
            default=32,
            help="elements per block when --normalization block-max (typical 16 or 32)",
        )
        p.add_argument(
            "--rotation",
            choices=["none", "orthogonal", "hadamard"],
            default="none",
            help="rotation along inner axis before VQ. orthogonal: per-tensor seeded random orthogonal (any size). hadamard: deterministic FWHT (requires power-of-2 last dim).",
        )
        p.add_argument(
            "--rotation-seed",
            type=int,
            default=None,
            help="seed for orthogonal rotation (deterministic)",
        )
        p.add_argument("--sample-vectors", type=int, default=None)
        p.add_argument("--iterations", type=int, default=12)
        p.add_argument("--max-values-per-tensor", type=int, default=None)
        p.add_argument(
            "--max-gpu-mem-gb",
            type=float,
            default=None,
            help="strict cap on per-process GPU memory (GB)",
        )
        p.add_argument(
            "--outlier-frac",
            type=float,
            default=0.0,
            help="fraction of top-magnitude weights kept as fp16 sidecar (e.g. 0.001 = 0.1%%)",
        )
        p.add_argument(
            "--awq-calibration",
            default=None,
            help="prompts file for AWQ calibration; enables activation-aware VQ",
        )
        p.add_argument(
            "--awq-model-dir",
            default=None,
            help="HF model dir for AWQ activation collection",
        )
        p.add_argument(
            "--awq-alpha",
            type=float,
            default=0.5,
            help="activation magnitude scaling power (default 0.5)",
        )
        p.add_argument("--calibration-max-prompts", type=int, default=32)
        p.add_argument("--calibration-max-length", type=int, default=256)
        p.add_argument(
            "--calibration-max-samples",
            type=int,
            default=4096,
            help="max activation samples retained per layer for AWQ calibration",
        )
        p.add_argument("--progress-file", help="file to write real-time progress status")
        p.add_argument(
            "--sensitivity-map",
            help="JSON file from sensitivity.py to enable mixed-precision",
        )
        p.add_argument(
            "--max-tensors",
            type=int,
            default=None,
            help="limit pack to first N tensors (for fail-fast iteration)",
        )
        p.add_argument(
            "--codebook-cache",
            default=None,
            help="dir to cache stage-0 codebooks (zero-loss reuse on identical configs)",
        )

    pack = sub.add_parser(
        "pack", help="pack candidate weight tensors into an .orka directory"
    )
    pack.add_argument("source")
    pack.add_argument("--out", required=True)
    add_pack_args(pack)
    pack.set_defaults(func=cmd_pack)

    kp = sub.add_parser(
        "kaggle-pack", help="Download from HF, pack on Kaggle, and upload back to HF"
    )
    kp.add_argument("--repo-id", required=True, help="HF model repo to download")
    kp.add_argument(
        "--out",
        default=None,
        help="output .orka directory (default on Kaggle: /kaggle/working/<slug>.orka)",
    )
    kp.add_argument("--upload-repo", help="HF repo to upload the result to")
    add_pack_args(kp)
    kp.add_argument("--run-eval", action="store_true",
                    help="run perplexity eval after packing")
    kp.add_argument("--eval-prompts", default=None,
                    help="prompts file for perplexity eval (defaults to AWQ calibration file)")
    kp.add_argument("--eval-max-prompts", type=int, default=16)
    kp.add_argument("--eval-max-length", type=int, default=128)
    kp.set_defaults(func=cmd_kaggle_pack)

    report = sub.add_parser("report", help="summarize an .orka artifact")
    report.add_argument("artifact")
    report.set_defaults(func=cmd_report)

    verify = sub.add_parser(
        "verify", help="decode an .orka artifact and recompute source MSE"
    )
    verify.add_argument("artifact")
    verify.set_defaults(func=cmd_verify)

    reconstruct = sub.add_parser(
        "reconstruct", help="decode an .orka artifact to JSON tensors"
    )
    reconstruct.add_argument("artifact")
    reconstruct.add_argument("--out", required=True)
    reconstruct.add_argument(
        "--format", choices=["json", "safetensors"], default="json"
    )
    reconstruct.set_defaults(func=cmd_reconstruct)

    sweep = sub.add_parser(
        "sweep", help="run a pack/report matrix and write a JSON summary"
    )
    sweep.add_argument("source")
    sweep.add_argument("--out", required=True)
    sweep.add_argument("--group-sizes", type=int, nargs="+", default=[8])
    sweep.add_argument(
        "--codebook-sizes",
        type=int,
        nargs="+",
        default=None,
        help="single-stage codebook sizes to sweep",
    )
    sweep.add_argument(
        "--quant-modes",
        nargs="+",
        default=None,
        help="compositional specs (e.g. vq-8 vq-16 vq-16-8 vq-16-16-16-16)",
    )
    sweep.add_argument(
        "--codebook-modes",
        choices=["per-tensor", "global", "family"],
        nargs="+",
        default=["global"],
    )
    sweep.add_argument(
        "--normalizations",
        choices=["none", "row-l2", "col-l2", "block-max", "awq", "awq-block-max", "slrq-block"],
        nargs="+",
        default=["none", "row-l2"],
    )
    sweep.add_argument(
        "--rotation", choices=["none", "orthogonal", "hadamard"], default="none"
    )
    sweep.add_argument("--rotation-seed", type=int, default=None)
    sweep.add_argument(
        "--backend", choices=["auto", "numpy", "torch"], default="auto"
    )
    sweep.add_argument(
        "--device",
        default="cpu",
        help="torch backend device, for example cpu, cuda, cuda:0, or auto",
    )
    sweep.add_argument("--sample-vectors", type=int, default=None)
    sweep.add_argument("--iterations", type=int, default=12)
    sweep.add_argument("--max-values-per-tensor", type=int, default=None)
    sweep.add_argument(
        "--verify",
        action="store_true",
        help="verify every sweep artifact after packing",
    )
    sweep.add_argument(
        "--max-gpu-mem-gb",
        type=float,
        default=None,
        help="strict cap on per-process GPU memory (GB)",
    )
    sweep.add_argument(
        "--progress-file", help="file to write real-time progress status"
    )
    sweep.add_argument(
        "--max-tensors", type=int, default=None, help="limit sweep to first N tensors"
    )
    sweep.add_argument(
        "--outlier-frac",
        type=float,
        default=0.0,
        help="fraction of top-magnitude weights kept as fp16 sidecar",
    )
    sweep.add_argument(
        "--awq-calibration",
        default=None,
        help="prompts file for AWQ calibration; enables activation-aware VQ",
    )
    sweep.add_argument(
        "--awq-model-dir",
        default=None,
        help="HF model dir for AWQ activation collection",
    )
    sweep.add_argument(
        "--awq-alpha",
        type=float,
        default=0.5,
        help="activation magnitude scaling power (default 0.5)",
    )
    sweep.add_argument(
        "--awq-alphas",
        type=float,
        nargs="+",
        default=None,
        help="sweep multiple AWQ alphas in one run; overrides --awq-alpha when set",
    )
    sweep.add_argument("--calibration-max-prompts", type=int, default=32)
    sweep.add_argument("--calibration-max-length", type=int, default=256)
    sweep.add_argument("--calibration-max-samples", type=int, default=4096)
    sweep.set_defaults(func=cmd_sweep)

    eval_cmd = sub.add_parser(
        "eval", help="evaluate an .orka artifact with Hugging Face prompt loss"
    )
    eval_cmd.add_argument("artifact")
    eval_cmd.add_argument(
        "--prompts", required=True, help="text file with one prompt per non-empty line"
    )
    eval_cmd.add_argument("--out", required=True)
    eval_cmd.add_argument(
        "--model-dir", default=None, help="override Hugging Face model directory"
    )
    eval_cmd.add_argument("--max-prompts", type=int, default=None)
    eval_cmd.add_argument("--max-length", type=int, default=512)
    eval_cmd.add_argument("--device", default="cpu")
    eval_cmd.add_argument("--reconstructed-model-dir", default=None)
    eval_cmd.add_argument(
        "--allow-download",
        action="store_true",
        help="allow transformers to download missing files",
    )
    eval_cmd.set_defaults(func=cmd_eval)

    eval_sweep_cmd = sub.add_parser(
        "eval-sweep", help="evaluate every artifact recorded in a sweep JSON"
    )
    eval_sweep_cmd.add_argument("sweep")
    eval_sweep_cmd.add_argument(
        "--prompts", required=True, help="text file with one prompt per non-empty line"
    )
    eval_sweep_cmd.add_argument("--out", required=True)
    eval_sweep_cmd.add_argument(
        "--model-dir", default=None, help="override Hugging Face model directory"
    )
    eval_sweep_cmd.add_argument("--max-prompts", type=int, default=None)
    eval_sweep_cmd.add_argument("--max-length", type=int, default=512)
    eval_sweep_cmd.add_argument("--device", default="cpu")
    eval_sweep_cmd.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="evaluate only the first N sweep runs",
    )
    eval_sweep_cmd.add_argument(
        "--reconstructed-model-root",
        default=None,
        help="keep reconstructed model directories under this root",
    )
    eval_sweep_cmd.add_argument(
        "--allow-download",
        action="store_true",
        help="allow transformers to download missing files",
    )
    eval_sweep_cmd.set_defaults(func=cmd_eval_sweep)

    def _run_tests(_args):
        from orka_test import run_selftests

        return run_selftests()

    slrq = sub.add_parser("slrq-eval", help="Test SLRQ hypothesis directly on a HuggingFace model in memory")
    slrq.add_argument("--model-id", required=True, help="HF model ID or path")
    slrq.add_argument("--prompts", default=None, help="Optional text file of prompts")
    slrq.add_argument("--max-prompts", type=int, default=16)
    slrq.add_argument("--block-size", type=int, default=16)
    slrq.add_argument("--bits", type=int, default=4)
    slrq.set_defaults(func=cmd_slrq_eval)

    selftest = sub.add_parser("selftest", help="run built-in tests")
    selftest.set_defaults(func=_run_tests)
    return parser


if __name__ == "__main__":
    import sys as _sys

    if Path("/kaggle/working").exists():
        # ── KAGGLE CONFIG ─────────────────────────────────────────────────────
        # Edit these values before pushing to Kaggle.
        # Defaults match the best-loss config (smollm2-135m-ultimate):
        #   awq-block-max + family + orthogonal rotation + sensitivity skip.
        _KAGGLE_CONFIG = {
            "repo_id":         "Qwen/Qwen3-0.6B",
            "upload_repo":     None,
            "quant_mode":      "rvq-16-8-8",       # 3 stages: [65536, 256, 256] = 4 bits/weight
            "codebook_mode":   "per-tensor",      # each tensor gets own codebook (gate/up_proj need this)
            "normalization":   "awq-block-max",   # AWQ scaling + block-max with real Wikitext calib
            "rotation":        "orthogonal",      # smear outliers
            "rotation_seed":   42,
            "backend":         "torch",
            "device":          "cuda",
            "max_gpu_mem_gb":  14.0,
            "sample_vectors":  1000000,
            "iterations":      12,
            "outlier_frac":    0.001,             # top 0.1% values escape as fp16 sidecar
            "group_size":      8,
            "codebook_size":   256,
            "awq_calibration": True,              # ON - use real Wikitext for activations
            "awq_alpha":       0.5,
            "calibration_max_prompts": 128,
            "calibration_max_length":  512,
            "skip_sensitive":  True,              # skip lm_head + embed_tokens (FP16 passthrough)
            "run_eval":        True,              # run perplexity eval after pack
            "eval_max_prompts": 64,
            "eval_max_length":  256,
        }
        # ── END CONFIG ────────────────────────────────────────────────────────

        if len(_sys.argv) == 1:
            print("Kaggle: building args from _KAGGLE_CONFIG", flush=True)
            cfg = _KAGGLE_CONFIG

            # Download Wikitext-2 samples for AWQ calibration + perplexity eval.
            calib_path = Path("/tmp/orka_calib_prompts.txt")
            if cfg.get("awq_calibration") or cfg.get("run_eval"):
                try:
                    from datasets import load_dataset
                    print("Kaggle: loading Wikitext-2-raw test split ...", flush=True)
                    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
                    samples = []
                    target = max(int(cfg.get("calibration_max_prompts", 128)),
                                 int(cfg.get("eval_max_prompts", 64))) + 32
                    for row in ds:
                        text = (row.get("text") or "").strip()
                        if len(text) >= 200:  # skip headers / short fragments
                            samples.append(text)
                            if len(samples) >= target:
                                break
                    if not samples:
                        raise RuntimeError("no usable Wikitext samples")
                    calib_path.write_text("\n".join(samples))
                    print(f"Kaggle: wrote {len(samples)} Wikitext samples to {calib_path}", flush=True)
                except Exception as exc:
                    print(f"Kaggle: Wikitext fetch failed ({exc}); falling back to inline prompts", flush=True)
                    calib_path.write_text("\n".join([
                        "The history of artificial intelligence began in antiquity.",
                        "Quantum mechanics describes physical properties of nature.",
                        "Climate change refers to long-term shifts in temperatures.",
                        "Machine learning algorithms build a model from data.",
                        "The theory of relativity is a theory of gravitation.",
                        "Photosynthesis is the process by which green plants synthesize foods.",
                        "DNA carries genetic instructions for the development of organisms.",
                        "Black holes are regions of spacetime where gravity is strong.",
                        "Neural networks are inspired by biological neural networks.",
                        "Cellular respiration converts biochemical energy from nutrients.",
                        "The water cycle describes movement of water on Earth.",
                        "Stars are luminous spheres of plasma held together by gravity.",
                        "Programming languages produce various kinds of output.",
                        "Mathematics is the abstract science of number, quantity, and space.",
                        "The Milky Way galaxy contains our Solar System.",
                        "Vaccines stimulate the immune system to combat pathogens.",
                    ]))

            # Stub sensitivity map: skip lm_head + embed_tokens (orka skips loss_delta>1.5 OR embed/lm_head substring).
            smap_path = Path("/tmp/orka_sensitivity_map.json")
            if cfg.get("skip_sensitive"):
                import json as _json
                smap_path.write_text(_json.dumps({
                    "base_loss": 0.0,
                    "layers": [
                        {"layer": "lm_head", "loss_delta": 999.0, "sensitivity": "high"},
                        {"layer": "model.embed_tokens", "loss_delta": 999.0, "sensitivity": "high"},
                    ],
                }))
                print(f"Kaggle: wrote sensitivity stub to {smap_path}", flush=True)

            _sys.argv += [
                "kaggle-pack",
                "--repo-id",        cfg["repo_id"],
                "--quant-mode",     cfg["quant_mode"],
                "--codebook-mode",  cfg["codebook_mode"],
                "--normalization",  cfg["normalization"],
                "--rotation",       cfg["rotation"],
                "--backend",        cfg["backend"],
                "--device",         cfg["device"],
                *(["--max-gpu-mem-gb", str(cfg["max_gpu_mem_gb"])] if cfg.get("max_gpu_mem_gb") is not None else []),
                *(["--rotation-seed", str(cfg["rotation_seed"])] if cfg.get("rotation_seed") is not None else []),
                "--sample-vectors", str(cfg["sample_vectors"]),
                "--iterations",     str(cfg["iterations"]),
                "--outlier-frac",   str(cfg["outlier_frac"]),
                "--group-size",     str(cfg["group_size"]),
                "--codebook-size",  str(cfg["codebook_size"]),
            ]
            if cfg.get("awq_calibration"):
                _sys.argv += [
                    "--awq-calibration", str(calib_path),
                    "--awq-alpha",       str(cfg["awq_alpha"]),
                    "--calibration-max-prompts", str(cfg.get("calibration_max_prompts", 32)),
                    "--calibration-max-length",  str(cfg.get("calibration_max_length", 256)),
                ]
            if cfg.get("skip_sensitive"):
                _sys.argv += ["--sensitivity-map", str(smap_path)]
            if cfg.get("run_eval"):
                # Always write a small prompts file for eval (even if AWQ off, we still need prompts).
                if not calib_path.exists():
                    calib_path.write_text("\n".join([
                        "The history of artificial intelligence began in antiquity.",
                        "Quantum mechanics describes physical properties of nature.",
                        "Climate change refers to long-term shifts in temperatures.",
                        "Machine learning algorithms build a model from data.",
                        "The theory of relativity is a theory of gravitation.",
                        "Photosynthesis is the process by which green plants synthesize foods.",
                        "DNA carries genetic instructions for the development of organisms.",
                        "Black holes are regions of spacetime where gravity is strong.",
                        "Neural networks are inspired by biological neural networks.",
                        "Cellular respiration converts biochemical energy from nutrients.",
                        "The water cycle describes movement of water on Earth.",
                        "Stars are luminous spheres of plasma held together by gravity.",
                        "Programming languages produce various kinds of output.",
                        "Mathematics is the abstract science of number, quantity, and space.",
                        "The Milky Way galaxy contains our Solar System.",
                        "Vaccines stimulate the immune system to combat pathogens.",
                    ]))
                _sys.argv += [
                    "--run-eval",
                    "--eval-prompts",     str(calib_path),
                    "--eval-max-prompts", str(cfg["eval_max_prompts"]),
                    "--eval-max-length",  str(cfg["eval_max_length"]),
                ]
            if cfg.get("upload_repo"):
                _sys.argv += ["--upload-repo", cfg["upload_repo"]]

    cli_args = build_parser().parse_args()
    raise SystemExit(cli_args.func(cli_args))
