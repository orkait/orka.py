"""Per-tensor deep-probe on a SAMPLED weight block (CPU, numpy, no GPU, no full download).

Given the leading weights of one tensor (range-fetched), runs orka's real VQ primitives on
the sample to produce the data the Tensor/3D views render: value distribution, an RVQ
rate-distortion curve, a learned codebook (utilization + index entropy), the 3 bpw residual
error map, and a 3D PCA projection of weight vectors -> codebook centroids."""
from __future__ import annotations

import contextlib
import os

import numpy as np

from orka.codebook import learn_codebook_auto, quantize_vectors_auto


@contextlib.contextmanager
def _silence():
    """orka's numpy k-means prints Lloyd progress + a tqdm bar; mute it for the API."""
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
        yield

_K = 256                 # codewords per RVQ stage (8 bits -> 1 bpw at group 8)
_ITERS = 6
_MAX_VECS = 4096         # cap vectors fed to k-means (keeps the probe ~100s of ms)


def _rvq(vecs: np.ndarray, stages: int, seed: int = 0):
    """Residual VQ: return (reconstruction, stage1_codebook, stage1_indices)."""
    residual = vecs.copy()
    recon = np.zeros_like(vecs)
    cb1 = idx1 = None
    k = min(_K, len(vecs))
    for s in range(stages):
        with _silence():
            cb, _, _ = learn_codebook_auto(residual, k, _ITERS, "numpy", "cpu", seed=seed + s)
            cb = np.asarray(cb, dtype=np.float32)
            idx, _ = quantize_vectors_auto(residual, cb, "numpy", "cpu")
        idx = np.asarray(idx).reshape(-1)
        decoded = cb[idx]
        recon = recon + decoded
        residual = vecs - recon
        if s == 0:
            cb1, idx1 = cb, idx
    return recon, cb1, idx1


def _entropy_bits(idx: np.ndarray, k: int) -> float:
    counts = np.bincount(idx, minlength=k).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _block(flat: np.ndarray, rows: int, cols: int) -> list[list[float]]:
    """First rows*cols values normalized to [-1, 1] by peak magnitude -> 2D grid."""
    n = rows * cols
    seg = flat[:n]
    if seg.size < n:
        seg = np.pad(seg, (0, n - seg.size))
    peak = float(np.abs(seg).max()) or 1.0
    return (seg.reshape(rows, cols) / peak).round(4).tolist()


def probe_tensor(flat: np.ndarray, group_size: int = 8) -> dict:
    flat = np.ascontiguousarray(flat, dtype=np.float32).reshape(-1)
    mean, std = float(flat.mean()), float(flat.std())
    lo, hi = (float(x) for x in np.percentile(flat, [0.5, 99.5]))
    counts, _ = np.histogram(flat, bins=33, range=(lo, hi))
    distribution = (counts / max(counts.max(), 1)).round(4).tolist()
    outlier_pct = float((np.abs(flat - mean) > 3 * std).mean() * 100)

    pad = (-flat.size) % group_size
    vecs = (np.pad(flat, (0, pad)) if pad else flat).reshape(-1, group_size)
    rng = np.random.default_rng(0)
    if len(vecs) > _MAX_VECS:
        vecs = vecs[rng.choice(len(vecs), _MAX_VECS, replace=False)]

    signal = float((vecs ** 2).mean()) or 1e-12
    rd = []
    for n in (1, 2, 3, 4):
        recon, _, _ = _rvq(vecs, n)
        mse = float(((vecs - recon) ** 2).mean())
        rd.append({"bpw": float(n), "sqnr": round(10 * np.log10(signal / max(mse, 1e-12)), 2)})

    recon3, cb1, idx1 = _rvq(vecs, 3)
    resid3 = vecs - recon3
    mse3 = float((resid3 ** 2).mean())
    sqnr3 = round(10 * np.log10(signal / max(mse3, 1e-12)), 2)
    error_pct = round(float(np.sqrt(mse3) / (np.abs(vecs).mean() + 1e-12)) * 100, 2)

    k1 = len(cb1)
    util_counts = np.sort(np.bincount(idx1, minlength=k1))[::-1].astype(np.float64)
    utilization = (util_counts[:26] / max(util_counts.max(), 1)).round(4).tolist()
    entropy_bits = round(_entropy_bits(idx1, k1), 2)

    # codebook centroid components as positions on the value axis (subsample for ticks)
    cb_vals = cb1.reshape(-1)
    cb_sample = cb_vals[rng.choice(len(cb_vals), min(64, len(cb_vals)), replace=False)]
    codebook_values = np.clip(cb_sample, lo, hi).round(5).tolist()

    # 3D PCA: weight vectors -> codebook centroids in a shared projection
    sub = vecs[rng.choice(len(vecs), min(len(vecs), 220), replace=False)]
    mu = sub.mean(0)
    _, _, vt = np.linalg.svd(sub - mu, full_matrices=False)
    comps = vt[:3]
    v3 = (sub - mu) @ comps.T
    c3 = (cb1[rng.choice(k1, min(48, k1), replace=False)] - mu) @ comps.T
    scale = float(np.abs(v3).max()) or 1.0
    vectors3d = (v3 / scale).round(4).tolist()
    centroids3d = (c3 / scale).round(4).tolist()

    return {
        "std": round(std, 5), "mean": round(mean, 5),
        "vmin": round(lo, 5), "vmax": round(hi, 5), "outlier_pct": round(outlier_pct, 3),
        "distribution": distribution, "dist_range": [round(lo, 5), round(hi, 5)],
        "codebook_values": codebook_values,
        "rd": rd,
        "utilization": utilization, "entropy_bits": entropy_bits, "entropy_max": 8.0,
        "weights_block": _block(flat, 11, 40),
        "error_block": _block(np.abs(resid3).reshape(-1), 7, 40),
        "sqnr_3bpw": sqnr3, "error_pct": error_pct,
        "vectors3d": vectors3d, "centroids3d": centroids3d,
    }
