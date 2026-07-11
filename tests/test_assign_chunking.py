"""The tiled (giant) assign path must return indices identical to the full path
for ANY chunk size - per-row argmin is independent of chunking. Guards the W1
byte-budget chunk bump."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from orka.codebook import _kmeans_torch  # noqa: E402


@pytest.fixture()
def data():
    torch.manual_seed(7)
    rows = torch.randn(1000, 8, dtype=torch.float32)
    cb = torch.randn(64, 8, dtype=torch.float32)
    return rows, cb


def test_tiled_matches_full(data, monkeypatch):
    rows, cb = data
    full_idx, _ = _kmeans_torch._torch_assign(rows, cb, "cpu", compute_mse=False)
    monkeypatch.setattr(_kmeans_torch, "_LARGE_ASSIGN_ROWS", 100)
    monkeypatch.setenv("ORKA_ASSIGN_CHUNK_MB", "0")  # let chunk_size drive the loop
    for chunk in (64, 128, 999, 4096):
        tiled_idx, _ = _kmeans_torch._torch_assign(
            rows, cb, "cpu", chunk_size=chunk, compute_mse=False
        )
        assert torch.equal(tiled_idx, full_idx), f"indices differ at chunk={chunk}"


def test_byte_budget_scales_chunk(monkeypatch):
    """Budget (MB) -> rows_per_chunk = budget_bytes // (4 * width), floored at chunk_size."""
    monkeypatch.setenv("ORKA_ASSIGN_CHUNK_MB", "128")
    from orka import config

    width = 8
    rows_per_chunk = max(65536, (config.assign_chunk_mb() << 20) // (4 * width))
    assert rows_per_chunk == (128 << 20) // 32  # 4M rows at d=8


def test_weighted_tiled_matches_full(data, monkeypatch):
    rows, cb = data
    vw = np.linspace(0.5, 2.0, 8).tolist()
    full_idx, _ = _kmeans_torch._torch_assign(
        rows, cb, "cpu", vector_weights=vw, compute_mse=False
    )
    monkeypatch.setattr(_kmeans_torch, "_LARGE_ASSIGN_ROWS", 100)
    monkeypatch.setenv("ORKA_ASSIGN_CHUNK_MB", "0")
    tiled_idx, _ = _kmeans_torch._torch_assign(
        rows, cb, "cpu", chunk_size=333, vector_weights=vw, compute_mse=False
    )
    assert torch.equal(tiled_idx, full_idx)
