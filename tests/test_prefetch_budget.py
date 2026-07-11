"""Prefetch byte backpressure: the queue caps candidate COUNT, the budget caps
candidate BYTES. Scheduling only - packed bytes must be unchanged."""
import threading
import time

import pytest

torch = pytest.importorskip("torch")

from orka.pipeline.pack_pipeline import PrefetchBudget  # noqa: E402


def test_reserve_blocks_until_release():
    b = PrefetchBudget(100)
    b.reserve(60)
    acquired = threading.Event()

    def second():
        b.reserve(60)
        acquired.set()

    t = threading.Thread(target=second, daemon=True)
    t.start()
    time.sleep(0.15)
    assert not acquired.is_set(), "second reserve must block over budget"
    b.release(60)
    assert acquired.wait(timeout=5.0), "release must unblock the waiter"
    assert b.outstanding == 60


def test_oversized_candidate_admitted_alone():
    b = PrefetchBudget(100)
    b.reserve(1000)  # must not deadlock: outstanding == 0 admits anything
    assert b.outstanding == 1000
    b.release(1000)
    assert b.outstanding == 0


def test_pack_bytes_unchanged_under_tiny_budget(monkeypatch, tmp_path):
    from orka.pipeline.pack import pack_checkpoint

    src = tmp_path / "model.pt"
    torch.manual_seed(11)
    torch.save(
        {"a.weight": torch.randn(32, 16), "b.weight": torch.randn(32, 16)}, src
    )

    def pack(out):
        return pack_checkpoint(
            source=src, out_dir=out, group_size=8, codebook_size=16,
            codebook_mode="per-tensor", backend="torch", device="cpu",
            normalization="slrq-block", sample_vectors=32, iterations=4,
            em_aq_passes=0,
        )

    m_default = pack(tmp_path / "default.orka")
    monkeypatch.setenv("ORKA_PREFETCH_BUDGET_GB", "0.000001")  # ~1KB: serializes all
    m_tiny = pack(tmp_path / "tiny.orka")

    for td, tt in zip(m_default["tensors"], m_tiny["tensors"]):
        assert td["name"] == tt["name"]
        assert td["mse"] == tt["mse"]
        assert td["sqnr"] == tt["sqnr"]
    d, t = tmp_path / "default.orka", tmp_path / "tiny.orka"
    for sub in sorted(p.relative_to(d) for p in d.rglob("*") if p.is_file()):
        if sub.name == "manifest.json":
            continue
        assert (d / sub).read_bytes() == (t / sub).read_bytes(), sub
