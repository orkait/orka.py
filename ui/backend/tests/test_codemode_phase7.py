"""Gates for the phase-7 UI-backend fixes: progress fan-out, replay for late/reconnecting
clients, bounded shutdown drain, honest trick gating, and the config/shape memo."""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from ui.backend import app as appmod
from ui.backend import fetch as fetchmod
from ui.backend.jobs import END, JobQueue


# ----------------------------------------------------------------- progress fan-out
@pytest.mark.asyncio
async def test_two_subscribers_each_receive_every_event():
    q = JobQueue()
    await q.start()
    gate = asyncio.Event()

    async def runner(job_id, emit):
        emit({"stage": "a"})
        await gate.wait()
        emit({"stage": "b"})
        return {"ok": True}

    job_id = q.submit(runner)
    await asyncio.sleep(0.02)                      # let the worker emit "a"
    s1 = q.job(job_id).subscribe()
    s2 = q.job(job_id).subscribe()
    gate.set()
    await q.wait(job_id)

    def drain(sub):
        out = []
        while not sub.empty():
            out.append(sub.get_nowait())
        return out

    got1, got2 = drain(s1), drain(s2)
    assert [e["stage"] for e in got1] == ["a", "b", "_end"]
    assert got1 == got2, "a shared queue split events between clients; each needs its own"
    await q.stop()


@pytest.mark.asyncio
async def test_late_subscriber_replays_history_then_terminates():
    q = JobQueue()
    await q.start()

    async def runner(job_id, emit):
        emit({"stage": "pack"})
        return {"ok": True}

    job_id = q.submit(runner)
    await q.wait(job_id)
    sub = q.job(job_id).subscribe()               # subscribing AFTER the job settled
    events = []
    while not sub.empty():
        events.append(sub.get_nowait())
    assert events == [{"stage": "pack"}, END], "late subscriber must not block forever"
    await q.stop()


def test_queue_survives_an_app_restart(monkeypatch):
    """``_queue`` is module-level, so it outlives any single app lifecycle. Work submitted
    after a restart must reach the NEW worker: while the pending FIFO was an asyncio.Queue
    created once, its waiter future belonged to the first loop, so a post-restart
    put_nowait woke a dead loop and the job sat in "queued" forever."""
    async def fake_run_live(model, job_id, emit):
        emit({"stage": "pack", "msg": "x"})
        from ui.backend.tests.test_live import _static_journey
        return _static_journey()

    monkeypatch.setattr(appmod, "run_live", fake_run_live)
    for cycle in range(2):
        with TestClient(appmod.app) as c:          # each `with` is a full startup/shutdown
            job_id = c.post("/pack", json={"model": "x/y"}).json()["job_id"]
            status = "queued"
            for _ in range(200):
                status = c.get(f"/jobs/{job_id}").json()["status"]
                if status in ("done", "failed"):
                    break
            assert status == "done", f"cycle {cycle}: job stuck in {status!r}"


def test_stream_endpoint_terminates_for_a_finished_job(monkeypatch):
    async def fake_run_live(model, job_id, emit):
        emit({"stage": "pack", "msg": "x"})
        from ui.backend.tests.test_live import _static_journey
        return _static_journey()

    monkeypatch.setattr(appmod, "run_live", fake_run_live)
    with TestClient(appmod.app) as c:
        job_id = c.post("/pack", json={"model": "x/y"}).json()["job_id"]
        for _ in range(200):
            if c.get(f"/jobs/{job_id}").json()["status"] in ("done", "failed"):
                break
        # Both a first and a *second* read must complete rather than hang on an empty bus.
        for _ in range(2):
            body = c.get(f"/jobs/{job_id}/stream").text
            assert "event: progress" in body
            assert "event: result" in body


# -------------------------------------------------------------- shutdown drain
@pytest.mark.asyncio
async def test_stop_settles_a_still_running_job():
    q = JobQueue()
    await q.start()
    never = asyncio.Event()

    async def runner(job_id, emit):
        emit({"stage": "start"})
        await never.wait()

    job_id = q.submit(runner)
    await asyncio.sleep(0.02)
    assert q.status(job_id) == "running"
    await q.stop(drain_timeout=0.05)
    job = q.job(job_id)
    assert job.status == "failed"
    assert "shut down" in (job.error or ""), "an abandoned job must not stay 'running'"
    assert job.done.is_set()


# ------------------------------------------------------------ honest trick gating
@pytest.mark.parametrize("tied,expected", [(True, True), (False, False)])
def test_keep_head_trick_applies_only_when_tied(tied, expected):
    from ui.backend.pipeline_steps import build_tricks
    from ui.backend.schema import Architecture

    arch = Architecture(arch_class="dense",
                        flags={"tied_head": tied, "has_moe": False, "has_ssm": False},
                        param_breakdown=[], layers=[])
    trick = {t.id: t for t in build_tricks(arch)}["keep_head_fp16"]
    assert trick.applies is expected
    assert trick.default is expected


def test_error_comp_warns_on_ssm_models():
    from ui.backend.pipeline_steps import build_tricks
    from ui.backend.schema import Architecture

    arch = Architecture(arch_class="hybrid",
                        flags={"tied_head": False, "has_moe": False, "has_ssm": True},
                        param_breakdown=[], layers=[])
    tricks = {t.id: t for t in build_tricks(arch)}
    assert tricks["error_comp"].warn is not None
    assert tricks["lattice"].warn is not None


# ------------------------------------------------------------------ config memo
def test_config_and_shapes_are_memoized_per_model(monkeypatch):
    calls = {"n": 0}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"vocab_size": 42}

    def fake_get(url, **kw):
        calls["n"] += 1
        return _Resp()

    fetchmod.clear_caches()
    monkeypatch.setattr(fetchmod.httpx, "get", fake_get)
    first = fetchmod.fetch_config("owner/model")
    second = fetchmod.fetch_config("owner/model")
    assert first == second == {"vocab_size": 42}
    assert calls["n"] == 1, "the bpw sweep must not refetch config per knob change"

    first["vocab_size"] = 0                       # caller mutation must not poison the memo
    assert fetchmod.fetch_config("owner/model") == {"vocab_size": 42}

    fetchmod.clear_caches()
    fetchmod.fetch_config("owner/model")
    assert calls["n"] == 2
