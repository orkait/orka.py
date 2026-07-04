import asyncio

import pytest

from ui.backend.jobs import JobQueue


@pytest.mark.asyncio
async def test_serial_execution_and_progress():
    q = JobQueue()
    await q.start()
    order = []

    async def runner(job_id, emit, *, tag):
        emit({"stage": "begin", "tag": tag})
        await asyncio.sleep(0.01)
        order.append(tag)
        emit({"stage": "done", "tag": tag})
        return {"tag": tag}

    id1 = q.submit(runner, tag="a")
    id2 = q.submit(runner, tag="b")
    r1 = await q.wait(id1)
    r2 = await q.wait(id2)
    assert r1["tag"] == "a" and r2["tag"] == "b"
    assert order == ["a", "b"]                 # serial, in submit order
    assert q.status(id1) == "done"
    await q.stop()
