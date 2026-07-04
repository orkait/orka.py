"""Single-GPU serial job queue. One worker drains a FIFO so two GPU jobs never run at once
(the orka crash lesson). Each job gets a progress bus (asyncio.Queue) that SSE drains."""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable

Runner = Callable[..., Awaitable[Any]]


class _Job:
    def __init__(self, runner: Runner, kwargs: dict):
        self.id = uuid.uuid4().hex[:12]
        self.runner = runner
        self.kwargs = kwargs
        self.status = "queued"
        self.result: Any = None
        self.error: str | None = None
        self.events: asyncio.Queue = asyncio.Queue()
        self.done = asyncio.Event()


class JobQueue:
    def __init__(self) -> None:
        self._jobs: dict[str, _Job] = {}
        self._fifo: asyncio.Queue = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            self._worker = None

    def submit(self, runner: Runner, **kwargs) -> str:
        job = _Job(runner, kwargs)
        self._jobs[job.id] = job
        self._fifo.put_nowait(job.id)
        return job.id

    def status(self, job_id: str) -> str:
        j = self._jobs.get(job_id)
        return j.status if j else "unknown"

    def job(self, job_id: str) -> _Job | None:
        return self._jobs.get(job_id)

    async def wait(self, job_id: str) -> Any:
        j = self._jobs[job_id]
        await j.done.wait()
        if j.error:
            raise RuntimeError(j.error)
        return j.result

    async def _run(self) -> None:
        while True:
            job_id = await self._fifo.get()
            job = self._jobs[job_id]
            job.status = "running"

            def emit(ev: dict, _q=job.events) -> None:
                _q.put_nowait(ev)

            try:
                job.result = await job.runner(job.id, emit, **job.kwargs)
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - surface to caller, never crash worker
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = "failed"
            finally:
                job.events.put_nowait({"stage": "_end"})
                job.done.set()
