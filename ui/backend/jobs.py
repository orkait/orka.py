"""Single-GPU serial job queue. One worker drains a FIFO so two GPU jobs never run at once
(the orka crash lesson).

Progress is fanned out, not consumed: every subscriber gets its own queue seeded with the
events already emitted. A single shared queue meant two SSE clients split one event stream
between them, and a client that reconnected after the terminal event blocked forever on an
empty queue."""
from __future__ import annotations

import asyncio
import uuid
from collections import deque
from typing import Any, Awaitable, Callable

Runner = Callable[..., Awaitable[Any]]

#: Terminal marker appended to every subscriber's queue when a job settles.
END = {"stage": "_end"}


class _Job:
    def __init__(self, runner: Runner, kwargs: dict):
        self.id = uuid.uuid4().hex[:12]
        self.runner = runner
        self.kwargs = kwargs
        self.status = "queued"
        self.result: Any = None
        self.error: str | None = None
        self.log: list[dict] = []
        self._subscribers: list[asyncio.Queue] = []
        self.done = asyncio.Event()

    def subscribe(self) -> asyncio.Queue:
        """A queue replaying everything emitted so far, then tailing live events. A job
        that already settled yields its history plus END, so a late or reconnecting
        client terminates instead of hanging."""
        q: asyncio.Queue = asyncio.Queue()
        for ev in self.log:
            q.put_nowait(ev)
        if self.done.is_set():
            q.put_nowait(END)
        else:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def emit(self, ev: dict) -> None:
        self.log.append(ev)
        for q in list(self._subscribers):
            q.put_nowait(ev)

    def settle(self) -> None:
        for q in list(self._subscribers):
            q.put_nowait(END)
        self._subscribers.clear()
        self.done.set()


class JobQueue:
    """FIFO of pending job ids + one worker.

    The pending list is a plain deque and the wake-up Event is rebuilt by ``start()``.
    An ``asyncio.Queue`` held across lifecycles does not survive a loop swap: its waiter
    futures belong to the loop that created them, so after a restart (a second app
    startup, or a second TestClient in one test session) ``put_nowait`` woke a future on
    the dead loop and the new worker never saw the job - it sat in "queued" forever.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, _Job] = {}
        self._pending: deque[str] = deque()
        self._wake: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker: asyncio.Task | None = None
        self._current: _Job | None = None

    async def start(self) -> None:
        if self._worker is None:
            self._loop = asyncio.get_running_loop()
            self._wake = asyncio.Event()
            if self._pending:
                self._wake.set()          # work submitted before this lifecycle started
            self._worker = asyncio.create_task(self._run())

    def _signal(self) -> None:
        """Wake the worker from either the loop thread or a threadpool worker (FastAPI runs
        sync endpoints off-loop, where touching an Event directly would race)."""
        wake, loop = self._wake, self._loop
        if wake is None or loop is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            wake.set()
        else:
            loop.call_soon_threadsafe(wake.set)

    async def stop(self, drain_timeout: float = 5.0) -> None:
        """Stop accepting work. An in-flight job's GPU work runs in a worker thread and
        cannot be cancelled, so give it a bounded chance to settle before dropping the
        supervising task - otherwise the job silently stays 'running' forever."""
        current = self._current
        if current is not None and not current.done.is_set():
            try:
                await asyncio.wait_for(current.done.wait(), timeout=drain_timeout)
            except asyncio.TimeoutError:
                current.status = "failed"
                current.error = "server shut down while the job was still running"
                current.settle()
        if self._worker:
            self._worker.cancel()
            self._worker = None
        self._current = None
        self._wake = None
        self._loop = None

    def submit(self, runner: Runner, **kwargs) -> str:
        job = _Job(runner, kwargs)
        self._jobs[job.id] = job
        self._pending.append(job.id)
        self._signal()
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
            if not self._pending:
                await self._wake.wait()
                self._wake.clear()
                continue
            job_id = self._pending.popleft()
            job = self._jobs[job_id]
            self._current = job
            job.status = "running"
            try:
                job.result = await job.runner(job.id, job.emit, **job.kwargs)
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - surface to caller, never crash worker
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = "failed"
            finally:
                job.settle()
                self._current = None
