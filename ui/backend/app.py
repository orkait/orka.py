"""FastAPI routes for the analysis engine.

/analyze is the instant static path (estimated numbers). /pack enqueues a live GPU job on a
single-worker serial queue; /jobs/{id} polls it and /jobs/{id}/stream streams SSE progress
then the measured journey."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from . import journey
from .jobs import JobQueue
from .live import run_live

_queue = JobQueue()


@asynccontextmanager
async def _lifespan(_app):
    await _queue.start()
    yield
    await _queue.stop()


app = FastAPI(title="orka compression-journey analysis engine", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/analyze")
def analyze(model: str = Query(...), bpw: float = 3.0,
            keep_head: bool = True, lattice: bool = False):
    try:
        j = journey.build_static_journey(model, bpw=bpw, keep_head=keep_head, lattice=lattice)
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code if exc.response is not None else 502
        if code == 404:
            raise HTTPException(404, f"model not found: {model}")
        if code in (401, 403):
            raise HTTPException(403, "model gated/private - set HF_TOKEN")
        raise HTTPException(502, f"HF fetch failed ({code})")
    return j.model_dump()


class PackRequest(BaseModel):
    model: str


@app.post("/pack")
def pack(req: PackRequest):
    async def runner(job_id, emit, *, model):
        return await run_live(model, job_id, emit)
    job_id = _queue.submit(runner, model=req.model)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = _queue.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    body = {"status": job.status, "error": job.error}
    if job.status == "done":
        body["journey"] = job.result.model_dump()
    return body


@app.get("/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    job = _queue.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")

    async def gen():
        while True:
            ev = await job.events.get()
            if ev.get("stage") == "_end":
                final = (job.result.model_dump() if job.status == "done"
                         else {"error": job.error})
                yield {"event": "result", "data": json.dumps(final)}
                break
            yield {"event": "progress", "data": json.dumps(ev)}

    return EventSourceResponse(gen())
