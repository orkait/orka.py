from fastapi.testclient import TestClient

from ui.backend import app as appmod


def test_pack_enqueues_and_completes(monkeypatch):
    async def fake_run_live(model, job_id, emit):
        emit({"stage": "pack", "msg": "x"})
        from ui.backend.tests.test_live import _static_journey
        j = _static_journey()
        j.result.source = "measured"
        return j
    monkeypatch.setattr(appmod, "run_live", fake_run_live)

    with TestClient(appmod.app) as c:               # triggers startup (queue.start)
        r = c.post("/pack", json={"model": "x/y"})
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        s = {}
        for _ in range(200):
            s = c.get(f"/jobs/{job_id}").json()
            if s["status"] in ("done", "failed"):
                break
        assert s["status"] == "done"
        assert s["journey"]["result"]["source"] == "measured"
