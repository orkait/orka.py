"""Live GPU path: pack the model with the validated config + eval, return a MEASURED
journey (same schema as static). The heavy GPU work runs in a thread (asyncio.to_thread) so
the event loop stays responsive; progress is emitted via the job's emit callback.

trusted/trust_reason ride on the reliable-eval hardening; until that lands eval may report
trusted=None and the UI shows 'unverified' rather than a false number."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from .journey import build_static_journey
from .schema import Journey, Result
from .settings import GPU_MEM_CAP_GB


def _pack_and_eval(model: str, emit) -> dict:
    """Blocking GPU work. Imports orka lazily so the module loads without torch present."""
    from huggingface_hub import snapshot_download
    from orka.eval import eval_artifact
    from orka.pipeline.pack import pack_checkpoint

    emit({"stage": "download", "msg": f"resolving {model}"})
    snap = Path(snapshot_download(model))

    with tempfile.TemporaryDirectory() as tmp:
        art = Path(tmp) / "art"
        prompts = Path(tmp) / "prompts.txt"
        prompts.write_text("The capital of France is Paris.\nWater boils at 100 C.\n")
        emit({"stage": "pack", "msg": "rvq-12-12 + em-aq + hessian + auto keep-head"})
        pack_checkpoint(
            snap, out_dir=art, codebook_sizes=[4096, 4096], em_aq_passes=3,
            keep_head_fp16="auto", awq_model_dir=snap, awq_calibration=prompts,
            max_gpu_mem_gb=GPU_MEM_CAP_GB, backend="torch", device="cuda",
        )
        emit({"stage": "eval", "msg": "reconstruct + perplexity"})
        out = Path(tmp) / "eval.json"
        res = eval_artifact(art, prompts, out, device="cuda")
        fp16_mb = sum(f.stat().st_size for f in snap.glob("*.safetensors")) / 1e6
        orka_mb = sum(f.stat().st_size for f in art.rglob("*") if f.is_file()) / 1e6
        return {
            "ratio": round(fp16_mb / max(orka_mb, 1e-9), 2), "fp16_mb": round(fp16_mb, 1),
            "orka_mb": round(orka_mb, 1),
            "ppl_base": res.get("original_perplexity"), "ppl_orka": res.get("orka_perplexity"),
            "ppl_ratio": (round(res["perplexity_ratio"], 3)
                          if res.get("perplexity_ratio") not in (None, float("inf")) else None),
            "trusted": res.get("trusted"), "trust_reason": res.get("trust_reason"),
        }


async def run_live(model: str, job_id: str, emit) -> Journey:
    base = build_static_journey(model)          # arch + pipeline + tricks (no GPU)
    measured = await asyncio.to_thread(_pack_and_eval, model, emit)
    base.result = Result(source="measured", bpw=3.0, notes=["measured: rvq-12-12+em-aq+hessian"],
                         **measured)
    return base
