import os
os.environ.setdefault("QAT_REPO", "MerlinSafety/HybridIntelligence-0.5B")
os.environ.setdefault("QAT_BPW", "4.0")
os.environ.setdefault("QAT_STEPS", "600")
"""Kaggle kernel: VQ-QAT 4bpw on HybridIntelligence-0.5B (Falcon H1), end to end.

Builds a 4bpw measured allocation, packs the PTQ baseline, fine-tunes the
quantized student with KL distillation (orka.qat_train), then runs a sliding
window wikitext-2 perplexity eval for fp16 / 4bpw-PTQ / 4bpw-QAT plus KL/top1
and generation. Goal: see how far QAT closes the PTQ gap vs llama.cpp Q4_K_M.

QAT configs follow the blessed Supra recipe: lr 3e-4, commit 0.25, cb-weight
0.5, seq-len 160, batch 2. The orka package ships as a Kaggle dataset; the HF
token comes from the mounted hf-token-private dataset.
"""

import json
import math
import os
import sys
from pathlib import Path

REPO = os.environ.get("QAT_REPO", "MerlinSafety/HybridIntelligence-0.5B")
WORK = Path("/kaggle/working")
STEPS = int(os.environ.get("QAT_STEPS", "600"))
SEQ_LEN = int(os.environ.get("QAT_SEQ_LEN", "160"))
TARGET_BPW = float(os.environ.get("QAT_BPW", "4.0"))
PPL_CTX = int(os.environ.get("QAT_PPL_CTX", "512"))
PPL_MAXTOK = int(os.environ.get("QAT_PPL_MAXTOK", "25600"))
CANDIDATES = ("vq-12", "rvq-12-8", "rvq-12-12", "rvq-12-12-8", "rvq-12-12-12")


def setup_orka() -> bool:
    base = Path("/kaggle/input")
    if not base.exists():
        return False
    for marker in base.rglob("orka/__init__.py"):
        pkg_parent = marker.parent.parent
        sys.path.insert(0, str(pkg_parent))
        print(f"orka package found at {pkg_parent}", flush=True)
        return True
    for zp in base.rglob("*.zip"):
        import shutil
        import zipfile
        ext = Path("/tmp/orka_extracted")
        if ext.exists():
            shutil.rmtree(ext)
        ext.mkdir(parents=True)
        with zipfile.ZipFile(zp, "r") as z:
            z.extractall(ext)
        if (ext / "orka" / "__init__.py").exists():
            sys.path.insert(0, str(ext))
            print(f"orka package extracted from {zp}", flush=True)
            return True
    print("=== /kaggle/input tree (no orka found) ===", flush=True)
    for p in list(base.rglob("*"))[:40]:
        print(f"  {p}", flush=True)
    return False


def _wikitext_ppl(model_dir, text, ctx, maxtok):
    """Sliding window (non-overlapping) perplexity over wikitext-2 test."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    ids = tok(text, return_tensors="pt").input_ids[0][:maxtok]
    m = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float32
    ).cuda().eval()
    nll = 0.0
    ntok = 0
    with torch.no_grad():
        for i in range(0, len(ids) - 1, ctx):
            chunk = ids[i:i + ctx].unsqueeze(0).cuda()
            if chunk.shape[1] < 2:
                break
            out = m(chunk, labels=chunk)
            nll += out.loss.item() * (chunk.shape[1] - 1)
            ntok += chunk.shape[1] - 1
    del m
    torch.cuda.empty_cache()
    return math.exp(nll / ntok), ntok


def main() -> int:
    if not setup_orka():
        print("ERROR: orka source not found", file=sys.stderr)
        return 1

    # 8-bit Adam (bitsandbytes) cuts optimizer m+v 4.2GB -> 1GB so QAT fits a
    # 16GB T4. Install quietly; --optim8bit below depends on it.
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=False)

    import os as _os
    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    print(f"=== GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'} ===", flush=True)

    from orka.deploy.kaggle import _load_hf_token
    from huggingface_hub import login, snapshot_download
    tok = _load_hf_token()
    if tok:
        login(token=tok)

    print("--- downloading model ---", flush=True)
    model_dir = snapshot_download(
        REPO, allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*", "merges*", "vocab*"]
    )
    src = next(Path(model_dir).glob("*.safetensors"))

    # --- train corpus (train split) + held-out eval (test split, disjoint) ---
    print("--- building wikitext corpus ---", flush=True)
    corpus = WORK / "corpus.txt"
    eval_prompts = WORK / "eval.txt"
    ppl_text = ""
    try:
        from datasets import load_dataset
        train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        tlines = [r["text"].strip() for r in train if len(r["text"].strip()) > 200]
        corpus.write_text("\n".join(tlines[:900]))
        test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        ppl_text = "\n\n".join(t for t in (r["text"] for r in test) if t.strip())
        tst = [r["text"].strip() for r in test if len(r["text"].strip()) > 200]
        eval_prompts.write_text("\n".join(tst[:12]))
    except Exception as exc:
        print(f"wikitext fetch failed ({exc}); using fallback", flush=True)
        fb = ["The history of science is a long and storied one." * 4] * 64
        corpus.write_text("\n".join(fb))
        eval_prompts.write_text("\n".join(fb[:11]))
        ppl_text = "\n".join(fb)

    # --- 4bpw measured allocation ---
    print(f"--- allocate {TARGET_BPW} bpw ---", flush=True)
    from orka.quant.allocate import build_allocation, allocation_tensor_stages
    alloc = build_allocation(
        src, TARGET_BPW, candidate_specs=CANDIDATES,
        group_size=8, sample_vectors=8192, iterations=4, backend="torch", device="cuda",
    )
    alloc_path = WORK / "alloc4bpw.json"
    alloc_path.write_text(json.dumps(alloc, indent=2))
    print(f"  achieved {alloc['achieved_bpw']:.3f} bpw", flush=True)

    # --- 4bpw PTQ baseline ---
    print("--- pack 4bpw PTQ baseline ---", flush=True)
    from orka.pipeline.pack import pack_checkpoint
    ptq_art = WORK / "ptq.orka"
    pack_checkpoint(
        source=src, out_dir=ptq_art, group_size=8, codebook_size=4096,
        codebook_mode="per-tensor", backend="torch", device="cuda",
        normalization="slrq-block", outlier_frac=0.005, sample_vectors=65536,
        iterations=8, em_aq_passes=1,
        tensor_stages_map=allocation_tensor_stages(alloc),
    )
    from orka.artifact.export import export_vllm
    ptq_hf = WORK / "ptq-hf"
    export_vllm(ptq_art, ptq_hf, model_dir=Path(model_dir), dtype="bfloat16")

    # --- QAT training ---
    # Config: the lr that converged on the H100 (3e-4 diverges on Falcon H1's
    # SSM). --checkpoint-quantize frees the ~8GB straight-through graph so the
    # full run fits a 16GB P100/T4 at batch 2 + grad-accum 4 (effective 8).
    print(f"--- QAT train {STEPS} steps ---", flush=True)
    qat_hf = WORK / "qat-hf"
    # Proven-fitting config (3060 fit at 10.2GB; T4 has more room): all three
    # memory cuts - checkpoint quantize (~8GB graph), 8-bit Adam (~3GB), bf16
    # backbone (~1GB) - at batch 1 + grad-accum 8 (effective batch 8).
    sys.argv = [
        "qat_train", str(model_dir), str(alloc_path), str(corpus), str(qat_hf),
        "--steps", str(STEPS), "--seq-len", "256", "--batch", "1", "--grad-accum", "8",
        "--lr", "1e-4", "--commit", "0.5", "--cb-weight", "0.5", "--device", "cuda",
        "--checkpoint-quantize", "--optim8bit", "--student-bf16", "--max-seqs", "1200",
    ]
    import traceback
    from orka.qat_train import main as qat_main
    try:
        qat_main()
    except BaseException:
        print("=== QAT TRAIN FAILED ===", flush=True)
        traceback.print_exc()
        (WORK / "qat_error.txt").write_text(traceback.format_exc())
        raise

    # Training only - no benchmark. The 3-model wikitext PPL eval is what blew
    # past Kaggle's 9h GPU limit on the T4. Deliverable is the trained qat-hf
    # model; PPL eval runs cheaply afterward (plain inference) on local hardware.
    report = {"repo": REPO, "steps": STEPS, "achieved_bpw": alloc["achieved_bpw"],
              "qat_hf": str(qat_hf), "ptq_hf": str(ptq_hf), "eval": "skipped (run locally)"}
    (WORK / "qat_report.json").write_text(json.dumps(report, indent=2))
    print("=== QAT TRAIN DONE (no eval) ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print(f"qat-hf saved -> {qat_hf}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
