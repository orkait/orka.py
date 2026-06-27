"""Kaggle kernel: VQ-QAT 2bpw experiment, end to end.

Downloads SmolLM2-360M, builds a 2bpw measured allocation, fine-tunes the
quantized student with KL distillation (orka.qat_train), then runs the
four-way A/B (fp16 / 4bpw-style reference is the PTQ baseline computed here /
2bpw-PTQ / 2bpw-QAT) and writes results to /kaggle/working.

The orka package is shipped as a Kaggle dataset; the HF token comes from the
mounted hf-token-private dataset (never hardcoded).
"""

import json
import os
import sys
from pathlib import Path

REPO = os.environ.get("QAT_REPO", "HuggingFaceTB/SmolLM2-360M")
WORK = Path("/kaggle/working")
STEPS = int(os.environ.get("QAT_STEPS", "800"))
SEQ_LEN = int(os.environ.get("QAT_SEQ_LEN", "160"))
TARGET_BPW = float(os.environ.get("QAT_BPW", "2.0"))


def setup_orka() -> bool:
    base = Path("/kaggle/input")
    if not base.exists():
        return False
    # Kaggle's mount depth varies (/kaggle/input/<slug> or
    # /kaggle/input/datasets/<user>/<slug>), so search recursively for the
    # package marker rather than assume a level.
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


def main() -> int:
    if not setup_orka():
        print("ERROR: orka source not found", file=sys.stderr)
        return 1

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

    # --- training corpus + held-out eval (disjoint) ---
    print("--- building wikitext corpus ---", flush=True)
    corpus = WORK / "corpus.txt"
    eval_prompts = WORK / "eval.txt"
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        lines = [r["text"].strip() for r in ds if len(r["text"].strip()) > 200]
        corpus.write_text("\n".join(lines[:900]))
        eval_prompts.write_text("\n".join(lines[950:962]))
    except Exception as exc:
        print(f"wikitext fetch failed ({exc}); using fallback", flush=True)
        fb = ["The history of science is a long and storied one." * 4] * 64
        corpus.write_text("\n".join(fb))
        eval_prompts.write_text("\n".join(fb[:11]))

    # --- 2bpw measured allocation ---
    print(f"--- allocate {TARGET_BPW} bpw ---", flush=True)
    from orka.allocate import build_allocation
    alloc = build_allocation(
        src, TARGET_BPW, candidate_specs=("vq-8", "vq-12", "rvq-12-4", "rvq-8-8", "rvq-12-8"),
        group_size=8, sample_vectors=8192, iterations=4, backend="torch", device="cuda",
    )
    alloc_path = WORK / "alloc2bpw.json"
    alloc_path.write_text(json.dumps(alloc, indent=2))
    print(f"  achieved {alloc['achieved_bpw']:.3f} bpw", flush=True)

    # --- 2bpw PTQ baseline (the broken one, for the A/B floor) ---
    print("--- pack 2bpw PTQ baseline ---", flush=True)
    from orka.pipeline.pack import pack_checkpoint
    from orka.allocate import allocation_tensor_stages
    ptq_art = WORK / "ptq.orka"
    pack_checkpoint(
        source=src, out_dir=ptq_art, group_size=8, codebook_size=4096,
        codebook_mode="per-tensor", backend="torch", device="cuda",
        normalization="slrq-block", outlier_frac=0.005, sample_vectors=65536,
        iterations=6, em_aq_passes=1,
        tensor_stages_map=allocation_tensor_stages(alloc),
    )
    from orka.artifact.export import export_vllm
    ptq_hf = WORK / "ptq-hf"
    export_vllm(ptq_art, ptq_hf, model_dir=Path(model_dir), dtype="bfloat16")

    # --- QAT training ---
    print(f"--- QAT train {STEPS} steps ---", flush=True)
    qat_hf = WORK / "qat-hf"
    sys.argv = [
        "qat_train", str(model_dir), str(alloc_path), str(corpus), str(qat_hf),
        "--steps", str(STEPS), "--seq-len", str(SEQ_LEN), "--batch", "2",
        "--lr", "3e-4", "--commit", "0.25", "--cb-weight", "0.5", "--device", "cuda",
        "--max-seqs", "700",
    ]
    from orka.qat_train import main as qat_main
    qat_main()

    # --- four-way uniform eval (KL vs fp16 + top-1) + generation ---
    print("--- eval ---", flush=True)
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    prompts = [l.strip() for l in eval_prompts.read_text().splitlines() if l.strip()]
    tk = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    def logits(d):
        m = AutoModelForCausalLM.from_pretrained(d, local_files_only=True, dtype=torch.float32).cuda().eval()
        out = []
        with torch.no_grad():
            for p in prompts:
                ids = tk(p, return_tensors="pt", truncation=True, max_length=192).input_ids.cuda()
                out.append(m(ids).logits.float().cpu())
        del m; torch.cuda.empty_cache(); return out

    ref = logits(model_dir)
    results = {"fp16": {"kl": 0.0, "top1": 1.0}}
    for tag, d in (("2bpw-PTQ", ptq_hf), ("2bpw-QAT", qat_hf)):
        lg = logits(d)
        kl = tot = match = 0.0
        for r, l in zip(ref, lg):
            kl += F.kl_div(F.log_softmax(l, -1), F.softmax(r, -1), reduction="sum").item()
            match += (r.argmax(-1) == l.argmax(-1)).sum().item(); tot += r.shape[1]
        results[tag] = {"kl": kl / tot, "top1": match / tot}
        print(f"  {tag}: KL {kl/tot:.4f}  top1 {match/tot:.4f}", flush=True)

    gens = {}
    gp = ["The capital of France is", "The chemical symbol for gold is",
          "Photosynthesis is the process by which"]
    for tag, d in (("fp16", model_dir), ("2bpw-PTQ", ptq_hf), ("2bpw-QAT", qat_hf)):
        m = AutoModelForCausalLM.from_pretrained(d, local_files_only=True, dtype=torch.bfloat16).cuda().eval()
        outs = []
        with torch.no_grad():
            for p in gp:
                ids = {k: v.cuda() for k, v in tk(p, return_tensors="pt").items()}
                o = m.generate(**ids, max_new_tokens=24, do_sample=False, pad_token_id=tk.eos_token_id)
                outs.append(tk.decode(o[0], skip_special_tokens=True))
        gens[tag] = outs; del m; torch.cuda.empty_cache()

    report = {"steps": STEPS, "achieved_bpw": alloc["achieved_bpw"],
              "metrics": results, "generations": gens, "prompts": gp}
    (WORK / "qat_report.json").write_text(json.dumps(report, indent=2))
    print("=== QAT REPORT ===", flush=True)
    print(json.dumps(report["metrics"], indent=2), flush=True)
    for i, p in enumerate(gp):
        print(f"\n[{i+1}] {p}", flush=True)
        for tag in ("fp16", "2bpw-PTQ", "2bpw-QAT"):
            print(f"  {tag}: {gens[tag][i][len(p):].strip()[:100]}", flush=True)
    print("=== QAT KAGGLE DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
