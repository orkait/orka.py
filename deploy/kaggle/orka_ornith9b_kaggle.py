import os
os.environ.setdefault("ORNITH_REPO", "deepreinforce-ai/Ornith-1.0-9B")
os.environ.setdefault("ORNITH_BPW", "4.0")
"""Kaggle kernel: PTQ VQ-pack of Ornith-1.0-9B (Qwen3.5 multimodal), end to end.

Downloads Ornith-9B on Kaggle (fast HF mirror + token), builds a measured
per-tensor bit allocation, then packs. Design decisions come from a structural
trace of the checkpoint against orka.quant.arch:

  * untied head (tie_word_embeddings=false) -> keep_head_fp16="auto" resolves to
    OFF, so the 248k-vocab head IS quantized (needed to beat GGUF Q4_K_M, which
    also quantizes it). The RD allocator naturally spends more bits on the
    high-energy head/embed than on the mlp/attn linears.
  * linear_attn blocks (A_log / dt_bias / conv1d) are detected as recurrent by
    ArchProfile -> block-OBS error-comp is skipped on them, but their big
    in_proj_*/out_proj Linears still pack as plain RVQ. No special handling here.
  * vision tower (model.visual.*) is standard 2-D ViT weight - packs like any
    linear. Only patch_embed.proj (a 4-D/5-D conv) is passed through, by
    excluding it from only_tensors.

Output: a .orka artifact + HF-format export in /kaggle/working. Eval runs
locally afterwards (plain inference) - the 3-model PPL sweep is what blew past
Kaggle's GPU limit on the hi05b run, so this kernel packs only.

orka ships as the orka-compiler-core-v2 dataset; the HF token comes from the
mounted hf-token-private dataset.
"""

import json
import os
import sys
from pathlib import Path

REPO = os.environ.get("ORNITH_REPO", "deepreinforce-ai/Ornith-1.0-9B")
WORK = Path("/kaggle/working")
TARGET_BPW = float(os.environ.get("ORNITH_BPW", "4.0"))
# Scratch cache (Kaggle /kaggle/working is capped ~19.5GB; the 18.8GB source
# must land on the roomier ephemeral /kaggle/tmp).
CACHE = os.environ.get("ORNITH_CACHE", "/kaggle/tmp/hf")
# RVQ menu for the allocator's RD probe. Specs must have DISTINCT bits-per-vector
# (at g=8: vq-8=3, vq-12=4, rvq-12-8=7, rvq-12-12=8, rvq-12-12-12=12) - two specs
# at the same rate make the greedy allocator's marginal step divide by zero.
CANDIDATES = ("vq-8", "vq-12", "rvq-12-8", "rvq-12-12", "rvq-12-12-12")


def setup_orka() -> bool:
    """Find the orka package in a mounted dataset (dir or zipped)."""
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


def _quant_tensor_names(model_dir: Path) -> list[str]:
    """All quantizable 2-D+ weight names except the vision patch_embed conv,
    which is passed through. Read from the safetensors index weight_map."""
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    names = []
    for n in idx["weight_map"]:
        if not n.endswith(".weight"):
            continue
        if "patch_embed.proj" in n:      # 4-D/5-D conv - passthrough
            continue
        names.append(n)
    return names


def main() -> int:
    if not setup_orka():
        print("ERROR: orka source not found", file=sys.stderr)
        return 1

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HOME", CACHE)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import torch
    print(f"=== GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'} ===", flush=True)

    from orka.deploy.kaggle import _load_hf_token
    from huggingface_hub import login, snapshot_download
    tok = _load_hf_token()
    if tok:
        login(token=tok)

    print(f"--- downloading {REPO} -> {CACHE} ---", flush=True)
    model_dir = Path(snapshot_download(
        REPO, cache_dir=CACHE,
        allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*", "merges*", "vocab*", "*.jinja"],
    ))
    src = model_dir  # pack_checkpoint accepts a sharded-safetensors dir
    print(f"  model at {model_dir}", flush=True)

    only = _quant_tensor_names(model_dir)
    print(f"--- {len(only)} quantizable tensors (patch_embed conv passed through) ---", flush=True)

    # --- measured per-tensor allocation (GPU) ---
    print(f"--- allocate {TARGET_BPW} bpw ---", flush=True)
    from orka.quant.allocate import build_allocation, allocation_tensor_stages
    alloc = build_allocation(
        src, TARGET_BPW, candidate_specs=CANDIDATES,
        group_size=8, sample_vectors=8192, iterations=4,
        backend="torch", device="cuda",
    )
    alloc_path = WORK / "ornith_alloc.json"
    alloc_path.write_text(json.dumps(alloc, indent=2))
    print(f"  achieved {alloc['achieved_bpw']:.3f} bpw over {len(alloc['tensors'])} tensors", flush=True)

    # build_allocation probed 386 tensors on cuda and leaves them cached; free it
    # before pack or the leftover ~11GB + the 1B-param head OOMs the T4.
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # --- pack (PTQ) ---
    print("--- pack ---", flush=True)
    from orka.pipeline.pack import pack_checkpoint
    art = WORK / "ornith9b.orka"
    manifest = pack_checkpoint(
        source=src, out_dir=art, group_size=8, codebook_size=4096,
        codebook_mode="per-tensor", backend="torch", device="cuda",
        normalization="slrq-block", outlier_frac=0.0, sample_vectors=65536,
        iterations=8, em_aq_passes=1,
        only_tensors=only, only_tensors_passthrough=True,
        # The 1B-param embed+head OOM a 15GB T4 during pack (normalized 4GB +
        # vectors 4GB + v_res onload 3.8GB > 15GB). Passthrough them fp16 for a
        # complete pack; quantizing them (to beat GGUF) needs a streaming
        # giant-tensor fix or a >15GB GPU - tracked as follow-up.
        keep_head_fp16="on",
        tensor_stages_map=allocation_tensor_stages(alloc),
    )
    (WORK / "ornith_manifest.json").write_text(json.dumps(manifest, indent=2))

    # NO HF export here: reconstructing the ~18GB bf16 model into the ~20GB-capped
    # /kaggle/working blows the disk (and the load OOMs RAM). The .orka artifact IS
    # the deliverable; reconstruct to HF locally (no cap) via orka.artifact.export.

    def _dir_gb(p: Path) -> float:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9

    report = {
        "repo": REPO, "target_bpw": TARGET_BPW,
        "achieved_bpw": alloc["achieved_bpw"],
        "orka_gb": round(_dir_gb(art), 2),
        "gguf_q4km_gb": 5.63, "eval": "run locally", "hf_export": "run locally",
    }
    (WORK / "ornith_report.json").write_text(json.dumps(report, indent=2))
    print("=== PACK DONE ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
