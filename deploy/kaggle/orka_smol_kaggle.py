import os
import sys
import gc
import json
import torch
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

def setup_orka():
    input_base = Path("/kaggle/input")
    if not input_base.exists(): return False

    for ds_dir in input_base.iterdir():
        if not ds_dir.is_dir(): continue

        # 1. Look for automatically unzipped folder (Kaggle default behavior)
        if (ds_dir / "orka").is_dir():
            print(f"Found Orka source folder in {ds_dir}")
            sys.path.insert(0, str(ds_dir))
            return True

        # 2. Look for zip file (fallback)
        for zip_path in ds_dir.glob("*.zip"):
            print(f"Extracting Orka core from {zip_path}...")
            import zipfile
            extract_dir = Path("/tmp/orka_extracted")
            if extract_dir.exists(): shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)
            sys.path.insert(0, str(extract_dir))
            return True

    print("WARNING: Could not find orka source zip or folder in /kaggle/input")
    return False

# Setup Orka path before importing
setup_orka()
sys.path.insert(0, "/kaggle/working") # Fallback

def get_device_map():
    """Optimize for Kaggle Dual T4 (16GB x 2)"""
    if torch.cuda.device_count() >= 2:
        print("Detected Dual GPU. Distributing model across cuda:0 and cuda:1")
        return "auto"
    return "cuda:0"

def run_smol_compression():
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    working_dir = Path("/kaggle/working/smol-orka")
    working_dir.mkdir(parents=True, exist_ok=True)

    # --- Hugging Face Authentication ---
    print("Authenticating with Hugging Face Hub...")
    from huggingface_hub import login
    login(token="***REMOVED-HF-TOKEN***")

    print(f"Downloading {model_id}...")
    # Use HF Hub to download only the safetensors to save Kaggle disk space
    model_path = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.safetensors", "*.json"],
        local_dir=str(working_dir / "hf_weights")
    )

    print("Initializing Orka pack_checkpoint for RVQ-Mixed SLRQ...")
    from orka.pipeline.pack import pack_checkpoint
    from orka.quant.spec import rvq_mixed_family_stages

    out_dir = working_dir / "compressed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Starting Aggressive Sweep (pack_checkpoint). This may take ~10-20 mins for SmolLM2...")
    try:
        manifest = pack_checkpoint(
            source=Path(model_path),
            out_dir=out_dir,
            group_size=64,
            iterations=80,  # Reduced from 150 - early stopping catches convergence
            codebook_mode="per-tensor", # Required for family_stages_map
            family_stages_map=rvq_mixed_family_stages(),
            normalization="slrq-block",
            backend="torch",
            device=get_device_map(),
            outlier_frac=0.005, # Protect top 0.5% of outlier weights
            em_aq_passes=3, # Full multi-stage refinement for max quality
            slrq_salient=True,
            sample_vectors=500_000, # Subsample for codebook training (all vectors still get quantized)
        )
        manifest_path = out_dir / "manifest.json"
        print(f"Compression Complete! Manifest saved to {manifest_path}")
    except torch.cuda.OutOfMemoryError:
        print("OOM Detected! Kaggle T4 limit reached.")
        torch.cuda.empty_cache()
        gc.collect()
        sys.exit(1)

    print("Packaging to GGUF...")
    print(f"\n--- SUCCESS ---")
    print(f"Output saved to: {out_dir}")

if __name__ == "__main__":
    run_smol_compression()
