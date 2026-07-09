"""Kaggle kernel entry for Orka pack.

Thin bootstrap: locate the orka package shipped as a Kaggle dataset, then hand
off to the standard CLI. All run parameters come from orka.deploy.kaggle._KAGGLE_CONFIG
via bootstrap_argv. The HF token is read from the mounted hf-token-private dataset
by orka.deploy.kaggle._load_hf_token (never hardcoded here).
"""

import sys
from pathlib import Path


def setup_orka() -> bool:
    """Add the orka package (shipped as a Kaggle dataset) to sys.path."""
    input_base = Path("/kaggle/input")
    if not input_base.exists():
        return False
    for ds_dir in input_base.iterdir():
        if not ds_dir.is_dir():
            continue
        if (ds_dir / "orka").is_dir():
            sys.path.insert(0, str(ds_dir))
            return True
        for zip_path in ds_dir.glob("*.zip"):
            import shutil
            import zipfile

            extract_dir = Path("/tmp/orka_extracted")
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)
            sys.path.insert(0, str(extract_dir))
            return True
    return False


if __name__ == "__main__":
    if not setup_orka():
        print("ERROR: orka source not found under /kaggle/input", file=sys.stderr)
        sys.exit(1)

    # GPU probe: report device count up front so dual-GPU availability is
    # visible in the kernel log immediately, before any packing starts.
    try:
        import torch

        n = torch.cuda.device_count()
        print(f"=== ORKA GPU PROBE: torch.cuda.device_count()={n} ===", flush=True)
        for i in range(n):
            print(f"    cuda:{i} = {torch.cuda.get_device_name(i)}", flush=True)
        if n < 1:
            print("WARNING: no GPUs visible. Check accelerator setting.", flush=True)
        elif n < 2:
            print(
                "INFO: single GPU detected; running in non-partitioned mode.",
                flush=True,
            )
    except Exception as exc:
        print(f"GPU probe failed: {exc}", flush=True)

    from huggingface_hub import login

    from orka.deploy.kaggle import _load_hf_token

    token = _load_hf_token()
    if token:
        login(token=token)

    from orka.cli import main

    if Path("/kaggle/working").exists():
        from orka.deploy.kaggle import bootstrap_argv

        bootstrap_argv(sys.argv)

    sys.exit(main())
