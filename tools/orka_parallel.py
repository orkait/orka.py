import os
import sys
import json
import math
import shutil
import argparse
import subprocess
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser(description="Auto-Parallel Orka Orchestrator")
    parser.add_argument("source", help="Path to safetensors")
    parser.add_argument("--out", required=True, help="Output directory")
    # Capture all other args to pass through
    args, unknown = parser.parse_known_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"Only {num_gpus} GPU(s) detected. Running standard Orka pipeline...")
        cmd = [sys.executable, "-m", "orka", "pack", args.source, "--out", args.out] + unknown
        subprocess.run(cmd, check=True)
        sys.exit(0)

    print(f"Detected {num_gpus} GPUs. Entering Auto-Parallel Mode.")

    # 1. Get candidates
    print("Inspecting checkpoint to extract layers...")
    try:
        from orka._checkpoint import inspect_checkpoint
    except ImportError:
        # Fallback if run outside root
        sys.path.insert(0, str(Path(__file__).parent))
        from orka._checkpoint import inspect_checkpoint

    report = inspect_checkpoint(Path(args.source))
    candidates = [t["name"] for t in report["tensors"] if t["candidate"]]

    if not candidates:
        print("No candidates found.")
        sys.exit(1)

    # 2. Split candidates
    chunk_size = math.ceil(len(candidates) / num_gpus)
    chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]

    # 3. Spawn subprocesses
    processes = []
    tmp_outs = []
    for i, chunk in enumerate(chunks):
        tmp_out = f"{args.out}_shard{i}"
        tmp_outs.append(tmp_out)

        cmd = [
            sys.executable, "-m", "orka", "pack",
            args.source, "--out", tmp_out
        ] + unknown + ["--only-tensors"] + chunk

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)

        print(f"Spawning Worker {i} on GPU {i} for {len(chunk)} tensors -> {tmp_out}")
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    # Wait for all workers
    for i, p in enumerate(processes):
        p.wait()
        if p.returncode != 0:
            print(f"Worker {i} failed with exit code {p.returncode}. Aborting.")
            sys.exit(p.returncode)

    print("All workers finished successfully. Merging artifacts...")

    # 4. Merge artifacts
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tensors").mkdir(parents=True, exist_ok=True)

    merged_manifest = None
    merged_passthrough = {}
    total_index_bytes = 0

    for i, tmp_out in enumerate(tmp_outs):
        tmp_path = Path(tmp_out)

        # Merge tensors
        for file in (tmp_path / "tensors").iterdir():
            shutil.move(str(file), str(out_dir / "tensors" / file.name))

        # Merge codebooks
        if (tmp_path / "codebooks").exists():
            (out_dir / "codebooks").mkdir(parents=True, exist_ok=True)
            for file in (tmp_path / "codebooks").iterdir():
                dest = out_dir / "codebooks" / file.name
                if not dest.exists():
                    shutil.copy(str(file), str(dest))

        # Merge manifest
        with open(tmp_path / "manifest.json", "r") as f:
            manifest = json.load(f)
            if merged_manifest is None:
                merged_manifest = manifest
                merged_manifest["tensors"] = []
            merged_manifest["tensors"].extend(manifest["tensors"])
            total_index_bytes += manifest.get("total_index_bytes", 0)

        # Merge true passthrough tensors
        # A tensor is true passthrough only if it was NOT in the global candidates list
        from safetensors import safe_open
        pt_file = tmp_path / "passthrough.safetensors"
        if pt_file.exists():
            with safe_open(str(pt_file), framework="np") as handle:
                for key in handle.keys():
                    if key not in candidates and key not in merged_passthrough:
                        merged_passthrough[key] = handle.get_tensor(key)

        # Clean up tmp
        shutil.rmtree(tmp_path)

    if merged_passthrough:
        from orka._format import _write_passthrough_tensors
        _write_passthrough_tensors(out_dir / "passthrough.safetensors", merged_passthrough)
        merged_manifest["passthrough_count"] = len(merged_passthrough)

    merged_manifest["total_index_bytes"] = total_index_bytes

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(merged_manifest, f, indent=2)

    print(f"Successfully created parallel artifact at {args.out}")

if __name__ == '__main__':
    main()
