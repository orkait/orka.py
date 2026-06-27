import os
import sys
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

    # 4. Merge artifacts via the canonical merge (compat checks + conflict detection)
    from orka.artifact.merge import merge_orka_artifacts

    out_dir = Path(args.out)
    if len(tmp_outs) == 1:
        shutil.move(tmp_outs[0], str(out_dir))
    else:
        merge_orka_artifacts([Path(t) for t in tmp_outs], out_dir)
        for tmp_out in tmp_outs:
            shutil.rmtree(tmp_out, ignore_errors=True)

    print(f"Successfully created parallel artifact at {args.out}")

if __name__ == '__main__':
    main()
