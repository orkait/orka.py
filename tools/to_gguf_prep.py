#!/usr/bin/env python3
"""Prepares an Orka compressed model for conversion to GGUF format by reconstructing it as a standard Hugging Face directory."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from orka.artifact.reconstruct import reconstruct_artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct an Orka model into a standard Hugging Face directory ready for GGUF conversion."
    )
    parser.add_argument("artifact", help="Path to the .orka directory")
    parser.add_argument("--out-dir", required=True, help="Target Hugging Face model directory to create")
    parser.add_argument("--device", default="cpu", help="Device to use for reconstruction (cpu or cuda)")
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    out_dir = Path(args.out_dir)

    if not artifact_path.exists():
        print(f"Error: Orka artifact does not exist: {artifact_path}", file=sys.stderr)
        sys.exit(1)

    manifest_path = artifact_path / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: Missing manifest.json in artifact: {artifact_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # 1. Resolve source model directory containing tokenizer & config files
    source_path = Path(manifest.get("source", ""))
    source_dir = source_path if source_path.is_dir() else source_path.parent

    if not source_dir.exists():
        print(f"Warning: Source directory {source_dir} not found. Tokenizer/config files must be copied manually.")

    # Create target directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Reconstruct weights into safetensors
    target_safetensors = out_dir / "model.safetensors"
    print(f"Reconstructing weights to {target_safetensors}...")
    reconstruct_artifact(artifact_path, target_safetensors, output_format="safetensors", device=args.device)

    # 3. Copy tokenizer & configuration sidecars
    if source_dir.exists():
        print("Copying Hugging Face configuration sidecars...")
        for child in source_dir.iterdir():
            if child.is_file() and not any(
                child.name.endswith(ext)
                for ext in (".safetensors", ".bin", ".pt", ".pth", ".onnx", ".gguf", ".index.json")
            ):
                shutil.copy2(child, out_dir / child.name)
                print(f"  Copied {child.name}")

    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Hugging Face directory ready: {out_dir}")
    print("\nTo convert this directory to a standard GGUF file using llama.cpp:")
    print("1. Clone and build llama.cpp.")
    print("2. Run the conversion script:")
    print(f"   python llama.cpp/convert_hf_to_gguf.py {out_dir} --outtype f16")
    print("=" * 60)


if __name__ == "__main__":
    main()
