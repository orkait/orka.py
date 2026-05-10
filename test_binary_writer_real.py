import torch
import numpy as np
import json
from pathlib import Path
from safetensors import safe_open
from orka.reconstruct import reconstruct_artifact

# 1. USE REAL ARTIFACT FROM PREVIOUS RUN
out_dir = Path("results/test-metrics.orka")
if not out_dir.exists():
    print("FAILED: results/test-metrics.orka not found. Run the metrics validation script first.")
    exit(1)

# 2. RUN RECONSTRUCTION WITH NEW BINARY WRITER
output_path = Path("reconstructed_binary.safetensors")
print(f"Reconstructing {out_dir} to {output_path}...")
result = reconstruct_artifact(out_dir, output_path, output_format="safetensors")
print(f"Result: {result}")

# 3. VERIFY WITH OFFICIAL LIBRARY
print("Verifying with safe_open...")
try:
    with safe_open(str(output_path), framework="pt") as f:
        keys = sorted(f.keys())
        print(f"  Found {len(keys)} tensors.")
        for k in keys[:5]: # Check first 5
            t = f.get_tensor(k)
            print(f"    Tensor: {k}, shape: {list(t.shape)}, dtype: {t.dtype}")
        
    print("\nSUCCESS: Binary writer produced a valid, readable Safetensors file!")
except Exception as e:
    print(f"\nCRITICAL FAILURE: {e}")
    import traceback
    traceback.print_exc()
