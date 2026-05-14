import torch
import numpy as np
import json
from pathlib import Path
from safetensors import safe_open
from orka.reconstruct import reconstruct_artifact

# 1. SETUP DUMMY ARTIFACT
out_dir = Path("test_artifact")
out_dir.mkdir(exist_ok=True)
tensors_dir = out_dir / "tensors"
tensors_dir.mkdir(exist_ok=True)

# Create a dummy quantized tensor
name = "dummy.weight"
shape = [128, 128]
data = np.random.randn(*shape).astype(np.float32)
data.tofile(tensors_dir / f"{name}.f32")

manifest = {
    "source": "nonexistent.safetensors",
    "tensors": [{
        "name": name,
        "shape": shape,
        "path": f"tensors/{name}.f32"
    }]
}
(out_dir / "manifest.json").write_text(json.dumps(manifest))

# 2. RUN RECONSTRUCTION
output_path = Path("reconstructed.safetensors")
reconstruct_artifact(out_dir, output_path, output_format="safetensors")

# 3. VERIFY WITH OFFICIAL LIBRARY
print("Verifying with safe_open...")
try:
    with safe_open("reconstructed.safetensors", framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            print(f"  Found tensor: {k}, shape: {list(t.shape)}, dtype: {t.dtype}")
            # Check values
            original = np.fromfile(tensors_dir / f"{name}.f32", dtype=np.float32).reshape(shape)
            if np.allclose(t.numpy(), original):
                print("  SUCCESS: Values match exactly!")
            else:
                print("  FAILED: Values mismatch!")
except Exception as e:
    print(f"  CRITICAL FAILURE: {e}")
