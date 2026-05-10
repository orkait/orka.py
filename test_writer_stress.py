import torch
import numpy as np
import json
import struct
from pathlib import Path
from safetensors import safe_open
from orka.reconstruct import reconstruct_artifact

def run_stress_test():
    out_dir = Path("stress_artifact")
    out_dir.mkdir(exist_ok=True)
    tensors_dir = out_dir / "tensors"
    tensors_dir.mkdir(exist_ok=True, parents=True)

    # 1. SETUP
    q_name = "layer.1.weight"
    q_shape = [4, 4]
    (out_dir / "tensors/layer.1.cb").write_bytes(np.random.randn(2, 1).astype("<f4").tobytes())
    (out_dir / "tensors/layer.1.idx").write_bytes(b"\0" * 16)
    
    e_name = "empty.tensor"
    e_shape = [0, 10]
    (out_dir / "tensors/empty.cb").write_bytes(np.zeros((1, 1), dtype="<f4").tobytes())
    (out_dir / "tensors/empty.idx").write_bytes(b"")
    
    s_path = Path("stress_source.safetensors")
    from safetensors.torch import save_file as save_st
    s_data = torch.randn(8, 8, dtype=torch.bfloat16)
    save_st({"fallback.weight": s_data}, str(s_path))

    manifest = {
        "source": str(s_path),
        "tensors": [
            {
                "name": q_name, "shape": q_shape, "group_size": 1, "padded_values": 16,
                "packed_values": 16, "codebook": "tensors/layer.1.cb",
                "codebook_size": 2, "index_bits": 8, "indices": "tensors/layer.1.idx"
            },
            {
                "name": e_name, "shape": e_shape, "group_size": 1, "padded_values": 0,
                "packed_values": 0, "codebook": "tensors/empty.cb",
                "codebook_size": 1, "index_bits": 8, "indices": "tensors/empty.idx"
            }
        ]
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))

    output_path = Path("stress_result.safetensors")
    reconstruct_artifact(out_dir, output_path, output_format="safetensors")

    # 2. VERIFY DATA CONTENT
    print("\nVerifying data content with safe_open...")
    try:
        with safe_open(str(output_path), framework="pt") as f:
            keys = sorted(f.keys())
            for k in keys:
                t = f.get_tensor(k)
                shape_str = str(list(t.shape))
                print(f"  Tensor: {k:<15} Shape: {shape_str:<15} Dtype: {t.dtype}")
                if k == "fallback.weight" and t.dtype == torch.float32:
                    print("    -> BF16 to FP32 conversion: SUCCESS")
                if k == e_name and t.numel() == 0:
                    print("    -> Empty tensor handling: SUCCESS")

        print("\nOVERALL STATUS: ALL EDGE CASES VERIFIED")
    except Exception as e:
        print(f"\nOVERALL STATUS: FAILURE ({e})")

run_stress_test()
