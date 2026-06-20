#!/usr/bin/env python3
import sys
import json
import math
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add local path and llama.cpp/gguf-py to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType

from orka import replace_linear_with_orka
from orka.transforms.rotate import _unrotate_flat

class GGUFOrkaLinear(nn.Module):
    def __init__(self, tensor_meta, gguf_tensors, reader, bias_tensor=None):
        super().__init__()
        self.tensor_meta = tensor_meta
        self.gguf_tensors = gguf_tensors
        self.reader = reader
        self.in_features = int(tensor_meta["shape"][1])
        self.out_features = int(tensor_meta["shape"][0])

        if bias_tensor is not None:
            self.bias = nn.Parameter(bias_tensor.clone().to(torch.float32))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("_reconstructed_weight", None)

    def reconstruct_weight(self, device):
        if self._reconstructed_weight is not None and self._reconstructed_weight.device == device:
            return self._reconstructed_weight

        # Decompress directly from GGUF tensors on CPU, then move to device
        name = self.tensor_meta["name"]
        group_size = int(self.tensor_meta["group_size"])
        padded_values = int(self.tensor_meta["padded_values"])
        packed_values = int(self.tensor_meta["packed_values"])
        shape = [int(x) for x in self.tensor_meta["shape"]]
        index_count = math.ceil(padded_values / group_size)

        decoded = np.zeros(index_count * group_size, dtype=np.float32)
        stages = self.tensor_meta.get("stages", [])
        if not stages:
            stages = [{
                "stage": 0,
                "codebook_size": int(self.tensor_meta["codebook_size"]),
                "index_bits": int(self.tensor_meta["index_bits"]),
            }]

        for stage in stages:
            sid = stage.get("stage", 0)
            cb_size = stage.get("codebook_size", int(self.tensor_meta["codebook_size"]))
            idx_bits = stage.get("index_bits", int(self.tensor_meta["index_bits"]))
            s_group_size = int(stage.get("group_size", group_size))

            # Load codebook
            original_cb_name = f"{name}.orka.s{sid}.codebook"
            cb_map_key = f"orka.cb_map.{original_cb_name}"
            field = self.reader.fields.get(cb_map_key)
            if field is None:
                raise ValueError(f"Missing cb_map metadata for {original_cb_name}")
            shared_cb_name = field.contents()

            cb_tensor = self.gguf_tensors[shared_cb_name]
            if cb_tensor.tensor_type == GGMLQuantizationType.Q8_0:
                cb_data = dequantize(cb_tensor.data, GGMLQuantizationType.Q8_0)
            else:
                cb_data = cb_tensor.data.view(np.float16).astype(np.float32)

            cb = cb_data.reshape(-1, s_group_size)

            # Load indices
            idx_tensor_name = f"{name}.orka.s{sid}.indices"
            idx_tensor = self.gguf_tensors[idx_tensor_name]
            if idx_bits > 8:
                indices = idx_tensor.data.view(np.uint16).astype(np.int64)
            else:
                indices = idx_tensor.data.view(np.uint8).astype(np.int64)

            decoded += cb[indices].reshape(-1)

        decoded = decoded[:packed_values]

        # Outlier / pillar escape (absolute positions, pre-rotation / pre-scale)
        outl = self.tensor_meta.get("outliers")
        if outl:
            pos = self.gguf_tensors[f"{name}.orka.outlier.idx"].data.astype(np.int64)
            ov_t = self.gguf_tensors[f"{name}.orka.outlier.val"]
            if ov_t.tensor_type == GGMLQuantizationType.Q8_0:
                ov = dequantize(ov_t.data, GGMLQuantizationType.Q8_0)
            else:
                ov = ov_t.data.view(np.float16).astype(np.float32)
            mask = pos < decoded.size
            decoded[pos[mask]] = ov[mask]

        # Apply rotation
        rotation = self.tensor_meta.get("rotation", "none")
        if rotation in {"orthogonal", "hadamard"}:
            seed = int(self.tensor_meta.get("rotation_seed") or 0)
            decoded_list = decoded.tolist()
            decoded_unrotated = _unrotate_flat(decoded_list, self.tensor_meta["shape"], rotation, seed)
            decoded = np.array(decoded_unrotated, dtype=np.float32)

        # Apply scales
        norm = self.tensor_meta.get("normalization", "none")
        if norm in ("block-max", "channel-block-max", "awq-block-max", "slrq-block"):
            scales_tensor_name = f"{name}.orka.scales"
            scales_tensor = self.gguf_tensors[scales_tensor_name]
            if scales_tensor.tensor_type == GGMLQuantizationType.Q8_0:
                scales = dequantize(scales_tensor.data, GGMLQuantizationType.Q8_0)
            else:
                scales = scales_tensor.data.view(np.float16).astype(np.float32)

            block_size = int(self.tensor_meta.get("block_scale_size") or 32)
            n = decoded.size
            pad = (-n) % block_size
            if pad:
                decoded = np.concatenate([decoded, np.zeros(pad, dtype=np.float32)])
            decoded = (decoded.reshape(-1, block_size) * scales[:decoded.size // block_size, None]).reshape(-1)
            if pad:
                decoded = decoded[:n]
        elif norm == "awq":
            scales_tensor_name = f"{name}.orka.scales"
            scales_tensor = self.gguf_tensors[scales_tensor_name]
            if scales_tensor.tensor_type == GGMLQuantizationType.Q8_0:
                scales = dequantize(scales_tensor.data, GGMLQuantizationType.Q8_0)
            else:
                scales = scales_tensor.data.view(np.float16).astype(np.float32)
            cols = scales.size
            rows = decoded.size // cols
            decoded = (decoded[:rows * cols].reshape(rows, cols) * scales[None, :]).reshape(-1)

        # Apply salient outliers
        salient = self.tensor_meta.get("salient")
        if salient:
            sal_idx_tensor = self.gguf_tensors[f"{name}.orka.salient.idx"]
            sal_idx = sal_idx_tensor.data.astype(np.int64)

            sal_val_tensor = self.gguf_tensors[f"{name}.orka.salient.val"]
            if sal_val_tensor.tensor_type == GGMLQuantizationType.Q8_0:
                sal_val = dequantize(sal_val_tensor.data, GGMLQuantizationType.Q8_0)
            else:
                sal_val = sal_val_tensor.data.view(np.float16).astype(np.float32)

            block_size = int(self.tensor_meta.get("block_scale_size") or 32)
            for b_idx, (local_idx, weight) in enumerate(zip(sal_idx, sal_val)):
                flat_idx = b_idx * block_size + int(local_idx)
                if flat_idx < decoded.size:
                    decoded[flat_idx] = float(weight)

        # Low-rank correction: decoded += A @ B^T, applied last
        lr = self.tensor_meta.get("lowrank")
        if lr:
            r = int(lr["rank"])
            a = self.gguf_tensors[f"{name}.orka.lowrank.a"].data.view(np.float16).astype(np.float32).reshape(-1, r)
            b = self.gguf_tensors[f"{name}.orka.lowrank.b"].data.view(np.float16).astype(np.float32).reshape(-1, r)
            rows = a.shape[0]
            cols = b.shape[0]
            decoded = (decoded[:rows * cols].reshape(rows, cols) + a @ b.T).reshape(-1)

        w_torch = torch.from_numpy(decoded.reshape(shape)).to(device=device, dtype=torch.float32)
        self._reconstructed_weight = w_torch
        return w_torch

    def forward(self, x):
        w = self.reconstruct_weight(x.device)
        return torch.nn.functional.linear(x.to(w.dtype), w, self.bias)

def replace_linear_with_gguf(model, gguf_path, manifest_path):
    with open(manifest_path) as f:
        manifest = json.load(f)

    meta_map = {t["name"]: t for t in manifest.get("tensors", [])}

    reader = GGUFReader(gguf_path)
    gguf_tensors = {t.name: t for t in reader.tensors}

    # Extract passthrough tensors
    pp_tensors = {}
    for name, t in gguf_tensors.items():
        if ".orka." not in name and not name.startswith("orka.shared_cb."):
            # Type 0 is FP32
            val = t.data.view(np.float32)
            pp_tensors[name] = torch.from_numpy(val.copy())

    def _replace(module, prefix=""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            # Embeddings replacement
            if name == "embed_tokens" and f"{full_name}.weight" in meta_map:
                # We can replace embed weights with the dequantized weights from GGUF
                meta = meta_map[f"{full_name}.weight"]
                # Decompress on CPU
                dummy_linear = GGUFOrkaLinear(meta, gguf_tensors, reader)
                w_decompressed = dummy_linear.reconstruct_weight("cpu")
                child.weight.data = w_decompressed.to(child.weight.dtype)
                print(f"  Successfully updated embedding layer: {full_name}")
                continue

            if isinstance(child, nn.Linear):
                weight_name = f"{full_name}.weight"
                bias_name = f"{full_name}.bias"

                if weight_name in meta_map:
                    bias_t = None
                    if bias_name in pp_tensors:
                        bias_t = pp_tensors[bias_name]
                    elif child.bias is not None:
                        bias_t = child.bias.data.clone()

                    layer = GGUFOrkaLinear(meta_map[weight_name], gguf_tensors, reader, bias_t)
                    setattr(module, name, layer)
            else:
                _replace(child, full_name)

    _replace(model)

def main():
    ap = argparse.ArgumentParser(
        description="Compare original vs raw-Orka vs GGUF-Orka generations on a few prompts."
    )
    ap.add_argument("model_dir", help="HF model dir (config/tokenizer + base weights)")
    ap.add_argument("orka_dir", help="Path to the reference .orka artifact directory")
    ap.add_argument("gguf_path", help="Path to the Orka GGUF file")
    ap.add_argument("--max-length", type=int, default=48, help="max generation length")
    args = ap.parse_args()
    model_dir, orka_dir, gguf_path = args.model_dir, args.orka_dir, args.gguf_path

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print("Loading Original HF Model...", flush=True)
    model_og = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)

    print("Loading Raw Orka Model...", flush=True)
    model_raw = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    replace_linear_with_orka(model_raw, orka_dir)

    print("Loading GGUF Orka Model...", flush=True)
    model_gguf = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    replace_linear_with_gguf(model_gguf, gguf_path, Path(orka_dir) / "manifest.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Moving models to {device}...", flush=True)
    model_og.to(device).eval()
    model_raw.to(device).eval()
    model_gguf.to(device).eval()

    prompts = [
        "The capital of France is",
        "Python is a programming language that",
        "Deep learning is a subset of",
    ]

    def generate(model, prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=args.max_length,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    for idx, prompt in enumerate(prompts, 1):
        print(f"\n" + "=" * 60)
        print(f"PROMPT {idx}: '{prompt}'")
        print("=" * 60)

        text_og = generate(model_og, prompt)
        print(f"[ORIGINAL MODEL]:\n{text_og}\n")

        text_raw = generate(model_raw, prompt)
        print(f"[RAW ORKA MODEL]:\n{text_raw}\n")

        text_gguf = generate(model_gguf, prompt)
        print(f"[GGUF ORKA MODEL]:\n{text_gguf}")

if __name__ == "__main__":
    main()
