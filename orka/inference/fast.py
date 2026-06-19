"""Fast inference loader: reconstruct-then-dense + torch.compile.

Profiling a Falcon H1 (Mamba2 hybrid) decode showed the bottleneck is NOT the
quantized linears (already bandwidth-optimal) but:
  - launch overhead: CPU launch time > GPU compute time (GPU idle ~45%)
  - the naive SSM selective-scan fallback (thousands of tiny elementwise ops)
  - dynamic KV-cache concat every step

torch.compile (default/inductor) fuses the elementwise ops and collapses the
per-layer launches into graphs, attacking all three at once. Measured on a
3060: 66 -> 147 tok/s (2.2x), bit-exact generations.

For a model that fits dense (<7B at 4bpw), reconstruct-then-dense amortizes the
one-time decode over all tokens and runs every token at ~92% HBM bandwidth -
strictly better than per-token VQ decode (which is gather-bound at ~4% BW). So
the fast path loads the dense bf16 export (export_vllm output), not VQLinear.

Note: mode="reduce-overhead" (CUDA graphs) segfaults on this model - the Mamba
conv_state in-place mutation is incompatible with cudagraph capture. Default
mode is the supported configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch


def load_fast(
    hf_dense_dir: Union[str, Path],
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    compile_model: bool = True,
):
    """Load a dense HF export (export_vllm output) and torch.compile it.

    Args:
        hf_dense_dir: directory produced by orka export-vllm (dense bf16 weights)
        device: target device
        dtype: compute dtype
        compile_model: apply torch.compile to the inner model (default True)

    Returns:
        (model, tokenizer) ready for generate().
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_dense_dir = Path(hf_dense_dir)
    tok = AutoTokenizer.from_pretrained(hf_dense_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_dense_dir, local_files_only=True, dtype=dtype
    ).to(device).eval()

    if compile_model:
        # Compile the inner transformer stack. The lm_head + generate loop stay
        # eager (cheap, and keeps the compiled region shape-stable for decode).
        model.model = torch.compile(model.model)

    return model, tok
