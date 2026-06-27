"""Load an .orka artifact straight into a live Hugging Face model.

An .orka artifact stores the compressed weights (codebooks + indices + scales) plus a
`passthrough.safetensors` for the un-quantized tensors (norms, biases, small embeddings).
Paired with the architecture scaffold (`config.json` + tokenizer), that is everything
needed to rebuild the model - no separate dense checkpoint required.

This is the "small repo, one call" consumer path: publish the .orka files + config +
tokenizer (≈compressed size on disk), then::

    from orka.hf import load_orka_model, load_orka_tokenizer
    model = load_orka_model("path/or/hf_repo_dir", device="cuda")
    tok = load_orka_tokenizer("path/or/hf_repo_dir")

The weights are decoded to dense at load time (the project's preferred inference path for
models that fit - full HBM bandwidth, see orka.inference.fast). The on-disk / download
footprint stays compressed; RAM after load is dense fp16. A packed-in-RAM path (VQLinear)
is a separate, future integration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from orka.reconstruct import _complete_decoded_tensor_map


def _orka_state_dict(orka_dir, dtype=None) -> dict:
    """Decode an .orka artifact into a name -> torch.Tensor state dict (dense).

    Pure reconstruction (no model needed) so it is unit-testable against the decoder.
    """
    import numpy as np
    import torch

    orka_dir = Path(orka_dir)
    manifest = json.loads((orka_dir / "manifest.json").read_text())
    tmap = _complete_decoded_tensor_map(orka_dir, manifest)

    state: dict = {}
    for name, entry in tmap.items():
        shape = [int(x) for x in entry["shape"]]
        arr = np.asarray(entry["flat"], dtype=np.float32).reshape(shape)
        t = torch.from_numpy(arr)
        state[name] = t.to(dtype) if dtype is not None else t
    return state


def load_orka_model(
    orka_dir,
    *,
    config_dir: Optional[str] = None,
    device: str = "cpu",
    dtype=None,
):
    """Build the HF CausalLM from config and load the decoded .orka weights into it.

    Args:
        orka_dir: directory with manifest.json, tensors/, passthrough.safetensors.
        config_dir: where config.json + tokenizer live (defaults to ``orka_dir``).
        device: target device for the ready model.
        dtype: compute dtype (default torch.float16).

    Returns:
        A ready-to-run ``transformers`` model in eval mode.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    if dtype is None:
        dtype = torch.float16
    orka_dir = Path(orka_dir)
    cfg_src = Path(config_dir) if config_dir else orka_dir

    cfg = AutoConfig.from_pretrained(cfg_src, local_files_only=True)
    model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)

    state = _orka_state_dict(orka_dir, dtype=dtype)
    # strict=False: tied weights (e.g. lm_head <- embeddings) and any arch-only buffers
    # are filled by the model itself; we only supply what the artifact carries.
    model.load_state_dict({k: v for k, v in state.items() if k in model.state_dict()}, strict=False)
    return model.to(device).eval()


def load_orka_tokenizer(config_dir):
    """Load the tokenizer that ships alongside the artifact."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(Path(config_dir), local_files_only=True)
