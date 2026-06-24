"""Native transformers quantizer for .orka artifacts.

Lets a compressed model load through stock ``AutoModelForCausalLM.from_pretrained`` -
no orka-specific load call - while the weights stay VQ-packed in RAM and run through
the VQLinear kernels.

Two pieces:
  * ``export_orka_hf_repo`` - serialize an .orka artifact into a HF repo: packed
    VQLinear buffers go into ``model.safetensors`` under ``<module>.<buffer>`` keys,
    quantized embeddings are stored dense (the inference kernel only backs Linear),
    passthrough tensors (norms/biases) stay dense, and ``config.json`` gets a
    ``quantization_config`` with the per-module VQ metadata.
  * ``OrkaConfig`` + ``OrkaHfQuantizer`` - registered under the ``"orka"`` quant method.
    At load time the quantizer swaps each target ``nn.Linear`` for a ``VQLinear`` skeleton
    (shapes from the metadata); the standard weight loader then fills the packed buffers
    from the checkpoint.

Consumer::

    import orka.hf_quantizer            # registers the "orka" method
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("you/model-orka", device_map="cuda")

Constraints (the inference kernel): every quantized tensor must be packed with
``group_size=8`` and ``block_size=32``. Sparse correction sidecars (outliers / salient)
are not yet supported on this path - pack without them, or use ``orka.hf.load_orka_model``
(dense reconstruct) for those artifacts.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

KERNEL_GROUP_SIZE = 8
KERNEL_BLOCK_SIZE = 32
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


def _is_embedding(name: str) -> bool:
    # The kernel only backs nn.Linear; embeddings (lookup / tied LM head) stay dense.
    return "embed" in name.lower()


def export_orka_hf_repo(artifact_dir, config_dir, out_dir) -> dict:
    """Serialize an .orka artifact into a transformers-loadable repo.

    Args:
        artifact_dir: the .orka directory (manifest.json, tensors/, passthrough.safetensors).
        config_dir: source of config.json + tokenizer (the architecture scaffold).
        out_dir: repo directory to create.

    Returns:
        Summary dict (counts + repo path).
    """
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from orka.inference.vq_linear import build_vq_linear
    from orka.pipeline.decode import _decode_tensor

    artifact_dir = Path(artifact_dir)
    config_dir = Path(config_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((artifact_dir / "manifest.json").read_text())

    shutil.copy(config_dir / "config.json", out_dir / "config.json")
    for fn in _TOKENIZER_FILES:
        if (config_dir / fn).exists():
            shutil.copy(config_dir / fn, out_dir / fn)

    state: dict = {}
    passthrough = artifact_dir / "passthrough.safetensors"
    if passthrough.exists():
        with safe_open(str(passthrough), "pt") as f:
            for k in f.keys():
                state[k] = f.get_tensor(k).to(torch.float16)

    modules_meta: dict = {}
    dense_embeddings = 0
    for tm in manifest["tensors"]:
        name = tm["name"]
        shape = [int(x) for x in tm["shape"]]
        if _is_embedding(name):
            arr = np.asarray(_decode_tensor(artifact_dir, tm), dtype=np.float32).reshape(shape)
            state[name] = torch.from_numpy(arr).to(torch.float16)
            dense_embeddings += 1
            continue

        gs = int(tm.get("group_size", 8))
        bs = int(tm.get("block_scale_size") or 32)
        if gs != KERNEL_GROUP_SIZE or bs != KERNEL_BLOCK_SIZE:
            raise ValueError(
                f"{name}: native quantizer needs group_size={KERNEL_GROUP_SIZE} and "
                f"block_size={KERNEL_BLOCK_SIZE} (got group_size={gs}, block_size={bs}). "
                "Re-pack with those, or use orka.hf.load_orka_model for dense reconstruct."
            )

        module_path = name[: -len(".weight")]
        bias = state.pop(module_path + ".bias", None)
        vq = build_vq_linear(artifact_dir, tm, bias.float() if bias is not None else None, device="cpu")

        if vq.corr_col.numel() != 0:
            raise ValueError(
                f"{name}: sparse correction (outliers/salient) is not supported on the native "
                "quantizer path yet. Pack without them, or use orka.hf.load_orka_model."
            )

        for bname, buf in vq.named_buffers():
            if buf is None or bname.startswith("corr_"):
                continue  # empty CSR correction stays out of the checkpoint
            state[f"{module_path}.{bname}"] = buf.contiguous()

        modules_meta[module_path] = {
            "out_features": vq.out_features,
            "in_features": vq.in_features,
            "n_stages": vq.n_stages,
            "group_size": vq.group_size,
            "block_size": vq.block_size,
            "cb_sizes": vq.cb_sizes,
            "has_bias": bias is not None,
            "group_major": bool(getattr(vq, "_group_major", False)),
        }

    save_file(state, str(out_dir / "model.safetensors"), metadata={"format": "pt"})

    cfg = json.loads((out_dir / "config.json").read_text())
    cfg["quantization_config"] = {"quant_method": "orka", "modules": modules_meta}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    return {
        "out": str(out_dir),
        "vq_linear_modules": len(modules_meta),
        "dense_embeddings": dense_embeddings,
        "state_tensors": len(state),
    }


def register_orka_quantizer() -> None:
    """Register the 'orka' quant method with transformers (idempotent)."""
    import torch
    from transformers.quantizers import HfQuantizer
    from transformers.quantizers.auto import (
        AUTO_QUANTIZER_MAPPING,
        register_quantization_config,
        register_quantizer,
    )
    from transformers.utils.quantization_config import QuantizationConfigMixin

    if "orka" in AUTO_QUANTIZER_MAPPING:
        return

    @register_quantization_config("orka")
    class OrkaConfig(QuantizationConfigMixin):
        def __init__(self, modules=None, **kwargs):
            self.quant_method = "orka"
            self.modules = modules or {}

        def to_dict(self):
            return {"quant_method": "orka", "modules": self.modules}

    def _set_submodule(model, path, new):
        parts = path.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new)

    @register_quantizer("orka")
    class OrkaHfQuantizer(HfQuantizer):
        requires_calibration = True  # weights arrive pre-quantized in the checkpoint
        requires_parameters_quantization = False

        def validate_environment(self, *args, **kwargs):
            import orka.inference.vq_linear  # noqa: F401

        def update_dtype(self, dtype):
            return torch.float16

        def _process_model_before_weight_loading(self, model, **kwargs):
            from orka.inference.vq_linear import VQLinear

            for path, m in self.quantization_config.modules.items():
                bias = torch.zeros(m["out_features"], dtype=torch.float16) if m["has_bias"] else None
                vq = VQLinear(
                    out_features=m["out_features"],
                    in_features=m["in_features"],
                    n_stages=m["n_stages"],
                    group_size=m["group_size"],
                    block_size=m["block_size"],
                    cb_sizes=m["cb_sizes"],
                    bias=bias,
                )
                if m.get("group_major"):
                    vq._group_major = True
                # empty correction buffers are not in the checkpoint
                for nb in ("corr_rowptr", "corr_col", "corr_val"):
                    vq._non_persistent_buffers_set.add(nb)
                _set_submodule(model, path, vq)

        def _process_model_after_weight_loading(self, model, **kwargs):
            return model

        def is_serializable(self, *args, **kwargs):
            return True

        @property
        def is_trainable(self):
            return False


# Register on import so `import orka.hf_quantizer` is all a consumer needs.
register_orka_quantizer()
