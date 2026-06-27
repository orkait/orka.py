"""vLLM quantization method for .orka (compressed-resident inference).

Registers ``"orka"`` as a vLLM quant method so a packed orka HF repo (produced by
``orka.integrations.hf_quantizer.export_orka_hf_repo``) loads through vLLM's engine - CUDA graphs,
paged attention, continuous batching - while weights stay compressed (the VQLinear
kernels). vLLM's runtime removes the ~2.3x transformers-eager overhead measured for the
HF path; the VQ kernels are reused as-is.

Status: validated functional. An orka HF repo (pythia-160m, group-8) loaded through vLLM
v0.23 with ``quantization="orka"`` and generated end to end - the orka VQLinear /
plane kernels run inside vLLM's engine (220 tok/s warm decode, enforce_eager). vLLM is
imported lazily so this module is import-safe without vllm installed.

Environment: vLLM's EngineCore subprocess JIT-compiles the orka CUDA/Triton kernels, so
it needs ``ninja``, a C compiler (``CC``), and ``nvcc`` on PATH (set ``CC``/``CUDA_HOME``
for the engine process). Run with ``enforce_eager=True`` until the plane custom op is
made CUDA-graph-safe.

Reference: https://docs.vllm.ai/en/latest/features/quantization/ ; vllm awq.py.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def _build_vq_linear_from_meta(meta: dict):
    """Construct an (unpopulated) VQLinear from one module's orka metadata. Reused by
    both the transformers quantizer and the vLLM method - this is the tested core."""
    from orka.inference.vq_linear import VQLinear

    bias = torch.zeros(meta["out_features"], dtype=torch.float16) if meta.get("has_bias") else None
    vq = VQLinear(
        out_features=meta["out_features"],
        in_features=meta["in_features"],
        n_stages=meta["n_stages"],
        group_size=meta["group_size"],
        block_size=meta["block_size"],
        cb_sizes=meta["cb_sizes"],
        bias=bias,
    )
    if meta.get("group_major"):
        vq._group_major = True
    return vq


def register_orka_vllm() -> None:
    """Register the "orka" quant method with vLLM. No-op (raises) if vllm is absent."""
    from vllm.model_executor.layers.linear import LinearMethodBase
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    @register_quantization_config("orka")
    class OrkaQuantConfig(QuantizationConfig):
        """vLLM config for orka-quantized checkpoints. Carries the per-module VQ metadata
        emitted in the HF repo's quantization_config."""

        def __init__(self, modules: Optional[dict] = None) -> None:
            super().__init__()
            self.modules = modules or {}

        @classmethod
        def get_name(cls) -> str:
            return "orka"

        @classmethod
        def get_supported_act_dtypes(cls) -> list:
            return [torch.float16, torch.bfloat16]

        @classmethod
        def get_min_capability(cls) -> int:
            return 70  # Volta+ (Triton/CUDA VQ kernels)

        @classmethod
        def get_config_filenames(cls) -> list:
            return []

        @classmethod
        def from_config(cls, config: dict) -> "OrkaQuantConfig":
            return cls(modules=config.get("modules", {}))

        def get_quant_method(self, layer, prefix: str):
            # Route only the modules orka actually quantized; others stay unquantized.
            meta = self.modules.get(prefix)
            if meta is None:
                return None
            return OrkaLinearMethod(meta)

    class OrkaLinearMethod(LinearMethodBase):
        """Backs a vLLM linear layer with an orka VQLinear (compressed-resident)."""

        def __init__(self, meta: dict) -> None:
            self.meta = meta

        def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                           input_size, output_size, params_dtype, **extra):
            # Build a VQLinear whose registered buffers ARE the vLLM weights; vLLM's loader
            # fills them from the checkpoint (names match the HF repo tensor names).
            vq = _build_vq_linear_from_meta(self.meta)
            layer.orka_vq = vq
            for name, buf in vq.named_buffers():
                if buf is None:
                    continue
                layer.register_parameter(
                    name, torch.nn.Parameter(buf, requires_grad=False)
                )

        def apply(self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
            vq = layer.orka_vq
            # sync any vLLM-loaded parameter data back onto the VQLinear buffers
            for name, p in layer.named_parameters(recurse=False):
                if hasattr(vq, name):
                    getattr(vq, name).copy_(p.data)
            out = vq(x)
            if bias is not None:
                out = out + bias
            return out

    return None
