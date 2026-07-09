"""Quantization spec + family classification + payload size estimation."""

from orka.quant.arch import (
    ArchProfile,
    is_output_head,
    is_recurrent_block,
    output_head_names,
    recurrent_block_names,
)
from orka.quant.family import classify_tensor_family
from orka.quant.spec import (
    QUANT_SPEC_MAX_PER_STAGE_BITS,
    QUANT_SPEC_MAX_TOTAL_BITS,
    RVQ_MIXED_FAMILY_BITS,
    PayloadEstimate,
    _resolve_quant_stages,
    estimate_payload,
    is_rvq_mixed_spec,
    parse_quant_spec,
    quant_spec_from_sizes,
    rvq_mixed_family_stages,
)
