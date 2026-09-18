import pytest

from orka.autoquant.schema import TensorConfig, to_packer_allocation
from orka.autoquant.validation import validate_config, validate_configs


def test_output_head_rvq_is_rejected():
    cfg = TensorConfig("rvq", 4, 2, "block-max", False, "llm", 0.8, "bad")
    with pytest.raises(ValueError, match="out-head"):
        validate_config("out-head", cfg)


def test_norm_must_remain_fp16():
    cfg = TensorConfig("int8", 8, 0, "none", False, "llm", 0.8, "bad")
    with pytest.raises(ValueError, match="norm"):
        validate_config("norm", cfg)


def test_packer_conversion_preserves_rvq_stages():
    cfg = {"x": TensorConfig("rvq", 4, 2, "block-max", False, "policy", 1, "ok")}
    assert to_packer_allocation(cfg, group_size=8)["tensors"]["x"]["stages"] == [16, 16]


def test_packer_conversion_marks_int8_dense():
    cfg = {"h": TensorConfig("int8", 8, 0, "block-max", False, "policy", 1, "head")}
    alloc = to_packer_allocation(cfg, group_size=8)
    assert "h" not in alloc["tensors"]
    assert alloc["dense"]["h"]["method"] == "int8"


def test_validate_configs_rejects_unsafe_map():
    cfgs = {"lm_head.weight": TensorConfig("rvq", 4, 2, "block-max", False, "llm", 0.8, "bad")}
    with pytest.raises(ValueError, match="out-head"):
        validate_configs(cfgs, {"lm_head.weight": "out-head"})
