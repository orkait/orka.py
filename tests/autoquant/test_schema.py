from orka.autoquant.schema import TensorConfig, to_allocation_map, from_allocation_map


def test_roundtrip():
    cfgs = {
        "lm_head.weight": TensorConfig(method="int8", bits=8, stages=0,
                                       normalization="block-max", keep_fp16=False,
                                       source="policy", confidence=1.0, rationale="head"),
        "blk.0.mlp.down.weight": TensorConfig(method="rvq", bits=3, stages=2,
                                       normalization="block-max", keep_fp16=False,
                                       source="llm", confidence=0.6, rationale="sensitive"),
    }
    m = to_allocation_map(cfgs)
    assert m["lm_head.weight"]["method"] == "int8"
    assert m["blk.0.mlp.down.weight"]["stages"] == 2
    back = from_allocation_map(m)
    assert back["lm_head.weight"] == cfgs["lm_head.weight"]


def test_fp16_tensor_serializes_keep_fp16():
    c = TensorConfig(method="fp16", bits=16, stages=0, normalization="none",
                     keep_fp16=True, source="policy", confidence=1.0, rationale="norm")
    assert to_allocation_map({"x": c})["x"]["keep_fp16"] is True
