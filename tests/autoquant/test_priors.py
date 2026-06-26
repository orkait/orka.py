from orka.autoquant.priors import ROLE_PRIORS, SQNR_TARGET_DB


def test_output_head_is_int8_never_rvq():
    p = ROLE_PRIORS["out-head"]
    assert p["method"] == "int8"
    assert p["allow_rvq"] is False


def test_norm_and_bias_kept_fp16():
    assert ROLE_PRIORS["norm"]["method"] == "fp16"
    assert ROLE_PRIORS["bias"]["method"] == "fp16"


def test_in_embed_allows_rvq():
    assert ROLE_PRIORS["in-embed"]["allow_rvq"] is True


def test_sqnr_target():
    assert SQNR_TARGET_DB == 30.0
