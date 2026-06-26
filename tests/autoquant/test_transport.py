import pytest
from orka.autoquant.transport import route_model, make_llm_fn, NoLLMBackend, LITE_MODEL, STRONG_MODEL


def test_route_model_lite_vs_strong():
    assert route_model(hard=False) == LITE_MODEL
    assert route_model(hard=True) == STRONG_MODEL


def test_make_llm_fn_uses_injected_backend():
    seen = {}

    def fake_completion(model, messages):
        seen["model"] = model
        return '{"method":"int8","bits":8}'

    fn = make_llm_fn(hard=True, _backends=[fake_completion])
    assert fn([{"role": "user", "content": "x"}]) == '{"method":"int8","bits":8}'
    assert seen["model"] == STRONG_MODEL


def test_make_llm_fn_raises_without_backend():
    with pytest.raises(NoLLMBackend):
        make_llm_fn(_backends=[])
