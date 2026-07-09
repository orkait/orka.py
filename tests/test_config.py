from orka import config


def test_numeric_defaults(monkeypatch):
    for var in ("ORKA_PREFLIGHT_MIN_AVAIL_GB", "ORKA_PREFLIGHT_MAX_SWAP_GB", "ORKA_HARD_CEILING_GB"):
        monkeypatch.delenv(var, raising=False)
    assert config.preflight_min_avail_gb() == 5.0
    assert config.preflight_max_swap_gb() == 4.0
    assert config.hard_ceiling_gb() == 25.0


def test_numeric_override(monkeypatch):
    monkeypatch.setenv("ORKA_HARD_CEILING_GB", "12.5")
    assert config.hard_ceiling_gb() == 12.5


def test_awq_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ORKA_ENABLE_AWQ", raising=False)
    assert config.awq_enabled() is False


def test_awq_falsy_strings_stay_disabled(monkeypatch):
    """'0' and 'false' must not be truthy. A naive bool(os.environ.get(...))
    would enable AWQ for every non-empty value, including '0'."""
    for value in ("0", "false", "no", "off", "  "):
        monkeypatch.setenv("ORKA_ENABLE_AWQ", value)
        assert config.awq_enabled() is False, value


def test_awq_truthy_strings(monkeypatch):
    for value in ("1", "true", "TRUE", "yes", "on", " on "):
        monkeypatch.setenv("ORKA_ENABLE_AWQ", value)
        assert config.awq_enabled() is True, value


def test_kmeans_iters_defaults_to_caller(monkeypatch):
    monkeypatch.delenv("ORKA_KMEANS_ITERS", raising=False)
    assert config.kmeans_iters(7) == 7


def test_kmeans_iters_override(monkeypatch):
    monkeypatch.setenv("ORKA_KMEANS_ITERS", "3")
    assert config.kmeans_iters(7) == 3


def test_awq_parsing_matches_features_module(monkeypatch):
    """orka.core._features.awq_feature_enabled must agree with config."""
    from orka.core import _features

    for value in ("", "0", "1", "true", "off", "yes"):
        monkeypatch.setenv("ORKA_ENABLE_AWQ", value)
        assert _features.awq_feature_enabled() == config.awq_enabled(), value


def test_faiss_truthy_set_is_narrower_than_awq(monkeypatch):
    """ORKA_KMEANS_FAISS does not accept 'on'; ORKA_ENABLE_AWQ does. Unifying them
    would silently enable faiss for anyone who set it to 'on' expecting a no-op."""
    monkeypatch.setenv("ORKA_KMEANS_FAISS", "on")
    monkeypatch.setenv("ORKA_ENABLE_AWQ", "on")
    assert config.kmeans_faiss_enabled() is False
    assert config.awq_enabled() is True

    for value in ("1", "true", "yes"):
        monkeypatch.setenv("ORKA_KMEANS_FAISS", value)
        assert config.kmeans_faiss_enabled() is True, value


def test_llm_model_defaults(monkeypatch):
    monkeypatch.delenv("ORKA_LLM_LITE", raising=False)
    monkeypatch.delenv("ORKA_LLM_STRONG", raising=False)
    assert config.llm_lite_model() == "claude-sonnet-4-6"
    assert config.llm_strong_model() == "claude-opus-4-8"
    monkeypatch.setenv("ORKA_LLM_LITE", "custom")
    assert config.llm_lite_model() == "custom"
