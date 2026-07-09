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
