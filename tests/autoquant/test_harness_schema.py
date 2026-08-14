import pytest

from orka.autoquant.harness_schema import RunConfig


def test_run_config_rejects_non_positive_limits(tmp_path):
    with pytest.raises(ValueError, match="max_steps"):
        RunConfig("http://gratis", "gratis-auto", 0.01, 0, 60, "knee", 0.02, tmp_path)


def test_run_config_requires_gratis_endpoint(tmp_path):
    with pytest.raises(ValueError, match="Gratis"):
        RunConfig("", "gratis-auto", 0.01, 3, 60, "knee", 0.02, tmp_path)
