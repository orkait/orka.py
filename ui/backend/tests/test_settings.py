from ui.backend import settings


def test_defaults_present():
    assert settings.GPU_MEM_CAP_GB == 10.0          # orka 10GB cap rule
    assert settings.SCHEMA_VERSION >= 1
    assert settings.LIVE_PARAM_CEILING > 0
    assert isinstance(settings.HF_CACHE, str)
