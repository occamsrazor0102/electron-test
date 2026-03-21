import importlib
import sys
import types

import pytest


def _reload_dashboard_with(fake_streamlit):
    """Reload dashboard with a stubbed streamlit module."""
    original = sys.modules.get("streamlit")
    sys.modules["streamlit"] = fake_streamlit
    try:
        import llm_detector.dashboard as dash

        return importlib.reload(dash)
    finally:
        if original is not None:
            sys.modules["streamlit"] = original
        else:
            sys.modules.pop("streamlit", None)


def _fake_streamlit(**attrs):
    mod = types.ModuleType("streamlit")
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def test_rerun_prefers_modern_rerun():
    calls = []
    dash = _reload_dashboard_with(
        _fake_streamlit(
            rerun=lambda: calls.append("rerun"),
            experimental_rerun=lambda: calls.append("experimental"),
        )
    )

    dash._rerun()
    assert calls == ["rerun"]


def test_rerun_falls_back_to_experimental():
    calls = []
    dash = _reload_dashboard_with(
        _fake_streamlit(experimental_rerun=lambda: calls.append("experimental"))
    )

    dash._rerun()
    assert calls == ["experimental"]


def test_rerun_raises_if_unavailable():
    dash = _reload_dashboard_with(_fake_streamlit())

    with pytest.raises(RuntimeError, match="Streamlit rerun unavailable"):
        dash._rerun()
