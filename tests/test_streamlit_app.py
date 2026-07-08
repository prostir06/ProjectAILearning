"""
Unit-тести для streamlit_app.py (без запуску UI-сесії).
"""

from unittest.mock import patch

import pytest

pytest.importorskip("streamlit")

import streamlit_app


def test_streamlit_app_exposes_main():
    """Модуль Streamlit має точку входу main."""
    assert callable(streamlit_app.main)


def test_ensure_models_ready_when_file_exists(tmp_path):
    """Якщо joblib уже є — навчання не запускається."""
    model_file = tmp_path / "diabetes_models.joblib"
    model_file.write_bytes(b"stub")

    with patch.object(streamlit_app, "MODELS_BUNDLE_PATH", model_file):
        streamlit_app.ensure_models_ready.clear()
        assert streamlit_app.ensure_models_ready() is True
