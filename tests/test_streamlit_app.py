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


def test_build_donut_html_contains_threshold_and_percent():
    """Donut HTML містить відсоток, поріг і SVG-лінію."""
    html = streamlit_app.build_donut_html(
        percent=37,
        threshold_percent=50,
        donut_label="ймовірність",
        is_positive=False,
        small=True,
    )

    assert "37%" in html
    assert "rotate(180" in html
    assert "st-donut-positive" not in html
    assert "ймовірність" in html


def test_build_donut_html_positive_uses_orange_class():
    """Позитивний результат додає клас st-donut-positive."""
    html = streamlit_app.build_donut_html(
        percent=65,
        threshold_percent=30,
        donut_label="середня ймовірність",
        is_positive=True,
    )

    assert "st-donut-positive" in html
    assert "65%" in html
