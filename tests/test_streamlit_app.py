"""
Unit-тести для streamlit_app.py (без запуску UI-сесії).

Перевіряють HTML-хелпери, кешовані завантажувачі таблиць
і стійкість до пошкоджених / неповних даних.
"""

from unittest.mock import patch

import pandas as pd
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


def test_build_donut_html_compact_wrap_class():
    """compact=True додає клас compact до обгортки."""
    html = streamlit_app.build_donut_html(
        percent=40,
        threshold_percent=50,
        donut_label="ймовірність",
        is_positive=False,
        compact=True,
    )

    assert "st-donut-wrap compact" in html


def test_build_results_grid_html_uses_css_grid():
    """Сітка результатів містить клас st-results-grid."""
    models = [
        {
            "model_name": "Test",
            "rank": 1,
            "probability": 0.4,
            "diabetes": 0,
            "label": "Ні",
            "error_rate": 0.05,
        }
    ]
    html = streamlit_app.build_results_grid_html(models, threshold_percent=50)

    assert "st-results-grid" in html
    assert "st-model-card-title" in html
    assert "Похибка на тесті: 5.0%" in html


def test_build_results_grid_html_empty_models():
    """Порожній список моделей дає порожню сітку без карток."""
    html = streamlit_app.build_results_grid_html([], threshold_percent=50)

    assert html == '<div class="st-results-grid"></div>'


def test_build_model_card_html_invalid_item_returns_empty():
    """Некоректна картка не ламає сітку — повертає порожній рядок."""
    html = streamlit_app.build_model_card_html(
        {"model_name": "Broken"},
        threshold_percent=50,
    )

    assert html == ""


def test_build_model_card_html_escapes_title():
    """Назва моделі екранується від HTML-ін'єкції."""
    html = streamlit_app.build_model_card_html(
        {
            "model_name": "<script>x</script>",
            "rank": 2,
            "is_best": True,
            "probability": 0.8,
            "diabetes": 1,
            "label": "Так",
            "error_rate": None,
        },
        threshold_percent=40,
    )

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "найкраща" in html
    assert "st-model-card-positive" in html


def test_build_summary_block_html_contains_label():
    """Підсумок містить donut і підпис результату."""
    html = streamlit_app.build_summary_block_html(
        {"label": "Ні"},
        threshold_percent=50,
        summary_percent=22,
        summary_positive=False,
    )

    assert "st-summary-block" in html
    assert "22%" in html
    assert "Ні" in html
    assert "st-result-negative" in html


def test_escape_html_replaces_special_chars():
    """_escape_html екранує &, <, >, \"."""
    assert streamlit_app._escape_html('a&b<c>"') == "a&amp;b&lt;c&gt;&quot;"


def test_load_metrics_table_success():
    """load_metrics_table будує DataFrame з відсотковими колонками."""
    fake_rows = [
        {
            "rank": 1,
            "model_name": "RF",
            "selection_score": 0.9,
            "roc_auc": 0.95,
            "recall": 0.8,
            "f1": 0.85,
            "accuracy": 0.97,
            "error_rate": 0.03,
            "is_best": True,
            "tuned": False,
        }
    ]

    with patch(
        "streamlit_app.format_metrics_for_display",
        return_value=fake_rows,
    ):
        streamlit_app.load_metrics_table.clear()
        table = streamlit_app.load_metrics_table()

    assert isinstance(table, pd.DataFrame)
    assert not table.empty
    assert table.iloc[0]["Алгоритм"] == "RF"
    assert table.iloc[0]["Найкраща"] == "так"


def test_load_metrics_table_handles_exception():
    """Помилка завантаження метрик дає порожній DataFrame."""
    with patch(
        "streamlit_app.get_training_metrics",
        side_effect=RuntimeError("broken"),
    ):
        streamlit_app.load_metrics_table.clear()
        table = streamlit_app.load_metrics_table()

    assert isinstance(table, pd.DataFrame)
    assert table.empty


def test_load_importance_table_success():
    """load_importance_table будує таблицю важливості ознак."""
    items = [
        {"label_uk": "Вік", "importance": 0.25},
        {"label_uk": "ІМТ", "importance": 0.15},
    ]

    with patch("streamlit_app.get_feature_importance", return_value=items):
        streamlit_app.load_importance_table.clear()
        table = streamlit_app.load_importance_table()

    assert list(table["Ознака"]) == ["Вік", "ІМТ"]
    assert table.iloc[0]["Важливість %"] == 25.0


def test_load_importance_table_handles_exception():
    """Помилка читання важливості ознак дає порожній DataFrame."""
    with patch(
        "streamlit_app.get_feature_importance",
        side_effect=OSError("missing"),
    ):
        streamlit_app.load_importance_table.clear()
        table = streamlit_app.load_importance_table()

    assert table.empty


def test_build_donut_html_clamps_out_of_range_percent():
    """Відсотки поза 0–100 обмежуються."""
    html = streamlit_app.build_donut_html(
        percent=150,
        threshold_percent=-10,
        donut_label="ймовірність",
        is_positive=False,
    )

    assert "100%" in html
    assert "rotate(0" in html


def test_build_donut_html_escapes_label():
    """Підпис donut екранується від HTML."""
    html = streamlit_app.build_donut_html(
        percent=10,
        threshold_percent=50,
        donut_label="<b>x</b>",
        is_positive=False,
    )

    assert "<b>x</b>" not in html
    assert "&lt;b&gt;x&lt;/b&gt;" in html


def test_ensure_models_ready_training_failure(tmp_path):
    """Помилка навчання при відсутності joblib → RuntimeError."""
    missing = tmp_path / "missing_models.joblib"

    with patch.object(streamlit_app, "MODELS_BUNDLE_PATH", missing):
        streamlit_app.ensure_models_ready.clear()
        with patch(
            "train_diabetes_model.train_all_models",
            side_effect=OSError("disk full"),
        ):
            with pytest.raises(RuntimeError, match="першому запуску"):
                streamlit_app.ensure_models_ready()
