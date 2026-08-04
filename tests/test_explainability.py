"""
Unit-тести для explainability.py.
"""

from unittest.mock import patch

from diabetes.ml.explainability import get_explanation


def test_get_explanation_filters_invalid_items():
    """Повертає лише валідні записи з числовим importance."""
    with patch(
        "diabetes.ml.explainability.get_feature_importance",
        return_value=[
            {"feature": "age", "label_uk": "Вік", "importance": 0.4},
            {"feature": "bmi", "importance": "bad"},
            "not-a-dict",
            {"feature": "x", "label_uk": "X", "importance": 0.1},
        ],
    ):
        result = get_explanation()

    assert len(result) == 2
    assert result[0]["label_uk"] == "Вік"
    assert result[1]["importance"] == 0.1


def test_get_explanation_handles_backend_error():
    """Помилка читання артефактів дає порожній список."""
    with patch(
        "diabetes.ml.explainability.get_feature_importance",
        side_effect=RuntimeError("boom"),
    ):
        assert get_explanation() == []


def test_get_explanation_non_list_returns_empty():
    """Некоректний тип відповіді → []."""
    with patch(
        "diabetes.ml.explainability.get_feature_importance",
        return_value={"not": "list"},
    ):
        assert get_explanation() == []
