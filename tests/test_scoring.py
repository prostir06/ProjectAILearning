"""
Unit-тести для scoring.py — єдине джерело рейтингу моделей.
"""

from diabetes.core.scoring import compute_selection_score, get_selection_score


def test_compute_selection_score_weights():
    """Композитний бал враховує ваги ROC-AUC / Recall / F1."""
    metrics = {"roc_auc": 0.9, "recall": 0.8, "f1": 0.7}
    assert compute_selection_score(metrics) == round(
        0.5 * 0.9 + 0.3 * 0.8 + 0.2 * 0.7, 4
    )


def test_compute_ignores_stored_score():
    """compute_selection_score перераховує, навіть якщо score уже є."""
    metrics = {
        "roc_auc": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "selection_score": 0.1,
    }
    assert compute_selection_score(metrics) == 1.0


def test_get_prefers_stored_score():
    """get_selection_score бере збережене значення, якщо воно валідне."""
    assert get_selection_score({"selection_score": 0.42}) == 0.42


def test_get_selection_score_recomputes_when_stored_missing():
    """Без збереженого score — перерахунок з метрик."""
    assert get_selection_score(
        {"roc_auc": 0.8, "recall": 0.8, "f1": 0.8}
    ) == 0.8


def test_compute_and_get_tolerant_to_bad_input():
    """Некоректні метрики не падають — повертають 0.0."""
    assert compute_selection_score(None) == 0.0
    assert compute_selection_score({"roc_auc": "x", "recall": 1, "f1": 1}) == 0.0
    assert get_selection_score(None) == 0.0
    assert get_selection_score({"selection_score": "bad"}) == 0.0
