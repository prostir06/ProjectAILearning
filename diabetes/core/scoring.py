"""
Спільні функції рейтингу моделей (selection score).

Єдине джерело правди для train / predict / Flask UI.
Формула ваг задається в ``config.BEST_MODEL_WEIGHTS``.
"""

from __future__ import annotations

from diabetes.core.config import BEST_MODEL_WEIGHTS


def compute_selection_score(metrics: dict | None) -> float:
    """
    Рахує зважений бал з ``roc_auc`` / ``recall`` / ``f1``.

    Ігнорує вже збережений ``selection_score`` (для перерахунку під час навчання).

    Args:
        metrics: Словник метрик однієї моделі або None.

    Returns:
        Композитний бал (вище — краще) або 0.0 при некоректних даних.
    """
    if not isinstance(metrics, dict):
        return 0.0

    try:
        roc_auc = metrics.get("roc_auc")
        recall = metrics.get("recall")
        f1 = metrics.get("f1")
        if roc_auc is None or recall is None or f1 is None:
            return 0.0

        return round(
            BEST_MODEL_WEIGHTS["roc_auc"] * float(roc_auc)
            + BEST_MODEL_WEIGHTS["recall"] * float(recall)
            + BEST_MODEL_WEIGHTS["f1"] * float(f1),
            4,
        )
    except (TypeError, ValueError):
        return 0.0


def get_selection_score(metrics: dict | None) -> float:
    """
    Повертає збережений ``selection_score`` або обчислює його.

    Args:
        metrics: Словник метрик однієї моделі або None.

    Returns:
        Бал рейтингу або 0.0.
    """
    if not isinstance(metrics, dict):
        return 0.0

    stored = metrics.get("selection_score")
    if stored is not None:
        try:
            return float(stored)
        except (TypeError, ValueError):
            return 0.0

    return compute_selection_score(metrics)
