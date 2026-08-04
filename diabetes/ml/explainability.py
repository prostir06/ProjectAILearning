"""
Тонкий шар explainability для REST API та UI.

Повертає важливість ознак найкращої моделі з JSON / joblib-бандла.
Не тягне важкі залежності (SHAP) — достатньо для навчального демо.
"""

from __future__ import annotations

import logging

from diabetes.ml.predict import get_feature_importance

logger = logging.getLogger(__name__)


def get_explanation() -> list[dict]:
    """
    Безпечно повертає список важливості ознак.

    Returns:
        Список словників ``{feature, label_uk, importance}``.
        Порожній список, якщо артефакти відсутні або пошкоджені.
    """
    try:
        items = get_feature_importance()
    except Exception as exc:  # noqa: BLE001 — API не повинен падати
        logger.warning("Не вдалося отримати explainability: %s", exc)
        return []

    if not isinstance(items, list):
        return []

    cleaned: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            importance = float(item.get("importance", 0.0))
        except (TypeError, ValueError):
            continue
        cleaned.append({
            "feature": str(item.get("feature", "")),
            "label_uk": str(item.get("label_uk", item.get("feature", ""))),
            "importance": importance,
        })
    return cleaned
