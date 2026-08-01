"""
Тонкий шар для explainability API.
"""

from __future__ import annotations

from predict_diabetes import get_feature_importance


def get_explanation() -> list[dict]:
    """Повертає важливість ознак для найкращої моделі."""
    return get_feature_importance()
