"""
Спільний bootstrap моделей для Streamlit / Flask cold-start.

Якщо diabetes_models.joblib відсутній — навчає моделі (без тюнінгу)
і зберігає повний + легкий (best-only) бандли.
"""

from __future__ import annotations

import logging
import os

from config import (
    BEST_MODELS_BUNDLE_PATH,
    MODELS_BUNDLE_PATH,
    QUICK_TRAIN_MAX_ROWS,
)
from exceptions import DataLoadError
from predict_diabetes import reset_pipeline_cache

logger = logging.getLogger(__name__)


def ensure_models_ready(*, enable_tuning: bool = False) -> bool:
    """
    Гарантує наявність пакета моделей на диску.

    Returns:
        True, якщо моделі доступні.

    Raises:
        RuntimeError: Якщо навчання / збереження не вдалось.
    """
    if MODELS_BUNDLE_PATH.exists():
        return True

    from train_diabetes_model import (
        save_feature_importance,
        save_metrics_json,
        save_models_bundle,
        train_all_models,
    )

    max_rows = QUICK_TRAIN_MAX_ROWS
    if max_rows <= 0:
        # Cold-start за замовчуванням обмежує вибірку, щоб UI не зависав надовго.
        max_rows = int(os.environ.get("QUICK_TRAIN_MAX_ROWS", "20000"))

    logger.info(
        "Моделі відсутні — швидке навчання (max_rows=%s, tuning=%s)",
        max_rows,
        enable_tuning,
    )

    try:
        models, metrics, best_key, importance, optimal_threshold = (
            train_all_models(
            enable_tuning=enable_tuning,
            max_rows=max_rows,
        )
        )
        save_models_bundle(
            models,
            metrics,
            best_key,
            importance,
            also_save_best=True,
            optimal_threshold=optimal_threshold,
        )
        save_metrics_json(metrics)
        save_feature_importance(importance)
    except (DataLoadError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"Не вдалося навчити моделі при першому запуску: {exc}"
        ) from exc

    reset_pipeline_cache()
    return MODELS_BUNDLE_PATH.exists() and BEST_MODELS_BUNDLE_PATH.exists()
