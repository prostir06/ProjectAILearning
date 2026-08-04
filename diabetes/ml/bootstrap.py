"""
Спільний bootstrap моделей для Flask cold-start.

Якщо ``diabetes_models.joblib`` відсутній — навчає моделі (без тюнінгу
за замовчуванням) і зберігає повний + легкий (best-only) бандли.
Для продакшену рекомендується комітити готовий joblib, а не покладатися
на cold-start.
"""

from __future__ import annotations

import logging
import os

from diabetes.core.config import (
    BEST_MODELS_BUNDLE_PATH,
    MODELS_BUNDLE_PATH,
    QUICK_TRAIN_MAX_ROWS,
)
from diabetes.core.exceptions import DataLoadError
from diabetes.ml.predict import reset_pipeline_cache

logger = logging.getLogger(__name__)


def _resolve_max_rows() -> int:
    """
    Кількість рядків для швидкого навчання при першому старті.

    Returns:
        Додатне ціле; якщо env/константа некоректні — 20000.
    """
    max_rows = QUICK_TRAIN_MAX_ROWS
    if max_rows > 0:
        return max_rows

    raw = os.environ.get("QUICK_TRAIN_MAX_ROWS", "20000")
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Некоректний QUICK_TRAIN_MAX_ROWS=%r, використано 20000",
            raw,
        )
        return 20000

    return parsed if parsed > 0 else 20000


def ensure_models_ready(*, enable_tuning: bool = False) -> bool:
    """
    Гарантує наявність пакета моделей на диску.

    Args:
        enable_tuning: Чи запускати RandomizedSearchCV (повільніше).

    Returns:
        True, якщо повний бандл доступний після перевірки / навчання.

    Raises:
        RuntimeError: Якщо навчання або збереження не вдалось.
    """
    if MODELS_BUNDLE_PATH.exists():
        return True

    # Лінивий імпорт: уникаємо важкого sklearn-циклу при звичайному старті.
    from diabetes.ml.train import (
        save_feature_importance,
        save_metrics_json,
        save_models_bundle,
        train_all_models,
    )

    max_rows = _resolve_max_rows()
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
    except Exception as exc:  # noqa: BLE001
        # Несподівані збої (наприклад, несумісна версія xgboost).
        raise RuntimeError(
            f"Несподівана помилка cold-start навчання: {exc}"
        ) from exc

    try:
        reset_pipeline_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не вдалося скинути кеш моделей: %s", exc)

    # Best-only файл бажаний, але не обов'язковий для роботи UI.
    if not BEST_MODELS_BUNDLE_PATH.exists():
        logger.warning(
            "Легкий бандл %s не створено; повний бандл доступний",
            BEST_MODELS_BUNDLE_PATH.name,
        )

    return MODELS_BUNDLE_PATH.exists()
