"""
Хелпери Flask UI: форма, поріг, метрики, повідомлення про помилки.
"""

from __future__ import annotations

import logging

from diabetes.core.config import (
    DEFAULT_FORM,
    DEFAULT_THRESHOLD_PERCENT,
    PREDICTION_THRESHOLD,
    SMOKING_OPTIONS_UK,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP_PERCENT,
)
from diabetes.core.exceptions import (
    DiabetesProjectError,
    InvalidPatientDataError,
    ModelNotFoundError,
    PredictionError,
)
from diabetes.core.scoring import get_selection_score
from diabetes.core.validators import parse_prediction_threshold
from diabetes.ml.predict import (
    get_bundle_optimal_threshold,
    get_feature_importance,
    get_training_metrics,
)
from diabetes.ml.registry import MODEL_LABELS_UK

logger = logging.getLogger(__name__)


def parse_form(form_data) -> dict:
    """
    Зчитує дані з ``request.form`` у словник для валідації.

    Відсутні ключі заповнюються з ``DEFAULT_FORM``.
    При пошкодженому form_data повертає копію defaults без винятку.
    """
    parsed = DEFAULT_FORM.copy()
    try:
        for key in DEFAULT_FORM:
            if key in form_data:
                parsed[key] = form_data.get(key, parsed[key])
        parsed["smoking_history"] = form_data.get(
            "smoking_history",
            parsed["smoking_history"],
        )
    except (TypeError, AttributeError) as exc:
        logger.warning("Некоректні дані форми: %s", exc)
        return DEFAULT_FORM.copy()
    return parsed


def parse_threshold_from_form(
    form_data,
    default: float = PREDICTION_THRESHOLD,
) -> float:
    """
    Зчитує поріг ймовірності з HTML-форми (поле в %).

    Некоректне значення тихо замінюється на ``default``.
    """
    if form_data is None or "prediction_threshold" not in form_data:
        return default

    try:
        return parse_prediction_threshold(
            form_data.get("prediction_threshold"),
            default=default,
        )
    except InvalidPatientDataError:
        return default


def parse_threshold_from_payload(
    value,
    default: float = PREDICTION_THRESHOLD,
) -> float:
    """
    Приймає threshold з JSON API як частку 0–1 або як відсотки.

    Raises:
        InvalidPatientDataError: Якщо значення не число або поза діапазоном.
    """
    if value is None:
        return default

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidPatientDataError(
            "Поріг ймовірності має бути числом."
        ) from exc

    if 0.0 <= numeric <= 1.0:
        threshold = numeric
    else:
        threshold = parse_prediction_threshold(numeric, default=default)

    if not THRESHOLD_MIN <= threshold <= THRESHOLD_MAX:
        raise InvalidPatientDataError(
            f"Поріг має бути в діапазоні "
            f"{int(THRESHOLD_MIN * 100)}–{int(THRESHOLD_MAX * 100)}% "
            f"або {THRESHOLD_MIN:.1f}–{THRESHOLD_MAX:.1f}."
        )
    return round(threshold, 2)


def get_error_message(error: Exception) -> str:
    """Перетворює виняток на зрозуміле повідомлення українською."""
    if isinstance(error, InvalidPatientDataError):
        return str(error)
    if isinstance(error, ModelNotFoundError):
        return str(error)
    if isinstance(error, PredictionError):
        return f"Не вдалося виконати передбачення: {error}"
    if isinstance(error, DiabetesProjectError):
        return str(error)

    return "Сталася непередбачена помилка. Спробуйте ще раз."


def format_metrics_for_display(metrics: dict) -> list[dict]:
    """
    Готує метрики для таблиці порівняння алгоритмів.

    Пропускає службові ключі (``_meta``) і биті записи.
    Сортує за ``selection_score`` (вище — краще) і додає ``rank``.
    """
    if not isinstance(metrics, dict):
        return []

    rows = []
    for model_key, model_metrics in metrics.items():
        if model_key.startswith("_"):
            continue
        if not isinstance(model_metrics, dict):
            continue

        try:
            rows.append({
                "model_key": model_key,
                "model_name": model_metrics.get(
                    "label_uk",
                    MODEL_LABELS_UK.get(model_key, model_key),
                ),
                "accuracy": model_metrics.get("accuracy"),
                "error_rate": model_metrics.get("error_rate"),
                "precision": model_metrics.get("precision"),
                "recall": model_metrics.get("recall"),
                "f1": model_metrics.get("f1"),
                "roc_auc": model_metrics.get("roc_auc"),
                "pr_auc": model_metrics.get("pr_auc"),
                "is_best": model_metrics.get("is_best", False),
                "tuned": model_metrics.get("tuned", False),
                "selection_score": get_selection_score(model_metrics),
            })
        except (TypeError, AttributeError) as exc:
            logger.warning(
                "Пропущено некоректні метрики для %s: %s",
                model_key,
                exc,
            )

    try:
        rows.sort(key=lambda row: -float(row.get("selection_score") or 0))
    except (TypeError, ValueError):
        pass

    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def load_metrics_rows() -> list[dict]:
    """Безпечно завантажує метрики; при помилці — порожній список."""
    try:
        return format_metrics_for_display(get_training_metrics())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не вдалося завантажити метрики: %s", exc)
        return []


def load_feature_importance() -> list[dict]:
    """Безпечно завантажує важливість ознак; при помилці — []."""
    try:
        return get_feature_importance()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не вдалося завантажити важливість ознак: %s", exc)
        return []


def get_default_threshold() -> float:
    """
    Повертає optimal threshold з бандла або ``PREDICTION_THRESHOLD``.

    Ніколи не піднімає виняток назовні.
    """
    try:
        return get_bundle_optimal_threshold(default=PREDICTION_THRESHOLD)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не вдалося прочитати optimal_threshold: %s", exc)
        return PREDICTION_THRESHOLD


def build_index_context(
    *,
    form: dict | None = None,
    results: list[dict] | None = None,
    summary: dict | None = None,
    error: str | None = None,
    threshold_percent: int = DEFAULT_THRESHOLD_PERCENT,
    metrics_rows: list[dict] | None = None,
    feature_importance: list[dict] | None = None,
) -> dict:
    """Будує контекст Jinja для головної сторінки."""
    try:
        resolved_metrics = (
            metrics_rows if metrics_rows is not None else load_metrics_rows()
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("build_index_context: метрики недоступні: %s", exc)
        resolved_metrics = []

    try:
        resolved_importance = (
            feature_importance
            if feature_importance is not None
            else load_feature_importance()
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("build_index_context: importance недоступна: %s", exc)
        resolved_importance = []

    try:
        threshold_percent = int(threshold_percent)
    except (TypeError, ValueError):
        threshold_percent = DEFAULT_THRESHOLD_PERCENT

    return {
        "form": form or DEFAULT_FORM.copy(),
        "results": results,
        "summary": summary,
        "metrics_rows": resolved_metrics,
        "feature_importance": resolved_importance,
        "error": error,
        "threshold_percent": threshold_percent,
        "threshold_min_percent": int(THRESHOLD_MIN * 100),
        "threshold_max_percent": int(THRESHOLD_MAX * 100),
        "threshold_step": THRESHOLD_STEP_PERCENT,
        "smoking_options": SMOKING_OPTIONS_UK,
        "show_pr_auc": any(
            row.get("pr_auc") is not None for row in resolved_metrics
        ),
    }
