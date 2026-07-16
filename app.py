"""
Веб-інтерфейс Flask для передбачення діабету.

Показує передбачення від кількох алгоритмів ML,
загальний підсумок та метрики похибки на тестовій вибірці.
"""

import logging

from flask import Flask, render_template, request

from config import (
    BEST_MODEL_WEIGHTS,
    DEFAULT_THRESHOLD_PERCENT,
    PREDICTION_THRESHOLD,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP_PERCENT,
    DEFAULT_FORM,
)
from exceptions import (
    DiabetesProjectError,
    InvalidPatientDataError,
    ModelNotFoundError,
    PredictionError,
)
from model_registry import MODEL_LABELS_UK
from predict_diabetes import (
    get_feature_importance,
    get_training_metrics,
    predict_with_summary,
)
from validators import parse_prediction_threshold, validate_person_data

# Логер для несподіваних помилок сервера.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def parse_form(form_data) -> dict:
    """
    Зчитує дані з Flask request.form у словник для валідації.

    Відсутні поля заповнюються значеннями з DEFAULT_FORM
    (наприклад, smoking_history = «No Info»).

    Args:
        form_data: ImmutableMultiDict з полями HTML-форми.

    Returns:
        Словник із сирими значеннями полів.
    """
    parsed = DEFAULT_FORM.copy()
    try:
        for key in DEFAULT_FORM:
            if key in form_data:
                parsed[key] = form_data.get(key, parsed[key])
    except (TypeError, AttributeError) as exc:
        logger.warning("Некоректні дані форми: %s", exc)
        return DEFAULT_FORM.copy()
    return parsed


def parse_threshold_from_form(form_data, default: float = PREDICTION_THRESHOLD) -> float:
    """
    Зчитує поріг ймовірності з форми (поле prediction_threshold у %).

    Args:
        form_data: ImmutableMultiDict з полями HTML-форми.
        default: Поріг, якщо поле відсутнє.

    Returns:
        Поріг у діапазоні 0.0–1.0.
    """
    if form_data is None or "prediction_threshold" not in form_data:
        return default

    try:
        return parse_prediction_threshold(form_data.get("prediction_threshold"))
    except InvalidPatientDataError:
        return default


def get_error_message(error: Exception) -> str:
    """
    Перетворює виняток на зрозуміле повідомлення для користувача.

    Args:
        error: Перехоплений виняток.

    Returns:
        Текст помилки українською.
    """
    if isinstance(error, InvalidPatientDataError):
        return str(error)
    if isinstance(error, ModelNotFoundError):
        return str(error)
    if isinstance(error, PredictionError):
        return f"Не вдалося виконати передбачення: {error}"
    if isinstance(error, DiabetesProjectError):
        return str(error)

    return "Сталася непередбачена помилка. Спробуйте ще раз."


def _get_selection_score(model_metrics: dict) -> float:
    """
    Повертає композитний бал рейтингу алгоритму.

    Якщо selection_score уже збережено в метриках — використовує його.
    Інакше обчислює за формулою з BEST_MODEL_WEIGHTS (ROC-AUC, Recall, F1).

    Args:
        model_metrics: Словник метрик однієї моделі.

    Returns:
        Бал рейтингу (вище — краще).
    """
    if not isinstance(model_metrics, dict):
        return 0.0

    stored = model_metrics.get("selection_score")
    if stored is not None:
        try:
            return float(stored)
        except (TypeError, ValueError):
            return 0.0

    try:
        roc_auc = model_metrics.get("roc_auc")
        recall = model_metrics.get("recall")
        f1 = model_metrics.get("f1")
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


def format_metrics_for_display(metrics: dict) -> list[dict]:
    """
    Готує метрики для таблиці порівняння алгоритмів.

    Сортування: за selection_score (рейтинг) від найвищого до найнижчого.

    Args:
        metrics: Словник метрик із JSON або пакета моделей.

    Returns:
        Відсортований список записів для шаблону з полем rank.
    """
    if not isinstance(metrics, dict):
        return []

    rows = []
    for model_key, model_metrics in metrics.items():
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
                "is_best": model_metrics.get("is_best", False),
                "tuned": model_metrics.get("tuned", False),
                "selection_score": _get_selection_score(model_metrics),
            })
        except (TypeError, AttributeError) as exc:
            logger.warning(
                "Пропущено некоректні метрики для %s: %s",
                model_key,
                exc,
            )

    rows.sort(
        key=lambda row: -row.get("selection_score", 0),
    )

    for index, row in enumerate(rows, start=1):
        row["rank"] = index

    return rows


def load_metrics_rows() -> list[dict]:
    """
    Безпечно завантажує метрики для відображення на головній сторінці.

    Returns:
        Список рядків таблиці або порожній список при помилці.
    """
    try:
        return format_metrics_for_display(get_training_metrics())
    except Exception as exc:
        logger.warning("Не вдалося завантажити метрики: %s", exc)
        return []


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Головна сторінка: форма, порівняння алгоритмів і результати.

    Returns:
        HTML-сторінка з формою та (за наявності) результатами передбачення.
    """
    results = None
    summary = None
    error = None
    form = DEFAULT_FORM.copy()
    threshold = PREDICTION_THRESHOLD
    threshold_percent = DEFAULT_THRESHOLD_PERCENT
    metrics_rows = load_metrics_rows()
    feature_importance = []
    try:
        feature_importance = get_feature_importance()
    except Exception as exc:
        logger.warning("Не вдалося завантажити важливість ознак: %s", exc)

    if request.method == "POST":
        form = parse_form(request.form)
        threshold = parse_threshold_from_form(request.form)
        threshold_percent = int(threshold * 100)
        try:
            person = validate_person_data(form)
            prediction = predict_with_summary(person, threshold=threshold)
            results = prediction["models"]
            summary = prediction["summary"]
        except (InvalidPatientDataError, ModelNotFoundError, PredictionError) as exc:
            error = get_error_message(exc)
        except Exception as exc:
            logger.exception("Несподівана помилка під час передбачення")
            error = get_error_message(exc)

    return render_template(
        "index.html",
        form=form,
        results=results,
        summary=summary,
        metrics_rows=metrics_rows,
        feature_importance=feature_importance,
        error=error,
        threshold_percent=threshold_percent,
        threshold_min_percent=int(THRESHOLD_MIN * 100),
        threshold_max_percent=int(THRESHOLD_MAX * 100),
        threshold_step=THRESHOLD_STEP_PERCENT,
    )


@app.errorhandler(500)
def handle_internal_error(error):
    """
    Глобальний обробник несподіваних помилок сервера.

    Логує виняток і повертає сторінку з формою та повідомленням.
    """
    logger.exception("Внутрішня помилка сервера: %s", error)
    return render_template(
        "index.html",
        form=DEFAULT_FORM.copy(),
        results=None,
        summary=None,
        metrics_rows=[],
        feature_importance=[],
        error=get_error_message(error),
        threshold_percent=DEFAULT_THRESHOLD_PERCENT,
        threshold_min_percent=int(THRESHOLD_MIN * 100),
        threshold_max_percent=int(THRESHOLD_MAX * 100),
        threshold_step=THRESHOLD_STEP_PERCENT,
    ), 500


if __name__ == "__main__":
    import os

    # Для локальної розробки: FLASK_DEBUG=1. За замовчуванням debug вимкнено.
    # HOST=0.0.0.0 потрібен у Docker, щоб порт був доступний ззовні контейнера.
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = os.environ.get("HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("PORT", "5000"))
    except ValueError:
        logger.warning("Некоректний PORT у середовищі, використано 5000")
        port = 5000
    app.run(debug=debug_mode, host=host, port=port)
