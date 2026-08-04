"""
Веб-інтерфейс Flask та JSON API для передбачення діабету.

Шар представлення:
- ``GET /`` / ``POST /`` — HTML-форма, метрики, результати;
- ``GET /health`` — перевірка живості для Docker;
- ``POST /api/predict`` — JSON-передбачення (CSRF exempt);
- ``GET /api/explain`` — важливість ознак.

Усі ризикові операції (форма, метрики, моделі, рендер) обгорнуті в try/except,
щоб користувач бачив зрозуміле повідомлення замість traceback.
"""

from __future__ import annotations

import logging
import os

import bootstrap_models
from explainability import get_explanation
from flask import Flask, jsonify, render_template, request

from diabetes.core.config import (
    DEFAULT_FORM,
    DEFAULT_THRESHOLD_PERCENT,
    FLASK_SECRET_KEY,
    MODELS_BUNDLE_PATH,
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
from model_registry import MODEL_LABELS_UK
from predict_diabetes import (
    get_bundle_optimal_threshold,
    get_feature_importance,
    get_training_metrics,
    predict_with_summary,
)
from diabetes.core.scoring import get_selection_score
from diabetes.core.validators import parse_prediction_threshold, validate_person_data

# CSRF — опційна залежність (flask-wtf); без неї додаток працює з попередженням.
try:
    from flask_wtf.csrf import CSRFProtect
except ImportError:  # pragma: no cover - optional dependency in tests
    CSRFProtect = None

# Логування серверних збоїв (не показуємо traceback у HTML).
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Екземпляр Flask-додатка та секрет для сесій / CSRF.
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

csrf = None
if CSRFProtect is not None:
    csrf = CSRFProtect(app)
else:
    logger.warning("flask-wtf is not installed; CSRF protection disabled.")


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
        # smoking_history може бути відсутнім у старих клієнтах.
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

    # 0.0–1.0 трактуємо як частку; інакше — як відсотки (наприклад 30 → 0.30).
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
        # Якщо score нечисловий — лишаємо порядок як є.
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


@app.route("/health", methods=["GET"])
def health():
    """
    Health endpoint для Docker / load balancer.

    Повертає також прапорець наявності файлу моделей (без завантаження joblib).
    """
    try:
        models_ready = MODELS_BUNDLE_PATH.exists()
    except OSError as exc:
        logger.warning("health: не вдалося перевірити бандл: %s", exc)
        models_ready = False

    return jsonify({
        "status": "ok",
        "models_ready": models_ready,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API: валідація → predict_with_summary → JSON відповідь."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON body має бути об'єктом."}), 400

    threshold = get_default_threshold()
    try:
        threshold = parse_threshold_from_payload(
            payload.get("threshold"),
            default=threshold,
        )
        mode = str(payload.get("mode", "all")).strip().lower() or "all"
        person = {
            key: value
            for key, value in payload.items()
            if key not in {"threshold", "mode"}
        }
        validated_person = validate_person_data(person)
        prediction = predict_with_summary(
            validated_person,
            threshold=threshold,
            mode=mode,
        )
        return jsonify(prediction)
    except InvalidPatientDataError as exc:
        return jsonify({"error": str(exc)}), 400
    except ModelNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except PredictionError as exc:
        return jsonify({"error": get_error_message(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        logger.exception("Несподівана помилка в /api/predict")
        return jsonify({"error": get_error_message(exc)}), 500


# JSON API не використовує HTML-форми — CSRF не застосовуємо.
if csrf is not None:
    csrf.exempt(api_predict)


@app.route("/api/explain", methods=["GET"])
def api_explain():
    """Повертає важливість ознак (порожній масив допустимий)."""
    try:
        return jsonify(get_explanation())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Несподівана помилка в /api/explain")
        return jsonify({"error": get_error_message(exc), "items": []}), 500


@app.route("/", methods=["GET", "POST"])
def index():
    """Головна сторінка: форма, метрики, результати передбачення."""
    results = None
    summary = None
    error = None
    form = DEFAULT_FORM.copy()
    threshold = get_default_threshold()
    try:
        threshold_percent = int(round(threshold * 100))
    except (TypeError, ValueError):
        threshold_percent = DEFAULT_THRESHOLD_PERCENT

    # Cold-start: якщо бандла немає — пробуємо швидко навчити моделі.
    if request.method == "GET" and not MODELS_BUNDLE_PATH.exists():
        try:
            bootstrap_models.ensure_models_ready()
        except RuntimeError as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Cold-start моделей не вдався")
            error = get_error_message(exc)

    metrics_rows = load_metrics_rows()
    feature_importance = load_feature_importance()

    if request.method == "POST":
        form = parse_form(request.form)
        threshold = parse_threshold_from_form(request.form, default=threshold)
        try:
            threshold_percent = int(round(threshold * 100))
        except (TypeError, ValueError):
            threshold_percent = DEFAULT_THRESHOLD_PERCENT
        try:
            person = validate_person_data(form)
            prediction = predict_with_summary(person, threshold=threshold)
            results = prediction["models"]
            summary = prediction["summary"]
        except (InvalidPatientDataError, ModelNotFoundError, PredictionError) as exc:
            error = get_error_message(exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Несподівана помилка під час передбачення")
            error = get_error_message(exc)

    context = build_index_context(
        form=form,
        results=results,
        summary=summary,
        error=error,
        threshold_percent=threshold_percent,
        metrics_rows=metrics_rows,
        feature_importance=feature_importance,
    )
    try:
        return render_template("index.html", **context)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Помилка рендерингу шаблону: %s", exc)
        return get_error_message(exc), 500


@app.errorhandler(500)
def handle_internal_error(error):
    """Глобальний обробник внутрішніх помилок сервера."""
    logger.exception("Внутрішня помилка сервера: %s", error)
    try:
        return render_template(
            "index.html",
            **build_index_context(error=get_error_message(error)),
        ), 500
    except Exception as exc:  # noqa: BLE001
        logger.exception("Не вдалося відрендерити сторінку помилки: %s", exc)
        return get_error_message(error), 500


if __name__ == "__main__":
    # Локальний / Docker запуск: Waitress у проді, Flask reloader у debug.
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = os.environ.get("HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("PORT", "5000"))
    except ValueError:
        logger.warning("Некоректний PORT у середовищі, використано 5000")
        port = 5000

    if not debug_mode:
        try:
            from waitress import serve
        except ImportError:
            logger.warning("waitress не встановлено; використано Flask dev server.")
        else:
            serve(app, host=host, port=port)
            raise SystemExit(0)

    app.run(debug=debug_mode, host=host, port=port)
