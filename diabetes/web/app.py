"""
Flask UI та JSON API для передбачення діабету.

Шар представлення:
- ``GET /`` / ``POST /`` — HTML-форма, метрики, результати;
- ``GET /health`` — перевірка живості для Docker;
- ``POST /api/predict`` — JSON-передбачення (CSRF exempt);
- ``GET /api/explain`` — важливість ознак.
"""

from __future__ import annotations

import logging

from flask import Flask, jsonify, render_template, request

from diabetes.core.config import (
    BASE_DIR,
    DEFAULT_FORM,
    DEFAULT_THRESHOLD_PERCENT,
    FLASK_SECRET_KEY,
    MODELS_BUNDLE_PATH,
)
from diabetes.core.exceptions import (
    InvalidPatientDataError,
    ModelNotFoundError,
    PredictionError,
)
from diabetes.core.validators import validate_person_data
from diabetes.ml.explainability import get_explanation
from diabetes.ml.predict import predict_with_summary
from diabetes.web.forms import (
    build_index_context,
    get_default_threshold,
    get_error_message,
    load_feature_importance,
    load_metrics_rows,
    parse_form,
    parse_threshold_from_form,
    parse_threshold_from_payload,
)

try:
    from flask_wtf.csrf import CSRFProtect
except ImportError:  # pragma: no cover - optional dependency in tests
    CSRFProtect = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """
    Створює й налаштовує Flask-додаток.

    Templates / static беруться з кореня репозиторію (``BASE_DIR``),
    а не з каталогу пакета ``diabetes.web``.
    """
    templates_dir = BASE_DIR / "templates"
    static_dir = BASE_DIR / "static"
    try:
        if not templates_dir.is_dir():
            logger.warning("Каталог templates відсутній: %s", templates_dir)
        if not static_dir.is_dir():
            logger.warning("Каталог static відсутній: %s", static_dir)
    except OSError as exc:
        logger.warning("Не вдалося перевірити templates/static: %s", exc)

    application = Flask(
        __name__,
        template_folder=str(templates_dir),
        static_folder=str(static_dir),
    )
    application.secret_key = FLASK_SECRET_KEY

    csrf = None
    if CSRFProtect is not None:
        csrf = CSRFProtect(application)
    else:
        logger.warning("flask-wtf is not installed; CSRF protection disabled.")

    @application.route("/health", methods=["GET"])
    def health():
        """Health endpoint для Docker / load balancer."""
        try:
            models_ready = MODELS_BUNDLE_PATH.exists()
        except OSError as exc:
            logger.warning("health: не вдалося перевірити бандл: %s", exc)
            models_ready = False

        return jsonify({
            "status": "ok",
            "models_ready": models_ready,
        })

    @application.route("/api/predict", methods=["POST"])
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

    if csrf is not None:
        csrf.exempt(api_predict)

    @application.route("/api/explain", methods=["GET"])
    def api_explain():
        """Повертає важливість ознак (порожній масив допустимий)."""
        try:
            return jsonify(get_explanation())
        except Exception as exc:  # noqa: BLE001
            logger.exception("Несподівана помилка в /api/explain")
            return jsonify({"error": get_error_message(exc), "items": []}), 500

    @application.route("/", methods=["GET", "POST"])
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

        # Без синхронного cold-start: якщо бандла немає — показуємо підказку.
        try:
            models_missing = (
                request.method == "GET" and not MODELS_BUNDLE_PATH.exists()
            )
        except OSError as exc:
            logger.warning("index: не вдалося перевірити бандл: %s", exc)
            models_missing = request.method == "GET"

        if models_missing:
            error = (
                "Моделі не знайдено. Запустіть навчання: python train.py"
            )

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
            except (
                InvalidPatientDataError,
                ModelNotFoundError,
                PredictionError,
            ) as exc:
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

    @application.errorhandler(500)
    def handle_internal_error(error):
        """Глобальний обробник внутрішніх помилок сервера."""
        logger.exception("Внутрішня помилка сервера: %s", error)
        try:
            return render_template(
                "index.html",
                **build_index_context(error=get_error_message(error)),
            ), 500
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Не вдалося відрендерити сторінку помилки: %s",
                exc,
            )
            return get_error_message(error), 500

    return application
