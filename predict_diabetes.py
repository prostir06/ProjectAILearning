"""
Передбачення наявності діабету для однієї людини.

Підтримує кілька алгоритмів ML: передбачення можна отримати
як від однієї моделі за замовчуванням, так і від усіх одразу.
"""

import json

import joblib
import pandas as pd

from config import (
    BEST_MODEL_WEIGHTS,
    FEATURE_IMPORTANCE_PATH,
    FEATURES,
    METRICS_PATH,
    MODELS_BUNDLE_PATH,
    PREDICTION_THRESHOLD,
)
from exceptions import ModelNotFoundError, PredictionError
from model_registry import DEFAULT_MODEL_KEY, MODEL_LABELS_UK
from validators import validate_person_data

# Кеш завантаженого пакета моделей.
_bundle = None


def reset_pipeline_cache() -> None:
    """Скидає кеш моделей (корисно для тестів)."""
    global _bundle
    _bundle = None


def _get_bundle() -> dict:
    """
    Завантажує пакет моделей із диска (з кешуванням).

    Returns:
        Словник із ключами models, metrics, default_model, model_labels.

    Raises:
        ModelNotFoundError: Якщо файл моделей відсутній.
        PredictionError: Якщо файл пошкоджений.
    """
    global _bundle

    if _bundle is not None:
        return _bundle

    if not MODELS_BUNDLE_PATH.exists():
        raise ModelNotFoundError(
            "Моделі не знайдено. Спочатку запустіть: "
            "python train_diabetes_model.py"
        )

    try:
        _bundle = joblib.load(MODELS_BUNDLE_PATH)
    except Exception as exc:
        raise PredictionError(
            f"Не вдалося завантажити моделі з {MODELS_BUNDLE_PATH}."
        ) from exc

    if "models" not in _bundle or not isinstance(_bundle["models"], dict):
        raise PredictionError("Файл моделей має некоректний формат.")

    if not _bundle["models"]:
        raise PredictionError("Пакет моделей не містить жодного алгоритму.")

    return _bundle


def _get_default_model_key() -> str:
    """Повертає ключ найкращої або основної моделі з пакета."""
    bundle = _get_bundle()
    return bundle.get("best_model") or bundle.get("default_model", DEFAULT_MODEL_KEY)


def _get_pipeline(model_key: str | None = None):
    """
    Повертає pipeline конкретного алгоритму.

    Args:
        model_key: Ключ алгоритму.

    Returns:
        Навчений sklearn Pipeline.

    Raises:
        PredictionError: Якщо модель з таким ключем відсутня.
    """
    bundle = _get_bundle()
    models = bundle["models"]
    resolved_key = model_key or _get_default_model_key()

    if resolved_key not in models:
        raise PredictionError(
            f"Алгоритм «{resolved_key}» не знайдено в пакеті."
        )

    return models[resolved_key]


def _get_selection_score(model_metrics: dict) -> float:
    """Композитний бал рейтингу (ROC-AUC + Recall + F1)."""
    stored = model_metrics.get("selection_score")
    if stored is not None:
        return float(stored)

    roc_auc = model_metrics.get("roc_auc")
    recall = model_metrics.get("recall")
    f1 = model_metrics.get("f1")
    if roc_auc is None or recall is None or f1 is None:
        return 0.0

    return round(
        BEST_MODEL_WEIGHTS["roc_auc"] * roc_auc
        + BEST_MODEL_WEIGHTS["recall"] * recall
        + BEST_MODEL_WEIGHTS["f1"] * f1,
        4,
    )


def _run_prediction(pipeline, feature_frame: pd.DataFrame) -> dict:
    """
    Виконує predict і predict_proba для одного pipeline.

    Args:
        pipeline: Навчений sklearn Pipeline.
        feature_frame: DataFrame з одним рядком ознак.

    Returns:
        Словник diabetes, label, probability.

    Raises:
        PredictionError: Якщо модель повернула неочікуваний результат.
    """
    try:
        prediction = int(pipeline.predict(feature_frame)[0])
        probability = float(pipeline.predict_proba(feature_frame)[0][1])
    except (IndexError, KeyError, AttributeError) as exc:
        raise PredictionError(
            "Модель повернула неочікуваний результат."
        ) from exc
    except Exception as exc:
        raise PredictionError(
            f"Помилка під час передбачення: {exc}"
        ) from exc

    return {
        "diabetes": prediction,
        "label": "Так" if prediction else "Ні",
        "probability": round(probability, 3),
    }


def get_training_metrics() -> dict:
    """
    Завантажує метрики похибки з JSON або з пакета моделей.

    Returns:
        Словник метрик по кожному алгоритму (може бути порожнім).
    """
    if METRICS_PATH.exists():
        try:
            data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    try:
        bundle = _get_bundle()
        return bundle.get("metrics", {})
    except (ModelNotFoundError, PredictionError):
        return {}


def get_feature_importance() -> list[dict]:
    """
    Завантажує важливість ознак найкращої моделі.

    Returns:
        Список ознак із вагами (може бути порожнім).
    """
    if FEATURE_IMPORTANCE_PATH.exists():
        try:
            data = json.loads(
                FEATURE_IMPORTANCE_PATH.read_text(encoding="utf-8")
            )
            if isinstance(data, list):
                return data
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    try:
        bundle = _get_bundle()
        return bundle.get("feature_importance", [])
    except (ModelNotFoundError, PredictionError):
        return []


def predict(person: dict, model_key: str | None = None) -> dict:
    """
    Робить передбачення одним алгоритмом (за замовчуванням — Random Forest).

    Args:
        person: Дані пацієнта.
        model_key: Ключ алгоритму.

    Returns:
        Словник diabetes, label, probability.
    """
    validated = validate_person_data(person)
    pipeline = _get_pipeline(model_key)
    feature_frame = pd.DataFrame([validated], columns=FEATURES)
    return _run_prediction(pipeline, feature_frame)


def predict_all(person: dict) -> list[dict]:
    """
    Робить передбачення всіма навченими алгоритмами.

    Args:
        person: Дані пацієнта.

    Returns:
        Список словників із полями model_key, model_name, diabetes,
        label, probability та error_rate (з навчання).
    """
    validated = validate_person_data(person)
    bundle = _get_bundle()
    models = bundle["models"]
    metrics = get_training_metrics()
    labels = bundle.get("model_labels", MODEL_LABELS_UK)
    feature_frame = pd.DataFrame([validated], columns=FEATURES)

    results = []
    for model_key, pipeline in models.items():
        try:
            prediction = _run_prediction(pipeline, feature_frame)
        except PredictionError as exc:
            raise PredictionError(
                f"Помилка алгоритму «{labels.get(model_key, model_key)}»: {exc}"
            ) from exc

        model_metrics = metrics.get(model_key, {})

        results.append({
            "model_key": model_key,
            "model_name": labels.get(model_key, model_key),
            "diabetes": prediction["diabetes"],
            "label": prediction["label"],
            "probability": prediction["probability"],
            "error_rate": model_metrics.get("error_rate"),
            "accuracy": model_metrics.get("accuracy"),
            "roc_auc": model_metrics.get("roc_auc"),
            "is_best": model_metrics.get("is_best", False),
            "selection_score": _get_selection_score(model_metrics),
        })

    results.sort(key=lambda item: -item.get("selection_score", 0))

    for index, item in enumerate(results, start=1):
        item["rank"] = index

    return results


def build_prediction_summary(
    results: list[dict],
    threshold: float = PREDICTION_THRESHOLD,
) -> dict:
    """
    Формує загальний підсумок за всіма алгоритмами.

    Підсумок базується на:
    - середній ймовірності діабету;
    - кількості голосів «Так» / «Ні»;
    - порозі PREDICTION_THRESHOLD для фінального класу.

    Args:
        results: Список передбачень окремих моделей.
        threshold: Поріг ймовірності для відповіді «Так».

    Returns:
        Словник із загальним результатом.

    Raises:
        PredictionError: Якщо список результатів порожній.
    """
    if not results:
        raise PredictionError("Немає результатів для формування підсумку.")

    total_models = len(results)
    votes_yes = sum(
        1 for item in results if item["probability"] >= threshold
    )
    votes_no = total_models - votes_yes
    average_probability = sum(
        item["probability"] for item in results
    ) / total_models
    diabetes = int(average_probability >= threshold)

    return {
        "model_name": "Загальний підсумок",
        "total_models": total_models,
        "votes_yes": votes_yes,
        "votes_no": votes_no,
        "votes_text": (
            f"{votes_yes} з {total_models} алгоритмів — «Так», "
            f"{votes_no} з {total_models} — «Ні»"
        ),
        "probability": round(average_probability, 3),
        "diabetes": diabetes,
        "label": "Так" if diabetes else "Ні",
    }


def apply_threshold_to_results(
    results: list[dict],
    threshold: float,
) -> list[dict]:
    """
    Перераховує мітки «Так»/«Ні» за заданим порогом ймовірності.

    Args:
        results: Список передбачень моделей.
        threshold: Поріг ймовірності (0.0–1.0).

    Returns:
        Оновлений список результатів.
    """
    for item in results:
        diabetes = int(item["probability"] >= threshold)
        item["diabetes"] = diabetes
        item["label"] = "Так" if diabetes else "Ні"
    return results


def predict_with_summary(
    person: dict,
    threshold: float = PREDICTION_THRESHOLD,
) -> dict:
    """
    Робить передбачення всіма моделями та повертає загальний підсумок.

    Args:
        person: Дані пацієнта.
        threshold: Поріг ймовірності для класифікації «Так»/«Ні».

    Returns:
        Словник із ключами models (список) та summary (загальний результат).

    Raises:
        InvalidPatientDataError: Якщо дані пацієнта некоректні.
        ModelNotFoundError: Якщо файл моделей відсутній.
        PredictionError: Якщо передбачення або підсумок не вдались.
    """
    try:
        model_results = predict_all(person)
        apply_threshold_to_results(model_results, threshold)
        summary = build_prediction_summary(model_results, threshold=threshold)
    except (ModelNotFoundError, PredictionError):
        raise
    except Exception as exc:
        raise PredictionError(
            f"Не вдалося сформувати підсумок передбачення: {exc}"
        ) from exc

    return {
        "models": model_results,
        "summary": summary,
    }


if __name__ == "__main__":
    example = {
        "gender": "Female",
        "age": 54.0,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "No Info",
        "bmi": 27.3,
        "HbA1c_level": 6.6,
        "blood_glucose_level": 140,
    }

    try:
        all_results = predict_all(example)
        summary = build_prediction_summary(all_results)
        print("Передбачення всіма алгоритмами:\n")
        for item in all_results:
            error_text = (
                f"{item['error_rate']:.1%}"
                if item["error_rate"] is not None
                else "н/д"
            )
            print(
                f"  {item['model_name']}: {item['label']} "
                f"({item['probability']:.1%}), похибка на тесті: {error_text}"
            )
        print(
            f"\nЗагальний підсумок: {summary['label']} "
            f"({summary['probability']:.1%})"
        )
        print(f"  {summary['votes_text']}")
    except (ModelNotFoundError, PredictionError) as error:
        print(f"Помилка: {error}")
