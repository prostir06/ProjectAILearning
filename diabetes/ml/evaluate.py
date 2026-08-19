"""
Метрики, Youden-поріг і важливість ознак після навчання.
"""

from __future__ import annotations

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from diabetes.core.config import (
    FEATURE_LABELS_UK,
    FEATURES,
    PREDICTION_THRESHOLD,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
)
from diabetes.core.exceptions import DataLoadError


def predict_probabilities(pipeline: Pipeline, x_data) -> np.ndarray:
    """Повертає ймовірності позитивного класу."""
    try:
        probabilities = pipeline.predict_proba(x_data)[:, 1]
    except (AttributeError, IndexError, ValueError) as exc:
        raise DataLoadError(
            "Модель повернула некоректний результат під час оцінки."
        ) from exc
    return np.asarray(probabilities, dtype=float)


def evaluate_model(
    pipeline: Pipeline,
    x_test,
    y_test,
    threshold: float = PREDICTION_THRESHOLD,
) -> dict:
    """
    Обчислює метрики якості для однієї моделі.

    Returns:
        Словник із accuracy, error_rate, precision, recall, f1, roc_auc, pr_auc.
    """
    try:
        probabilities = predict_probabilities(pipeline, x_test)
        predictions = (probabilities >= float(threshold)).astype(int)
        accuracy = accuracy_score(y_test, predictions)
    except (TypeError, ValueError) as exc:
        raise DataLoadError(
            "Модель повернула некоректний результат під час оцінки."
        ) from exc

    try:
        roc_auc = round(roc_auc_score(y_test, probabilities), 4)
    except ValueError:
        roc_auc = 0.0

    try:
        pr_auc = round(average_precision_score(y_test, probabilities), 4)
    except ValueError:
        pr_auc = 0.0

    return {
        "accuracy": round(accuracy, 4),
        "error_rate": round(1 - accuracy, 4),
        "precision": round(
            precision_score(y_test, predictions, zero_division=0),
            4,
        ),
        "recall": round(
            recall_score(y_test, predictions, zero_division=0),
            4,
        ),
        "f1": round(
            f1_score(y_test, predictions, zero_division=0),
            4,
        ),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": round(float(threshold), 4),
    }


def select_best_model_key(metrics_by_model: dict[str, dict]) -> str:
    """
    Обирає найкращу модель за композитним балом.

    Raises:
        DataLoadError: Якщо словник метрик порожній.
    """
    candidate_keys = [
        key
        for key, value in metrics_by_model.items()
        if not key.startswith("_") and isinstance(value, dict)
    ]
    if not candidate_keys:
        raise DataLoadError(
            "Немає навчених моделей для вибору найкращої."
        )

    try:
        return max(
            candidate_keys,
            key=lambda key: float(
                metrics_by_model[key].get("selection_score", 0.0)
            ),
        )
    except (TypeError, ValueError) as exc:
        raise DataLoadError(
            "Некоректні selection_score у метриках моделей."
        ) from exc


def find_optimal_threshold(
    pipeline: Pipeline,
    x_val,
    y_val,
) -> float:
    """Підбирає поріг за критерієм Youden J на validation."""
    probabilities = predict_probabilities(pipeline, x_val)
    thresholds = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, num=81)

    best_threshold = float(PREDICTION_THRESHOLD)
    best_score = float("-inf")

    y_true = np.asarray(y_val, dtype=int)

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tp = int(((predictions == 1) & (y_true == 1)).sum())
        fp = int(((predictions == 1) & (y_true == 0)).sum())
        tn = int(((predictions == 0) & (y_true == 0)).sum())
        fn = int(((predictions == 0) & (y_true == 1)).sum())

        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        youden_j = tpr - fpr

        if youden_j > best_score or (
            np.isclose(youden_j, best_score)
            and abs(threshold - PREDICTION_THRESHOLD)
            < abs(best_threshold - PREDICTION_THRESHOLD)
        ):
            best_score = youden_j
            best_threshold = float(threshold)

    return round(best_threshold, 4)


def map_transformed_name_to_feature(transformed_name: str) -> str:
    """Повертає базову назву ознаки з імені після препроцесора."""
    for feature in FEATURES:
        if transformed_name == feature:
            return feature
        if transformed_name.startswith(f"num__{feature}"):
            return feature
        if transformed_name.startswith(f"cat__{feature}_"):
            return feature
    return transformed_name


def normalize_importance_rows(
    raw_names,
    importances,
    top_n: int = 8,
) -> list[dict]:
    """Агрегує важливості після one-hot назад до базових ознак."""
    mapped: dict[str, float] = {}

    for name, importance in zip(raw_names, importances, strict=False):
        base_feature = map_transformed_name_to_feature(str(name))
        mapped[base_feature] = mapped.get(base_feature, 0.0) + float(importance)

    ranked = sorted(mapped.items(), key=lambda item: item[1], reverse=True)
    total = sum(value for _, value in ranked) or 1.0

    return [
        {
            "feature": feature,
            "label_uk": FEATURE_LABELS_UK.get(feature, feature),
            "importance": round(value / total, 4),
        }
        for feature, value in ranked[:top_n]
    ]


def extract_feature_importance(
    pipeline: Pipeline,
    top_n: int = 8,
) -> list[dict]:
    """Витягує важливість ознак із pipeline (tree-based)."""
    classifier = pipeline.named_steps.get("classifier")
    preprocessor = pipeline.named_steps.get("preprocessor")

    if classifier is None or not hasattr(classifier, "feature_importances_"):
        return []

    try:
        raw_names = preprocessor.get_feature_names_out()
    except (AttributeError, TypeError, ValueError):
        raw_names = [f"feature_{index}" for index in range(len(FEATURES))]

    try:
        importances = classifier.feature_importances_
    except (AttributeError, TypeError):
        return []

    return normalize_importance_rows(raw_names, importances, top_n=top_n)


def compute_permutation_importance(
    pipeline: Pipeline,
    x_data,
    y_data,
    top_n: int = 8,
):
    """Рахує permutation importance як fallback."""
    if x_data is None or y_data is None or len(x_data) == 0:
        return []

    sample_size = min(len(x_data), 1000)
    x_sample = x_data
    y_sample = y_data
    if sample_size < len(x_data):
        x_sample, _, y_sample, _ = train_test_split(
            x_data,
            y_data,
            train_size=sample_size,
            random_state=42,
            stratify=y_data,
        )

    try:
        result = permutation_importance(
            pipeline,
            x_sample,
            y_sample,
            n_repeats=5,
            random_state=42,
            scoring="roc_auc",
            n_jobs=1,
        )
    except Exception:  # pragma: no cover - defensive
        return []

    try:
        raw_names = list(x_sample.columns)
    except AttributeError:
        raw_names = FEATURES

    return normalize_importance_rows(
        raw_names,
        result.importances_mean,
        top_n=top_n,
    )
