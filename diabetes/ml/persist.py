"""
Збереження joblib-бандлів і JSON-артефактів навчання.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from diabetes.core.config import (
    BEST_MODELS_BUNDLE_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODELS_BUNDLE_PATH,
)
from diabetes.ml.data import compute_data_checksum
from diabetes.ml.registry import DEFAULT_MODEL_KEY, MODEL_LABELS_UK

try:
    import xgboost
except Exception:  # pragma: no cover - optional metadata only
    xgboost = None


def build_bundle_metadata(optimal_threshold: float | None) -> dict[str, object]:
    """Формує метадані бандла."""
    return {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": getattr(xgboost, "__version__", None),
        "data_checksum": compute_data_checksum(),
        "optimal_threshold": optimal_threshold,
    }


def save_models_bundle(
    models: dict[str, Pipeline],
    metrics: dict[str, dict],
    best_model_key: str,
    feature_importance: list[dict],
    bundle_path: Path = MODELS_BUNDLE_PATH,
    also_save_best: bool = True,
    optimal_threshold: float | None = None,
) -> None:
    """Зберігає пакет моделей і, за потреби, окремий best-only бандл."""
    metadata = build_bundle_metadata(optimal_threshold)
    bundle = {
        "models": models,
        "metrics": metrics,
        "default_model": best_model_key,
        "best_model": best_model_key,
        "model_labels": MODEL_LABELS_UK,
        "feature_importance": feature_importance,
        "metadata": metadata,
    }

    try:
        joblib.dump(bundle, bundle_path)
        if also_save_best:
            best_bundle = {
                **bundle,
                "models": {best_model_key: models[best_model_key]},
                "metrics": {
                    best_model_key: metrics.get(best_model_key, {}),
                    "_meta": metrics.get("_meta", {}),
                },
            }
            joblib.dump(best_bundle, BEST_MODELS_BUNDLE_PATH)
    except OSError as exc:
        raise OSError(
            f"Не вдалося зберегти моделі: {bundle_path}"
        ) from exc

    print(f"\nМоделі збережено: {bundle_path}")
    if also_save_best:
        print(f"Best-only бандл збережено: {BEST_MODELS_BUNDLE_PATH}")
    print(f"Найкраща модель: {MODEL_LABELS_UK[best_model_key]}")


def save_metrics_json(
    metrics: dict[str, dict],
    metrics_path: Path = METRICS_PATH,
) -> None:
    """Зберігає метрики в JSON для швидкого читання веб-інтерфейсом."""
    try:
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise OSError(
            f"Не вдалося зберегти метрики: {metrics_path}"
        ) from exc

    print(f"Метрики збережено: {metrics_path}")


def save_feature_importance(
    feature_importance: list[dict],
    importance_path: Path = FEATURE_IMPORTANCE_PATH,
) -> None:
    """Зберігає важливість ознак у JSON."""
    try:
        importance_path.write_text(
            json.dumps(feature_importance, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise OSError(
            f"Не вдалося зберегти важливість ознак: {importance_path}"
        ) from exc

    print(f"Важливість ознак збережено: {importance_path}")


def save_model(pipeline: Pipeline, model_path: Path = MODELS_BUNDLE_PATH) -> None:
    """Зберігає одну модель у вигляді пакета (зворотна сумісність із тестами)."""
    bundle = {
        "models": {DEFAULT_MODEL_KEY: pipeline},
        "metrics": {},
        "default_model": DEFAULT_MODEL_KEY,
        "best_model": DEFAULT_MODEL_KEY,
        "model_labels": MODEL_LABELS_UK,
        "feature_importance": [],
        "metadata": build_bundle_metadata(None),
    }

    try:
        joblib.dump(bundle, model_path)
    except OSError as exc:
        raise OSError(
            f"Не вдалося зберегти модель: {model_path}"
        ) from exc

    print(f"Модель збережено: {model_path}")
