"""
Навчання кількох моделей ML з валідацією, тюнінгом і фінальною оцінкою.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from config import (
    BEST_MODELS_BUNDLE_PATH,
    DATA_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_LABELS_UK,
    FEATURES,
    METRICS_PATH,
    MODELS_BUNDLE_PATH,
    PREDICTION_THRESHOLD,
    QUICK_TRAIN_MAX_ROWS,
    TARGET,
    TEST_SIZE,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    TUNE_TOP_N,
    TUNING_CV_FOLDS,
    TUNING_MAX_SAMPLES,
    TUNING_N_ITER,
    VAL_SIZE,
)
from exceptions import DataLoadError
from model_registry import (
    CATEGORICAL_FEATURES,
    DEFAULT_MODEL_KEY,
    MODEL_LABELS_UK,
    NUMERIC_FEATURES,
    TUNING_PARAM_GRIDS,
    build_pipeline as registry_build_pipeline,
    create_smote,
    get_classifiers,
    get_model_pipelines,
)
from scoring import compute_selection_score

try:
    import xgboost
except Exception:  # pragma: no cover - optional metadata only
    xgboost = None

__all_feature_types__ = (CATEGORICAL_FEATURES, NUMERIC_FEATURES)

MODEL_SHORTCUTS = {
    "rf": "random_forest",
    "xgb": "xgboost",
    "lr": "logistic_regression",
    "hgb": "hist_gradient_boosting",
    "dt": "decision_tree",
    "ada": "adaboost",
}


def load_data() -> pd.DataFrame:
    """
    Завантажує та очищує навчальний датасет.

    Returns:
        DataFrame без рядків із пропущеними значеннями в ознаках або цілі.

    Raises:
        DataLoadError: Якщо файл відсутній, порожній або пошкоджений.
    """
    if not DATA_PATH.exists():
        raise DataLoadError(f"Файл даних не знайдено: {DATA_PATH}")

    try:
        dataframe = pd.read_csv(DATA_PATH)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DataLoadError(f"Не вдалося прочитати CSV: {DATA_PATH}") from exc
    except UnicodeDecodeError as exc:
        raise DataLoadError(
            f"CSV має некоректне кодування: {DATA_PATH}"
        ) from exc
    except OSError as exc:
        raise DataLoadError(
            f"Помилка доступу до файлу даних: {DATA_PATH}"
        ) from exc

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise DataLoadError(
            "У CSV відсутні стовпці: "
            f"{', '.join(missing_columns)}."
        )

    cleaned = dataframe.dropna(subset=required_columns)
    if cleaned.empty:
        raise DataLoadError(
            "Після видалення пропусків датасет став порожнім."
        )

    return cleaned


def build_pipeline(model_key: str = DEFAULT_MODEL_KEY) -> Pipeline:
    """
    Повертає pipeline для одного алгоритму (зворотна сумісність із тестами).

    Args:
        model_key: Ключ алгоритму з model_registry.

    Returns:
        Ненавчений pipeline.
    """
    classifiers = get_classifiers()
    if model_key not in classifiers:
        raise ValueError(f"Невідомий алгоритм: {model_key}")

    return registry_build_pipeline(
        classifiers[model_key],
        use_smote=True,
        smote=create_smote(2),
    )


def _predict_probabilities(pipeline: Pipeline, x_data) -> np.ndarray:
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

    Args:
        pipeline: Навчений pipeline.
        x_test: Ознаки вибірки.
        y_test: Цільова змінна вибірки.
        threshold: Поріг для бінаризації ймовірностей.

    Returns:
        Словник із accuracy, error_rate, precision, recall, f1, roc_auc, pr_auc.
    """
    try:
        probabilities = _predict_probabilities(pipeline, x_test)
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

    Args:
        metrics_by_model: Метрики всіх алгоритмів.

    Returns:
        Ключ найкращої моделі.

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


def _get_tuning_sample(x_train, y_train):
    """Повертає підвибірку для швидшого RandomizedSearchCV."""
    if len(x_train) <= TUNING_MAX_SAMPLES:
        return x_train, y_train

    x_sample, _, y_sample, _ = train_test_split(
        x_train,
        y_train,
        train_size=TUNING_MAX_SAMPLES,
        random_state=42,
        stratify=y_train,
    )
    return x_sample, y_sample


def _subsample_dataframe(
    dataframe: pd.DataFrame,
    max_rows: int,
) -> pd.DataFrame:
    """За потреби зменшує датасет зі stratify по цілі."""
    effective_max_rows = max(max_rows, QUICK_TRAIN_MAX_ROWS)
    if effective_max_rows <= 0 or len(dataframe) <= effective_max_rows:
        return dataframe

    sampled, _ = train_test_split(
        dataframe,
        train_size=effective_max_rows,
        random_state=42,
        stratify=dataframe[TARGET],
    )
    return sampled.reset_index(drop=True)


def _split_dataset(features, target):
    """Робить 3-way split: train / validation / test."""
    try:
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            features,
            target,
            test_size=TEST_SIZE,
            random_state=42,
            stratify=target,
        )
        validation_ratio = VAL_SIZE / (1 - TEST_SIZE)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val,
            y_train_val,
            test_size=validation_ratio,
            random_state=42,
            stratify=y_train_val,
        )
    except ValueError as exc:
        raise DataLoadError(
            "Недостатньо даних для stratified train/val/test split."
        ) from exc

    return x_train, x_val, x_test, y_train, y_val, y_test


def tune_top_models(
    trained_models: dict[str, Pipeline],
    metrics_by_model: dict[str, dict],
    x_train,
    y_train,
    x_test,
    y_test,
    top_n: int = TUNE_TOP_N,
) -> dict[str, Pipeline]:
    """
    Тюнить гіперпараметри для топ-N моделей за selection_score.

    Args:
        trained_models: Навчені pipeline.
        metrics_by_model: Метрики моделей (оновлюються після тюнінгу).
        x_train: Train-ознаки.
        y_train: Train-ціль.
        x_test: Validation/test-ознаки для метрик після тюнінгу.
        y_test: Validation/test-ціль для метрик після тюнінгу.
        top_n: Скільки моделей тюнити.

    Returns:
        Оновлений словник навчених pipeline.
    """
    ranked_keys = sorted(
        (
            key
            for key in metrics_by_model
            if not key.startswith("_") and "selection_score" in metrics_by_model[key]
        ),
        key=lambda key: metrics_by_model[key]["selection_score"],
        reverse=True,
    )
    x_tune, y_tune = _get_tuning_sample(x_train, y_train)
    minority_count = int(y_tune.value_counts().min())
    smote = create_smote(minority_count)

    for model_key in ranked_keys[:top_n]:
        if model_key not in TUNING_PARAM_GRIDS:
            continue

        label = MODEL_LABELS_UK[model_key]
        print(f"\nТюнінг: {label}...")

        try:
            pipelines = get_model_pipelines(smote=smote, model_keys=[model_key])
            pipeline = pipelines.get(model_key)
            if pipeline is None:
                continue

            search = RandomizedSearchCV(
                pipeline,
                param_distributions=TUNING_PARAM_GRIDS[model_key],
                n_iter=TUNING_N_ITER,
                cv=TUNING_CV_FOLDS,
                scoring="roc_auc",
                random_state=42,
                n_jobs=1,
                verbose=0,
            )
            search.fit(x_tune, y_tune)
            best_pipeline = search.best_estimator_
            best_pipeline.fit(x_train, y_train)

            tuned_metrics = evaluate_model(best_pipeline, x_test, y_test)
            tuned_metrics["label_uk"] = label
            tuned_metrics["selection_score"] = compute_selection_score(
                tuned_metrics
            )
            tuned_metrics["tuned"] = True
            tuned_metrics["best_params"] = {
                key: value for key, value in search.best_params_.items()
            }

            trained_models[model_key] = best_pipeline
            metrics_by_model[model_key] = tuned_metrics

            print(f"  Найкращі параметри: {search.best_params_}")
            print(
                f"  ROC-AUC: {tuned_metrics['roc_auc']:.2%}, "
                f"PR-AUC: {tuned_metrics['pr_auc']:.2%}, "
                f"Recall: {tuned_metrics['recall']:.2%}"
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  Попередження: тюнінг «{label}» не вдався: {exc}")

    return trained_models


def extract_feature_importance(
    pipeline: Pipeline,
    top_n: int = 8,
) -> list[dict]:
    """
    Витягує важливість ознак із pipeline.

    Args:
        pipeline: Навчений pipeline із кроком classifier.
        top_n: Скільки ознак показати.

    Returns:
        Список словників feature, label_uk, importance.
    """
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

    return _normalize_importance_rows(raw_names, importances, top_n=top_n)


def _normalize_importance_rows(
    raw_names,
    importances,
    top_n: int = 8,
) -> list[dict]:
    """Агрегує важливості після one-hot назад до базових ознак."""
    mapped: dict[str, float] = {}

    for name, importance in zip(raw_names, importances, strict=False):
        base_feature = _map_transformed_name_to_feature(str(name))
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

    return _normalize_importance_rows(
        raw_names,
        result.importances_mean,
        top_n=top_n,
    )


def _map_transformed_name_to_feature(transformed_name: str) -> str:
    """Повертає базову назву ознаки з імені після препроцесора."""
    for feature in FEATURES:
        if transformed_name == feature:
            return feature
        if transformed_name.startswith(f"num__{feature}"):
            return feature
        if transformed_name.startswith(f"cat__{feature}_"):
            return feature
    return transformed_name


def find_optimal_threshold(
    pipeline: Pipeline,
    x_val,
    y_val,
) -> float:
    """
    Підбирає поріг за критерієм Youden J на validation.
    """
    probabilities = _predict_probabilities(pipeline, x_val)
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


def train_all_models(
    enable_tuning: bool = True,
    max_rows: int = 0,
    model_keys: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, Pipeline], dict[str, dict], str, list[dict], float]:
    """
    Навчає моделі, тюнить топові та обирає найкращу на validation.

    Args:
        enable_tuning: Чи виконувати RandomizedSearchCV для топ-моделей.
        max_rows: Максимальний розмір підвибірки для швидкого запуску.
        model_keys: Обмежити запуск конкретними моделями.

    Returns:
        Кортеж (моделі, тестові метрики, ключ найкращої моделі,
        важливість ознак, optimal_threshold).
    """
    dataframe = _subsample_dataframe(load_data(), max_rows=max_rows)
    features = dataframe[FEATURES]
    target = dataframe[TARGET]

    if target.nunique() < 2:
        raise DataLoadError(
            "Цільова змінна містить менше двох класів — "
            "навчання неможливе."
        )

    x_train, x_val, x_test, y_train, y_val, y_test = _split_dataset(
        features,
        target,
    )

    minority_count = int(y_train.value_counts().min())
    smote = create_smote(minority_count)

    pipelines = get_model_pipelines(smote=smote, model_keys=model_keys)
    if model_keys is not None:
        model_keys_set = set(model_keys)
        pipelines = {
            key: pipeline
            for key, pipeline in pipelines.items()
            if key in model_keys_set
        }

    if not pipelines:
        raise DataLoadError("Немає моделей для навчання після фільтрації.")

    trained_models: dict[str, Pipeline] = {}
    validation_metrics: dict[str, dict] = {}

    print("Порівняння алгоритмів (train/validation/test split):\n")
    print(
        f"{'Алгоритм':<35} {'ROC-AUC':>8} {'PR-AUC':>8} "
        f"{'Recall':>8} {'F1':>8}"
    )
    print("-" * 80)

    for model_key, pipeline in pipelines.items():
        label = MODEL_LABELS_UK.get(model_key, model_key)
        print(f"Навчання: {label}...")

        try:
            pipeline.fit(x_train, y_train)
        except Exception as exc:
            raise DataLoadError(
                f"Не вдалося навчити модель «{label}»: {exc}"
            ) from exc

        model_metrics = evaluate_model(pipeline, x_val, y_val)
        model_metrics["label_uk"] = label
        model_metrics["selection_score"] = compute_selection_score(
            model_metrics
        )
        model_metrics["tuned"] = False
        model_metrics["evaluated_on"] = "validation"

        trained_models[model_key] = pipeline
        validation_metrics[model_key] = model_metrics

        print(
            f"{label:<35} "
            f"{model_metrics['roc_auc']:>7.2%} "
            f"{model_metrics['pr_auc']:>7.2%} "
            f"{model_metrics['recall']:>7.2%} "
            f"{model_metrics['f1']:>7.2%}"
        )

    if enable_tuning:
        print(f"\n--- Тюнінг топ-{TUNE_TOP_N} моделей ---")
        trained_models = tune_top_models(
            trained_models,
            validation_metrics,
            x_train,
            y_train,
            x_val,
            y_val,
        )

    best_model_key = select_best_model_key(validation_metrics)
    best_pipeline = trained_models[best_model_key]
    optimal_threshold = find_optimal_threshold(best_pipeline, x_val, y_val)

    metrics_by_model: dict[str, dict] = {}
    for model_key, pipeline in trained_models.items():
        final_metrics = evaluate_model(pipeline, x_test, y_test)
        final_metrics["label_uk"] = MODEL_LABELS_UK.get(model_key, model_key)
        final_metrics["selection_score"] = compute_selection_score(
            final_metrics
        )
        final_metrics["selection_score_validation"] = validation_metrics[
            model_key
        ]["selection_score"]
        final_metrics["tuned"] = validation_metrics[model_key].get(
            "tuned",
            False,
        )
        if "best_params" in validation_metrics[model_key]:
            final_metrics["best_params"] = validation_metrics[model_key][
                "best_params"
            ]
        final_metrics["evaluated_on"] = "test"
        final_metrics["is_best"] = model_key == best_model_key
        metrics_by_model[model_key] = final_metrics

    metrics_by_model["_meta"] = {
        "selected_on": "validation",
        "reported_on": "test",
        "optimal_threshold": optimal_threshold,
        "default_threshold": PREDICTION_THRESHOLD,
        "threshold_search_min": THRESHOLD_MIN,
        "threshold_search_max": THRESHOLD_MAX,
        "sample_rows": int(len(dataframe)),
        "train_rows": int(len(x_train)),
        "validation_rows": int(len(x_val)),
        "test_rows": int(len(x_test)),
    }

    predictions = (
        _predict_probabilities(best_pipeline, x_test) >= optimal_threshold
    ).astype(int)
    print(
        f"\nНайкраща модель: {MODEL_LABELS_UK[best_model_key]} "
        f"(validation score: "
        f"{validation_metrics[best_model_key]['selection_score']})"
    )
    print(f"Оптимальний поріг на validation: {optimal_threshold:.2f}")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Ні", "Так"],
            zero_division=0,
        )
    )

    feature_importance = extract_feature_importance(best_pipeline)
    if not feature_importance:
        feature_importance = compute_permutation_importance(
            best_pipeline,
            x_val,
            y_val,
        )

    return (
        trained_models,
        metrics_by_model,
        best_model_key,
        feature_importance,
        optimal_threshold,
    )


def train_and_evaluate() -> Pipeline:
    """
    Навчає найкращу модель (зворотна сумісність із тестами).

    Returns:
        Навчений pipeline найкращої моделі.
    """
    models, _, best_key, _, _ = train_all_models(enable_tuning=False)
    return models[best_key]


def _compute_data_checksum() -> str | None:
    """Повертає SHA256 датасету, якщо файл доступний."""
    if not DATA_PATH.exists():
        return None

    digest = hashlib.sha256()
    try:
        with DATA_PATH.open("rb") as data_file:
            for chunk in iter(lambda: data_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None

    return digest.hexdigest()


def _build_bundle_metadata(optimal_threshold: float | None) -> dict[str, object]:
    """Формує метадані бандла."""
    metadata: dict[str, object] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": getattr(xgboost, "__version__", None),
        "data_checksum": _compute_data_checksum(),
        "optimal_threshold": optimal_threshold,
    }
    return metadata


def save_models_bundle(
    models: dict[str, Pipeline],
    metrics: dict[str, dict],
    best_model_key: str,
    feature_importance: list[dict],
    bundle_path: Path = MODELS_BUNDLE_PATH,
    also_save_best: bool = True,
    optimal_threshold: float | None = None,
) -> None:
    """
    Зберігає пакет моделей і, за потреби, окремий best-only бандл.
    """
    metadata = _build_bundle_metadata(optimal_threshold)
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
    """
    Зберігає метрики в JSON для швидкого читання веб-інтерфейсом.
    """
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
    """
    Зберігає важливість ознак у JSON.
    """
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
    """
    Зберігає одну модель у вигляді пакета (зворотна сумісність із тестами).
    """
    bundle = {
        "models": {DEFAULT_MODEL_KEY: pipeline},
        "metrics": {},
        "default_model": DEFAULT_MODEL_KEY,
        "best_model": DEFAULT_MODEL_KEY,
        "model_labels": MODEL_LABELS_UK,
        "feature_importance": [],
        "metadata": _build_bundle_metadata(None),
    }

    try:
        joblib.dump(bundle, model_path)
    except OSError as exc:
        raise OSError(
            f"Не вдалося зберегти модель: {model_path}"
        ) from exc

    print(f"Модель збережено: {model_path}")


def _parse_model_keys(raw_models: str | None) -> list[str] | None:
    """Мапить короткі CLI-аліаси моделей у внутрішні ключі."""
    if not raw_models:
        return None

    resolved_keys: list[str] = []
    for raw_key in raw_models.split(","):
        model_key = raw_key.strip()
        if not model_key:
            continue
        model_key = MODEL_SHORTCUTS.get(model_key, model_key)
        if model_key not in MODEL_LABELS_UK:
            raise ValueError(f"Невідомий ключ моделі: {raw_key.strip()}")
        if model_key not in resolved_keys:
            resolved_keys.append(model_key)

    return resolved_keys or None


def _build_arg_parser() -> argparse.ArgumentParser:
    """Створює CLI parser для скрипта навчання."""
    parser = argparse.ArgumentParser(
        description="Навчання моделей передбачення діабету."
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Вимкнути RandomizedSearchCV для топ-моделей.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Список моделей через кому: rf,xgb,lr,hgb,dt,ada.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Обмежити кількість рядків для швидкого навчання.",
    )
    parser.add_argument(
        "--serve-best-only",
        action="store_true",
        help="Зберегти лише best-only bundle.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Точка входу скрипта навчання.

    Returns:
        0 при успіху, 1 при помилці.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv if argv is not None else [])

    try:
        selected_model_keys = _parse_model_keys(args.models)
        (
            models,
            metrics,
            best_key,
            importance,
            optimal_threshold,
        ) = train_all_models(
            enable_tuning=not args.no_tune,
            max_rows=args.sample,
            model_keys=selected_model_keys,
        )

        if args.serve_best_only:
            save_models_bundle(
                models={best_key: models[best_key]},
                metrics={
                    best_key: metrics[best_key],
                    "_meta": metrics.get("_meta", {}),
                },
                best_model_key=best_key,
                feature_importance=importance,
                bundle_path=BEST_MODELS_BUNDLE_PATH,
                also_save_best=False,
                optimal_threshold=optimal_threshold,
            )
        else:
            save_models_bundle(
                models,
                metrics,
                best_key,
                importance,
                optimal_threshold=optimal_threshold,
            )

        save_metrics_json(metrics)
        save_feature_importance(importance)
    except (DataLoadError, ValueError, OSError) as exc:
        print(f"Помилка навчання: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
