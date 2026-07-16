"""
Навчання кількох моделей ML з SMOTE, порівнянням та вибором найкращої.

Скрипт:
1. Завантажує CSV-дані.
2. Навчає алгоритми з SMOTE на train-вибірці.
3. Обчислює метрики (ROC-AUC, precision, recall, F1).
4. Тюнить топ-2 моделі через RandomizedSearchCV.
5. Обирає найкращу модель і зберігає feature importance.
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from config import (
    BEST_MODEL_WEIGHTS,
    DATA_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_LABELS_UK,
    FEATURES,
    METRICS_PATH,
    MODELS_BUNDLE_PATH,
    TARGET,
    TUNE_TOP_N,
    TUNING_CV_FOLDS,
    TUNING_MAX_SAMPLES,
    TUNING_N_ITER,
)
from exceptions import DataLoadError
from model_registry import (
    DEFAULT_MODEL_KEY,
    MODEL_LABELS_UK,
    TUNING_PARAM_GRIDS,
    create_smote,
    get_model_pipelines,
)

# Категоріальні ознаки — для сумісності з тестами.
CATEGORICAL_FEATURES = ["gender", "smoking_history"]
NUMERIC_FEATURES = [
    feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES
]


def load_data() -> pd.DataFrame:
    """
    Завантажує та очищує навчальний датасет.

    Returns:
        DataFrame без рядків із пропущеними значеннями в ознаках або цілі.

    Raises:
        DataLoadError: Якщо файл відсутній, порожній або пошкоджений.
    """
    if not DATA_PATH.exists():
        raise DataLoadError(
            f"Файл даних не знайдено: {DATA_PATH}"
        )

    try:
        dataframe = pd.read_csv(DATA_PATH)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DataLoadError(
            f"Не вдалося прочитати CSV: {DATA_PATH}"
        ) from exc
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
    pipelines = get_model_pipelines()
    if model_key not in pipelines:
        raise ValueError(f"Невідомий алгоритм: {model_key}")
    return pipelines[model_key]


def evaluate_model(pipeline: Pipeline, x_test, y_test) -> dict:
    """
    Обчислює метрики якості для однієї моделі на тестовій вибірці.

    Args:
        pipeline: Навчений pipeline.
        x_test: Ознаки тестової вибірки.
        y_test: Цільова змінна тестової вибірки.

    Returns:
        Словник із accuracy, error_rate, precision, recall, f1, roc_auc.
    """
    try:
        predictions = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        accuracy = accuracy_score(y_test, predictions)
    except (AttributeError, IndexError, ValueError) as exc:
        raise DataLoadError(
            "Модель повернула некоректний результат під час оцінки."
        ) from exc

    # ROC-AUC потребує обох класів у тестовій вибірці.
    try:
        roc_auc = round(roc_auc_score(y_test, probabilities), 4)
    except ValueError:
        roc_auc = 0.0

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
    }


def compute_selection_score(metrics: dict) -> float:
    """
    Рахує зважений бал для вибору найкращої моделі.

    Формула: ROC-AUC×0.5 + Recall×0.3 + F1×0.2 (див. BEST_MODEL_WEIGHTS).

    Args:
        metrics: Метрики однієї моделі (roc_auc, recall, f1).

    Returns:
        Композитний бал (вище — краще). При відсутніх ключах — 0.0.
    """
    try:
        return round(
            BEST_MODEL_WEIGHTS["roc_auc"] * float(metrics["roc_auc"])
            + BEST_MODEL_WEIGHTS["recall"] * float(metrics["recall"])
            + BEST_MODEL_WEIGHTS["f1"] * float(metrics["f1"]),
            4,
        )
    except (KeyError, TypeError, ValueError):
        # Неповні або нечислові метрики не повинні зупиняти навчання інших моделей.
        return 0.0


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
    if not metrics_by_model:
        raise DataLoadError(
            "Немає навчених моделей для вибору найкращої."
        )

    try:
        return max(
            metrics_by_model,
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
        x_test: Test-ознаки.
        y_test: Test-ціль.
        top_n: Скільки моделей тюнити.

    Returns:
        Оновлений словник навчених pipeline.
    """
    ranked_keys = sorted(
        metrics_by_model,
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
            pipeline = get_model_pipelines(smote=smote)[model_key]
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=TUNING_PARAM_GRIDS[model_key],
                n_iter=TUNING_N_ITER,
                cv=TUNING_CV_FOLDS,
                scoring="roc_auc",
                random_state=42,
                n_jobs=-1,
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

            trained_models[model_key] = best_pipeline
            metrics_by_model[model_key] = tuned_metrics

            print(
                f"  Найкращі параметри: {search.best_params_}"
            )
            print(
                f"  ROC-AUC: {tuned_metrics['roc_auc']:.2%}, "
                f"Recall: {tuned_metrics['recall']:.2%}, "
                f"F1: {tuned_metrics['f1']:.2%}"
            )
        except Exception as exc:
            print(f"  Попередження: тюнінг «{label}» не вдався: {exc}")

    return trained_models


def extract_feature_importance(
    pipeline: Pipeline,
    top_n: int = 8,
) -> list[dict]:
    """
    Витягує важливість ознак із tree-based pipeline.

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

    mapped = {}

    for name, importance in zip(raw_names, importances, strict=False):
        base_feature = _map_transformed_name_to_feature(name)
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


def train_all_models(
    enable_tuning: bool = True,
) -> tuple[dict[str, Pipeline], dict[str, dict], str, list[dict]]:
    """
    Навчає всі алгоритми, тюнить топ-моделі та обирає найкращу.

    Args:
        enable_tuning: Чи виконувати RandomizedSearchCV для топ-моделей.

    Returns:
        Кортеж (моделі, метрики, ключ найкращої моделі, важливість ознак).

    Raises:
        DataLoadError: Якщо дані некоректні або недостатні для навчання.
    """
    dataframe = load_data()
    features = dataframe[FEATURES]
    target = dataframe[TARGET]

    if target.nunique() < 2:
        raise DataLoadError(
            "Цільова змінна містить менше двох класів — "
            "навчання неможливе."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    minority_count = int(y_train.value_counts().min())
    smote = create_smote(minority_count)

    trained_models: dict[str, Pipeline] = {}
    metrics_by_model: dict[str, dict] = {}

    print("Порівняння алгоритмів (SMOTE на train, тест 20%):\n")
    print(
        f"{'Алгоритм':<35} {'ROC-AUC':>8} {'Recall':>8} "
        f"{'F1':>8} {'Точність':>10}"
    )
    print("-" * 80)

    for model_key, pipeline in get_model_pipelines(smote=smote).items():
        label = MODEL_LABELS_UK[model_key]
        print(f"Навчання: {label}...")

        try:
            pipeline.fit(x_train, y_train)
        except Exception as exc:
            raise DataLoadError(
                f"Не вдалося навчити модель «{label}»: {exc}"
            ) from exc

        model_metrics = evaluate_model(pipeline, x_test, y_test)
        model_metrics["label_uk"] = label
        model_metrics["selection_score"] = compute_selection_score(
            model_metrics
        )
        model_metrics["tuned"] = False

        trained_models[model_key] = pipeline
        metrics_by_model[model_key] = model_metrics

        print(
            f"{label:<35} "
            f"{model_metrics['roc_auc']:>7.2%} "
            f"{model_metrics['recall']:>7.2%} "
            f"{model_metrics['f1']:>7.2%} "
            f"{model_metrics['accuracy']:>9.2%}"
        )

    if enable_tuning:
        print(f"\n--- Тюнінг топ-{TUNE_TOP_N} моделей ---")
        trained_models = tune_top_models(
            trained_models,
            metrics_by_model,
            x_train,
            y_train,
            x_test,
            y_test,
        )

    best_model_key = select_best_model_key(metrics_by_model)
    for key, model_metrics in metrics_by_model.items():
        model_metrics["is_best"] = key == best_model_key

    best_pipeline = trained_models[best_model_key]
    predictions = best_pipeline.predict(x_test)
    print(
        f"\nНайкраща модель: {MODEL_LABELS_UK[best_model_key]} "
        f"(бал: {metrics_by_model[best_model_key]['selection_score']})"
    )
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Ні", "Так"],
        )
    )

    feature_importance = extract_feature_importance(best_pipeline)

    return trained_models, metrics_by_model, best_model_key, feature_importance


def train_and_evaluate() -> Pipeline:
    """
    Навчає найкращу модель (зворотна сумісність із тестами).

    Returns:
        Навчений pipeline найкращої моделі.
    """
    models, _, best_key, _ = train_all_models(enable_tuning=False)
    return models[best_key]


def save_models_bundle(
    models: dict[str, Pipeline],
    metrics: dict[str, dict],
    best_model_key: str,
    feature_importance: list[dict],
    bundle_path: Path = MODELS_BUNDLE_PATH,
) -> None:
    """
    Зберігає всі моделі, метрики та найкращу модель у joblib-файл.

    Args:
        models: Словник навчених pipeline.
        metrics: Метрики кожного алгоритму.
        best_model_key: Ключ найкращої моделі.
        feature_importance: Важливість ознак найкращої моделі.
        bundle_path: Шлях для збереження.

    Raises:
        OSError: Якщо запис на диск не вдався.
    """
    bundle = {
        "models": models,
        "metrics": metrics,
        "default_model": best_model_key,
        "best_model": best_model_key,
        "model_labels": MODEL_LABELS_UK,
        "feature_importance": feature_importance,
    }

    try:
        joblib.dump(bundle, bundle_path)
    except OSError as exc:
        raise OSError(
            f"Не вдалося зберегти моделі: {bundle_path}"
        ) from exc

    print(f"\nУсі моделі збережено: {bundle_path}")
    print(f"Найкраща модель: {MODEL_LABELS_UK[best_model_key]}")


def save_metrics_json(
    metrics: dict[str, dict],
    metrics_path: Path = METRICS_PATH,
) -> None:
    """
    Зберігає метрики в JSON для швидкого читання веб-інтерфейсом.

    Args:
        metrics: Метрики кожного алгоритму.
        metrics_path: Шлях до JSON-файлу.

    Raises:
        OSError: Якщо запис на диск не вдався.
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

    Args:
        feature_importance: Список ознак із вагами.
        importance_path: Шлях до JSON-файлу.

    Raises:
        OSError: Якщо запис на диск не вдався.
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

    Args:
        pipeline: Навчений pipeline.
        model_path: Шлях для збереження.

    Raises:
        OSError: Якщо запис на диск не вдався.
    """
    bundle = {
        "models": {DEFAULT_MODEL_KEY: pipeline},
        "metrics": {},
        "default_model": DEFAULT_MODEL_KEY,
        "best_model": DEFAULT_MODEL_KEY,
        "model_labels": MODEL_LABELS_UK,
        "feature_importance": [],
    }

    try:
        joblib.dump(bundle, model_path)
    except OSError as exc:
        raise OSError(
            f"Не вдалося зберегти модель: {model_path}"
        ) from exc

    print(f"Модель збережено: {model_path}")


def main() -> int:
    """
    Точка входу скрипта навчання.

    Returns:
        0 при успіху, 1 при помилці.
    """
    try:
        models, metrics, best_key, importance = train_all_models()
        save_models_bundle(models, metrics, best_key, importance)
        save_metrics_json(metrics)
        save_feature_importance(importance)
    except (DataLoadError, ValueError, OSError) as exc:
        print(f"Помилка навчання: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
