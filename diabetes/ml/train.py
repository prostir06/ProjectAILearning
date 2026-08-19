"""
Оркестрація навчання: split → fit → tune → test metrics.

Реалізація рознесена по ``data``, ``evaluate``, ``tune``, ``persist``, ``cli``.
Цей модуль збирає пайплайн і реекспортує публічний API для тестів і CLI.
"""

from __future__ import annotations

import sys

from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from diabetes.core.config import (
    FEATURES,
    PREDICTION_THRESHOLD,
    TARGET,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    TUNE_TOP_N,
)
from diabetes.core.exceptions import DataLoadError
from diabetes.core.scoring import compute_selection_score
from diabetes.ml.data import load_data, split_dataset, subsample_dataframe
from diabetes.ml.evaluate import (
    compute_permutation_importance,
    evaluate_model,
    extract_feature_importance,
    find_optimal_threshold,
    predict_probabilities,
    select_best_model_key,
)
from diabetes.ml.persist import (
    save_feature_importance,
    save_metrics_json,
    save_model,
    save_models_bundle,
)
from diabetes.ml.registry import (
    DEFAULT_MODEL_KEY,
    MODEL_LABELS_UK,
    MODELS_USE_SMOTE,
    compute_scale_pos_weight,
    create_smote,
    get_classifiers,
    get_model_pipelines,
    resolve_scale_pos_weight,
)
from diabetes.ml.registry import (
    build_pipeline as registry_build_pipeline,
)
from diabetes.ml.tune import tune_top_models

# Зворотна сумісність: тести імпортують звідси.
_resolve_scale_pos_weight = resolve_scale_pos_weight

__all__ = [
    "build_pipeline",
    "compute_scale_pos_weight",
    "evaluate_model",
    "extract_feature_importance",
    "load_data",
    "main",
    "save_feature_importance",
    "save_metrics_json",
    "save_model",
    "save_models_bundle",
    "select_best_model_key",
    "train_all_models",
    "train_and_evaluate",
]


def build_pipeline(model_key: str = DEFAULT_MODEL_KEY) -> Pipeline:
    """
    Повертає pipeline для одного алгоритму (зворотна сумісність із тестами).
    """
    classifiers = get_classifiers()
    if model_key not in classifiers:
        raise ValueError(f"Невідомий алгоритм: {model_key}")

    use_smote = MODELS_USE_SMOTE.get(model_key, False)
    return registry_build_pipeline(
        classifiers[model_key],
        use_smote=use_smote,
        smote=create_smote(2) if use_smote else None,
    )


def train_all_models(
    enable_tuning: bool = True,
    max_rows: int = 0,
    model_keys: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, Pipeline], dict[str, dict], str, list[dict], float]:
    """
    Навчає моделі, тюнить топові та обирає найкращу на validation.

    Returns:
        Кортеж (моделі, тестові метрики, ключ найкращої моделі,
        важливість ознак, optimal_threshold).
    """
    dataframe = subsample_dataframe(load_data(), max_rows=max_rows)
    features = dataframe[FEATURES]
    target = dataframe[TARGET]

    if target.nunique() < 2:
        raise DataLoadError(
            "Цільова змінна містить менше двох класів — "
            "навчання неможливе."
        )

    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        features,
        target,
    )

    minority_count = int(y_train.value_counts().min())
    smote = create_smote(minority_count)
    scale_pos_weight = resolve_scale_pos_weight(y_train)

    pipelines = get_model_pipelines(
        smote=smote,
        model_keys=model_keys,
        scale_pos_weight=scale_pos_weight,
    )
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
        predict_probabilities(best_pipeline, x_test) >= optimal_threshold
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
    """Навчає найкращу модель (зворотна сумісність із тестами)."""
    models, _, best_key, _, _ = train_all_models(enable_tuning=False)
    return models[best_key]


def main(argv: list[str] | None = None) -> int:
    """Делегує в ``diabetes.ml.cli.main`` (цикл імпорту уникається тут)."""
    from diabetes.ml.cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
