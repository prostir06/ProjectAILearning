"""
Гіперпараметричний тюнінг топ-моделей (RandomizedSearchCV).
"""

from __future__ import annotations

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from diabetes.core.config import (
    TUNE_TOP_N,
    TUNING_CV_FOLDS,
    TUNING_MAX_SAMPLES,
    TUNING_N_ITER,
)
from diabetes.core.scoring import compute_selection_score
from diabetes.ml.evaluate import evaluate_model
from diabetes.ml.registry import (
    MODEL_LABELS_UK,
    TUNING_PARAM_GRIDS,
    create_smote,
    get_model_pipelines,
    resolve_scale_pos_weight,
)


def get_tuning_sample(x_train, y_train):
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

    x_test / y_test тут — validation-вибірка під час навчання.
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
    x_tune, y_tune = get_tuning_sample(x_train, y_train)
    minority_count = int(y_tune.value_counts().min())
    smote = create_smote(minority_count)
    scale_pos_weight = resolve_scale_pos_weight(y_train)

    for model_key in ranked_keys[:top_n]:
        if model_key not in TUNING_PARAM_GRIDS:
            continue

        label = MODEL_LABELS_UK[model_key]
        print(f"\nТюнінг: {label}...")

        try:
            pipelines = get_model_pipelines(
                smote=smote,
                model_keys=[model_key],
                scale_pos_weight=scale_pos_weight,
            )
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
