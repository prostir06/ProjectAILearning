"""
Unit-тести для diabetes.ml.registry (алгоритми, SMOTE, scale_pos_weight).
"""

import math

import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier

from diabetes.ml.registry import (
    DEFAULT_MODEL_KEY,
    DEFAULT_SCALE_POS_WEIGHT,
    MODEL_LABELS_UK,
    build_pipeline,
    build_preprocessor,
    compute_scale_pos_weight,
    create_smote,
    get_classifiers,
    get_model_pipelines,
    normalize_scale_pos_weight,
)


def test_get_classifiers_contains_expected_algorithms():
    """Реєстр містить усі заявлені алгоритми."""
    classifiers = get_classifiers()

    assert DEFAULT_MODEL_KEY in classifiers
    assert len(classifiers) == len(MODEL_LABELS_UK)
    assert "logistic_regression" in classifiers
    assert "hist_gradient_boosting" in classifiers
    assert "xgboost" in classifiers


def test_build_preprocessor_has_numeric_and_categorical_steps():
    """Препроцесор містить кроки для числових і категоріальних ознак."""
    preprocessor = build_preprocessor()
    transformer_names = [name for name, _, _ in preprocessor.transformers]

    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_build_pipeline_structure():
    """Pipeline містить preprocessor, smote і classifier."""
    pipeline = build_pipeline(RandomForestClassifier(random_state=42))

    assert isinstance(pipeline, ImbPipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "smote" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_get_model_pipelines_matches_labels():
    """Кількість pipeline відповідає кількості підписів алгоритмів."""
    pipelines = get_model_pipelines()

    assert set(pipelines.keys()) == set(MODEL_LABELS_UK.keys())


def test_get_model_pipelines_uses_smote_only_for_logistic_regression():
    """SMOTE додається лише для logistic_regression."""
    pipelines = get_model_pipelines()

    assert "smote" in pipelines["logistic_regression"].named_steps
    assert "smote" not in pipelines["random_forest"].named_steps
    assert "smote" not in pipelines["decision_tree"].named_steps
    assert "smote" not in pipelines["hist_gradient_boosting"].named_steps
    assert "smote" not in pipelines["adaboost"].named_steps
    assert "smote" not in pipelines["xgboost"].named_steps


def test_create_smote_adapts_k_neighbors():
    """SMOTE зменшує k_neighbors для малих вибірок."""
    smote = create_smote(minority_count=3)

    assert smote.k_neighbors == 2


def test_create_smote_rejects_zero_minority():
    """create_smote відхиляє minority_count < 1."""
    with pytest.raises(ValueError, match="minority_count"):
        create_smote(minority_count=0)


def test_get_model_pipelines_filters_by_model_keys():
    """model_keys обмежує набір pipeline."""
    pipelines = get_model_pipelines(model_keys=["xgboost", "random_forest"])
    assert set(pipelines) == {"xgboost", "random_forest"}


def test_compute_scale_pos_weight_balanced_classes():
    """При рівній кількості класів scale_pos_weight = 1."""
    assert compute_scale_pos_weight([0, 1, 0, 1]) == 1.0


def test_compute_scale_pos_weight_imbalanced():
    """neg/pos ratio для imbalanced train."""
    assert compute_scale_pos_weight([0, 0, 0, 1]) == 3.0


def test_compute_scale_pos_weight_single_class():
    """Один клас → без зміни ваги."""
    assert compute_scale_pos_weight([1, 1, 1]) == 1.0


def test_get_model_pipelines_sets_xgboost_scale_pos_weight():
    """XGBoost отримує переданий scale_pos_weight."""
    pipelines = get_model_pipelines(
        model_keys=["xgboost"],
        scale_pos_weight=2.5,
    )
    classifier = pipelines["xgboost"].named_steps["classifier"]
    assert classifier.scale_pos_weight == 2.5


def test_normalize_scale_pos_weight_accepts_valid_ratio():
    """Додатне скінченне значення повертається без змін."""
    assert normalize_scale_pos_weight(3.25) == 3.25


@pytest.mark.parametrize(
    "invalid_value",
    [0, -1.0, float("nan"), float("inf"), "bad", None],
)
def test_normalize_scale_pos_weight_rejects_invalid(invalid_value):
    """Некоректні значення замінюються на DEFAULT_SCALE_POS_WEIGHT."""
    assert normalize_scale_pos_weight(invalid_value) == DEFAULT_SCALE_POS_WEIGHT


def test_compute_scale_pos_weight_none_returns_default():
    """None y → безпечний fallback."""
    assert compute_scale_pos_weight(None) == DEFAULT_SCALE_POS_WEIGHT


def test_compute_scale_pos_weight_empty_returns_default():
    """Порожній масив → fallback."""
    assert compute_scale_pos_weight([]) == DEFAULT_SCALE_POS_WEIGHT


def test_compute_scale_pos_weight_pandas_series():
    """pandas Series обробляється так само, як list."""
    series = pd.Series([0, 0, 1, 1, 1])
    assert compute_scale_pos_weight(series) == pytest.approx(2 / 3)


def test_compute_scale_pos_weight_ignores_nan_labels():
    """NaN-мітки відкидаються перед підрахунком класів."""
    assert compute_scale_pos_weight([0, 0, 1, math.nan]) == 2.0


def test_compute_scale_pos_weight_all_nan_returns_default():
    """Якщо всі мітки NaN — fallback."""
    assert compute_scale_pos_weight([math.nan, math.nan]) == DEFAULT_SCALE_POS_WEIGHT


def test_compute_scale_pos_weight_invalid_type_returns_default():
    """Неконвертований тип y не ламає навчання."""
    assert compute_scale_pos_weight({"bad": "labels"}) == DEFAULT_SCALE_POS_WEIGHT


def test_compute_scale_pos_weight_missing_positive_class():
    """Відсутність класу 1 → fallback."""
    assert compute_scale_pos_weight([0, 0, 0]) == DEFAULT_SCALE_POS_WEIGHT


def test_compute_scale_pos_weight_missing_negative_class():
    """Відсутність класу 0 → fallback."""
    assert compute_scale_pos_weight([1, 1, 1]) == DEFAULT_SCALE_POS_WEIGHT


def test_get_classifiers_normalizes_invalid_xgboost_weight():
    """get_classifiers захищає XGBoost від NaN/inf scale_pos_weight."""
    classifiers = get_classifiers(scale_pos_weight=float("nan"))
    assert classifiers["xgboost"].scale_pos_weight == DEFAULT_SCALE_POS_WEIGHT


def test_compute_scale_pos_weight_logs_on_none(caplog):
    """При y=None пишеться попередження в лог."""
    with caplog.at_level("WARNING"):
        compute_scale_pos_weight(None)
    assert "y=None" in caplog.text
