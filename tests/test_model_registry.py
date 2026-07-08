"""
Unit-тести для model_registry.py.
"""

import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier

from model_registry import (
    DEFAULT_MODEL_KEY,
    MODEL_LABELS_UK,
    build_pipeline,
    build_preprocessor,
    create_smote,
    get_classifiers,
    get_model_pipelines,
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


def test_create_smote_adapts_k_neighbors():
    """SMOTE зменшує k_neighbors для малих вибірок."""
    smote = create_smote(minority_count=3)

    assert smote.k_neighbors == 2


def test_create_smote_rejects_zero_minority():
    """create_smote відхиляє minority_count < 1."""
    with pytest.raises(ValueError, match="minority_count"):
        create_smote(minority_count=0)
