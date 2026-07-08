"""
Спільні фікстури pytest для тестів проєкту.
"""

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import FEATURES, TARGET
import predict_diabetes as predict_module
import train_diabetes_model as train_module


@pytest.fixture
def sample_person():
    """Валідні дані одного пацієнта для передбачення."""
    return {
        "gender": "Female",
        "age": 54.0,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "never",
        "bmi": 27.3,
        "HbA1c_level": 6.6,
        "blood_glucose_level": 140,
    }


@pytest.fixture
def tiny_dataframe():
    """Мінімальний датасет із двома класами для навчання в тестах."""
    rows = []
    for index in range(10):
        has_diabetes = index % 2
        rows.append(
            {
                "gender": "Female" if index % 2 == 0 else "Male",
                "age": 30.0 + index,
                "hypertension": has_diabetes,
                "heart_disease": 0,
                "smoking_history": "never" if index % 2 == 0 else "current",
                "bmi": 22.0 + index,
                "HbA1c_level": 5.0 + has_diabetes,
                "blood_glucose_level": 90 + index * 10,
                "diabetes": has_diabetes,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def trained_pipeline(tiny_dataframe):
    """Невеликий навчений pipeline для тестів передбачення."""
    categorical = ["gender", "smoking_history"]
    numeric = [feature for feature in FEATURES if feature not in categorical]

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("num", StandardScaler(), numeric),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore"),
                            categorical,
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=10,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(
        tiny_dataframe[FEATURES],
        tiny_dataframe[TARGET],
    )
    return pipeline


@pytest.fixture(autouse=True)
def reset_model_cache():
    """Скидає кеш моделі перед і після кожного тесту."""
    predict_module.reset_pipeline_cache()
    yield
    predict_module.reset_pipeline_cache()
