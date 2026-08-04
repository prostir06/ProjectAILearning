"""
Реєстр алгоритмів машинного навчання для передбачення діабету.

Кожен алгоритм обгортається в pipeline зі спільним препроцесингом.
SMOTE застосовується лише там, де це явно потрібно.
"""

from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from diabetes.core.config import FEATURES

CATEGORICAL_FEATURES = ["gender", "smoking_history"]
NUMERIC_FEATURES = [
    feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES
]

MODEL_LABELS_UK = {
    "random_forest": "Випадковий ліс (Random Forest)",
    "logistic_regression": "Логістична регресія",
    "hist_gradient_boosting": "Градієнтний бустинг",
    "decision_tree": "Дерево рішень",
    "adaboost": "AdaBoost",
    "xgboost": "XGBoost",
}

DEFAULT_MODEL_KEY = "random_forest"

MODELS_USE_SMOTE = {
    "logistic_regression": True,
}

TUNING_PARAM_GRIDS = {
    "xgboost": {
        "classifier__max_depth": [4, 6, 8, 10],
        "classifier__learning_rate": [0.03, 0.05, 0.1],
        "classifier__n_estimators": [100, 200, 300],
        "classifier__subsample": [0.7, 0.85, 1.0],
    },
    "hist_gradient_boosting": {
        "classifier__max_depth": [6, 8, 10, 12],
        "classifier__learning_rate": [0.03, 0.05, 0.1],
        "classifier__max_iter": [100, 200, 300],
    },
    "random_forest": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [8, 10, 12, None],
    },
}


def build_preprocessor() -> ColumnTransformer:
    """
    Створює спільний препроцесор для всіх алгоритмів.

    Returns:
        ColumnTransformer із масштабуванням чисел і one-hot категорій.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def create_smote(minority_count: int) -> SMOTE:
    """
    Створює SMOTE з безпечним k_neighbors для малих вибірок.

    Args:
        minority_count: Кількість записів у меншинному класі на train.

    Returns:
        Налаштований об'єкт SMOTE.

    Raises:
        ValueError: Якщо minority_count < 1.
    """
    if minority_count < 1:
        raise ValueError(
            f"minority_count має бути >= 1, отримано {minority_count}."
        )

    k_neighbors = max(1, min(5, minority_count - 1))
    return SMOTE(random_state=42, k_neighbors=k_neighbors)


def get_classifiers() -> dict[str, ClassifierMixin]:
    """
    Повертає словник класифікаторів для порівняння.

    Returns:
        dict: ключ моделі → ненавчений sklearn/xgboost класифікатор.
    """
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=1,
            class_weight="balanced",
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        ),
        "adaboost": AdaBoostClassifier(
            n_estimators=50,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric="logloss",
            n_jobs=1,
            scale_pos_weight=1,
        ),
    }


def build_pipeline(
    classifier: ClassifierMixin,
    use_smote: bool = True,
    smote: SMOTE | None = None,
) -> Pipeline | ImbPipeline:
    """
    Збирає pipeline для конкретного класифікатора.

    Args:
        classifier: sklearn/xgboost класифікатор.
        use_smote: Чи додавати крок SMOTE після препроцесингу.
        smote: Готовий об'єкт SMOTE (опційно).

    Returns:
        Pipeline або ImbPipeline із preprocessor [, smote] і classifier.
    """
    steps = [("preprocessor", build_preprocessor())]
    if use_smote:
        steps.append(("smote", smote or SMOTE(random_state=42)))
    steps.append(("classifier", classifier))

    if use_smote:
        return ImbPipeline(steps=steps)

    return Pipeline(steps=steps)


def get_model_pipelines(
    smote: SMOTE | None = None,
    model_keys: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Pipeline | ImbPipeline]:
    """
    Повертає pipeline для вибраних зареєстрованих алгоритмів.

    Args:
        smote: Налаштований SMOTE для train-вибірки.
        model_keys: Необов'язковий список ключів моделей.

    Returns:
        dict: ключ моделі → Pipeline.
    """
    classifiers = get_classifiers()

    if model_keys is not None:
        classifiers = {
            key: classifier
            for key, classifier in classifiers.items()
            if key in set(model_keys)
        }

    return {
        key: build_pipeline(
            classifier,
            use_smote=MODELS_USE_SMOTE.get(key, False),
            smote=smote,
        )
        for key, classifier in classifiers.items()
    }
