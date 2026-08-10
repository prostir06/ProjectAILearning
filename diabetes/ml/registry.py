"""
Реєстр алгоритмів машинного навчання для передбачення діабету.

Кожен алгоритм обгортається в pipeline зі спільним препроцесингом.
SMOTE застосовується лише там, де це явно потрібно.
"""

from __future__ import annotations

import logging

import numpy as np
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

logger = logging.getLogger(__name__)

# Безпечне значення scale_pos_weight, якщо train-вибірка не дозволяє обчислити ratio.
DEFAULT_SCALE_POS_WEIGHT = 1.0

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


def normalize_scale_pos_weight(value: float) -> float:
    """
    Нормалізує ``scale_pos_weight`` перед передачею в XGBoost.

    XGBoost очікує додатне скінченне число. Некоректні значення
    (NaN, inf, <= 0) замінюються на ``DEFAULT_SCALE_POS_WEIGHT``.

    Args:
        value: Запропонований коефіцієнт балансування класів.

    Returns:
        Безпечне додатне float для ``XGBClassifier(scale_pos_weight=...)``.
    """
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "scale_pos_weight=%r не є числом (%s) — використовуємо %.1f",
            value,
            exc,
            DEFAULT_SCALE_POS_WEIGHT,
        )
        return DEFAULT_SCALE_POS_WEIGHT

    if not np.isfinite(weight) or weight <= 0:
        logger.warning(
            "scale_pos_weight=%s поза допустимим діапазоном — використовуємо %.1f",
            value,
            DEFAULT_SCALE_POS_WEIGHT,
        )
        return DEFAULT_SCALE_POS_WEIGHT

    return weight


def compute_scale_pos_weight(y) -> float:
    """
    Обчислює ``scale_pos_weight`` для XGBoost із train-міток ``y``.

    Формула (документація XGBoost): ``neg_count / pos_count``, де
    позитивний клас — ``1`` (наявність діабету), негативний — ``0``.

    Функція **ніколи не піднімає виняток** назовні: при порожній вибірці,
    одному класі, NaN-мітках або некоректному типі ``y`` повертає
    ``DEFAULT_SCALE_POS_WEIGHT`` і пише попередження в лог.

    Args:
        y: Мітки класів (list, numpy array, pandas Series тощо).

    Returns:
        Коефіцієнт для ``XGBClassifier(scale_pos_weight=...)``.
    """
    if y is None:
        logger.warning(
            "compute_scale_pos_weight: y=None — fallback %.1f",
            DEFAULT_SCALE_POS_WEIGHT,
        )
        return DEFAULT_SCALE_POS_WEIGHT

    try:
        # dtype=float дозволяє однаково обробити int/bool/str-числа.
        values = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "compute_scale_pos_weight: не вдалося перетворити y (%s) — fallback %.1f",
            exc,
            DEFAULT_SCALE_POS_WEIGHT,
        )
        return DEFAULT_SCALE_POS_WEIGHT

    if values.size == 0:
        logger.warning(
            "compute_scale_pos_weight: порожня вибірка — fallback %.1f",
            DEFAULT_SCALE_POS_WEIGHT,
        )
        return DEFAULT_SCALE_POS_WEIGHT

    # NaN не є валідною міткою класу — відкидаємо перед підрахунком.
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        logger.warning(
            "compute_scale_pos_weight: усі мітки NaN/inf — fallback %.1f",
            DEFAULT_SCALE_POS_WEIGHT,
        )
        return DEFAULT_SCALE_POS_WEIGHT

    values = values[finite_mask]

    try:
        unique, counts = np.unique(values, return_counts=True)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "compute_scale_pos_weight: np.unique не вдався (%s) — fallback %.1f",
            exc,
            DEFAULT_SCALE_POS_WEIGHT,
        )
        return DEFAULT_SCALE_POS_WEIGHT

    if len(unique) < 2:
        # Один клас — ratio не визначений; XGBoost працює з дефолтною вагою.
        return DEFAULT_SCALE_POS_WEIGHT

    counts_by_class = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    # Підтримуємо як 0/1 (int), так і 0.0/1.0 після astype(float).
    pos_count = counts_by_class.get(1.0, counts_by_class.get(1, 0))
    neg_count = counts_by_class.get(0.0, counts_by_class.get(0, 0))

    if pos_count < 1 or neg_count < 1:
        logger.warning(
            "compute_scale_pos_weight: neg=%s pos=%s — fallback %.1f",
            neg_count,
            pos_count,
            DEFAULT_SCALE_POS_WEIGHT,
        )
        return DEFAULT_SCALE_POS_WEIGHT

    return normalize_scale_pos_weight(float(neg_count) / float(pos_count))


def get_classifiers(*, scale_pos_weight: float = DEFAULT_SCALE_POS_WEIGHT) -> dict[str, ClassifierMixin]:
    """
    Повертає словник класифікаторів для порівняння.

    Args:
        scale_pos_weight: Вага позитивного класу для XGBoost
            (``neg_count / pos_count`` з train; нормалізується всередині).

    Returns:
        dict: ключ моделі → ненавчений sklearn/xgboost класифікатор.
    """
    # Інші алгоритми використовують class_weight='balanced'; лише XGBoost — ratio.
    xgb_weight = normalize_scale_pos_weight(scale_pos_weight)

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
            # Компенсація дисбалансу класів (аналог class_weight='balanced').
            scale_pos_weight=xgb_weight,
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
    *,
    scale_pos_weight: float = DEFAULT_SCALE_POS_WEIGHT,
) -> dict[str, Pipeline | ImbPipeline]:
    """
    Повертає pipeline для вибраних зареєстрованих алгоритмів.

    Args:
        smote: Налаштований SMOTE для train-вибірки.
        model_keys: Необов'язковий список ключів моделей.
        scale_pos_weight: Коефіцієнт для XGBoost (див. ``compute_scale_pos_weight``).

    Returns:
        dict: ключ моделі → Pipeline.
    """
    classifiers = get_classifiers(scale_pos_weight=scale_pos_weight)

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
