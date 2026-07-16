"""
Unit-тести для config.py — інваріанти констант і діапазонів.
"""

from config import (
    BEST_MODEL_WEIGHTS,
    FEATURES,
    GENDERS,
    PREDICTION_THRESHOLD,
    SMOKING_HISTORY_VALUES,
    TARGET,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    VALID_RANGES,
)


def test_features_include_expected_columns():
    """FEATURES містить усі основні ознаки пацієнта."""
    assert "gender" in FEATURES
    assert "age" in FEATURES
    assert "bmi" in FEATURES
    assert "HbA1c_level" in FEATURES
    assert "blood_glucose_level" in FEATURES
    assert TARGET == "diabetes"
    assert TARGET not in FEATURES


def test_best_model_weights_sum_to_one():
    """Ваги рейтингу моделей сумуються до 1.0."""
    total = sum(BEST_MODEL_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9
    assert set(BEST_MODEL_WEIGHTS) == {"roc_auc", "recall", "f1"}


def test_threshold_bounds_are_valid():
    """Поріг за замовчуванням лежить у дозволеному діапазоні."""
    assert 0.0 < THRESHOLD_MIN < THRESHOLD_MAX < 1.0
    assert THRESHOLD_MIN <= PREDICTION_THRESHOLD <= THRESHOLD_MAX


def test_valid_ranges_cover_numeric_features():
    """VALID_RANGES має min < max для кожного числового поля."""
    for name, (low, high) in VALID_RANGES.items():
        assert low < high, name
        assert name in FEATURES


def test_gender_and_smoking_options_non_empty():
    """Довідники статі та куріння непорожні."""
    assert "Female" in GENDERS
    assert "Male" in GENDERS
    assert "No Info" in SMOKING_HISTORY_VALUES
