"""
Unit-тести для validators.py.
"""

import pytest

from diabetes.core.exceptions import InvalidPatientDataError
from diabetes.core.validators import (
    _to_float,
    parse_prediction_threshold,
    validate_person_data,
)


def test_validate_person_data_success(sample_person):
    """Валідні дані повертаються з правильними типами."""
    result = validate_person_data(sample_person)

    assert result["gender"] == "Female"
    assert result["age"] == 54.0
    assert result["hypertension"] == 0
    assert result["blood_glucose_level"] == 140


def test_validate_person_data_from_form_strings():
    """Рядкові значення з HTML-форми коректно перетворюються."""
    form_data = {
        "gender": "Male",
        "age": "45",
        "hypertension": "1",
        "heart_disease": "0",
        "smoking_history": "current",
        "bmi": "27.5",
        "HbA1c_level": "6.1",
        "blood_glucose_level": "155",
    }

    result = validate_person_data(form_data)

    assert result["gender"] == "Male"
    assert result["hypertension"] == 1
    assert result["bmi"] == 27.5


def test_validate_person_data_defaults_smoking_to_no_info(sample_person):
    """Порожнє поле куріння замінюється на No Info."""
    sample_person["smoking_history"] = ""

    result = validate_person_data(sample_person)

    assert result["smoking_history"] == "No Info"


def test_validate_person_data_missing_field(sample_person):
    """Відсутнє поле викликає InvalidPatientDataError."""
    incomplete = sample_person.copy()
    del incomplete["bmi"]

    with pytest.raises(InvalidPatientDataError, match="Відсутні обов'язкові поля"):
        validate_person_data(incomplete)


def test_validate_person_data_invalid_gender(sample_person):
    """Невідома стать викликає InvalidPatientDataError."""
    sample_person["gender"] = "Unknown"

    with pytest.raises(InvalidPatientDataError, match="Невідома стать"):
        validate_person_data(sample_person)


def test_validate_person_data_invalid_smoking(sample_person):
    """Невідома історія куріння викликає InvalidPatientDataError."""
    sample_person["smoking_history"] = "sometimes"

    with pytest.raises(InvalidPatientDataError, match="Невідома історія куріння"):
        validate_person_data(sample_person)


def test_validate_person_data_out_of_range_bmi(sample_person):
    """ІМТ поза діапазоном викликає InvalidPatientDataError."""
    sample_person["bmi"] = 5.0

    with pytest.raises(InvalidPatientDataError, match="bmi"):
        validate_person_data(sample_person)


def test_validate_person_data_invalid_binary_field(sample_person):
    """Бінарне поле з некоректним значенням викликає помилку."""
    sample_person["hypertension"] = 2

    with pytest.raises(InvalidPatientDataError, match="hypertension"):
        validate_person_data(sample_person)


def test_validate_person_data_not_dict():
    """Недопустимий тип вхідних даних викликає помилку."""
    with pytest.raises(InvalidPatientDataError, match="словником"):
        validate_person_data(["Female", 45])


def test_validate_person_data_non_numeric_age(sample_person):
    """Нечислове значення віку викликає InvalidPatientDataError."""
    sample_person["age"] = "abc"

    with pytest.raises(InvalidPatientDataError, match="age"):
        validate_person_data(sample_person)


def test_validate_person_data_out_of_range_glucose(sample_person):
    """Глюкоза поза діапазоном викликає InvalidPatientDataError."""
    sample_person["blood_glucose_level"] = 999

    with pytest.raises(InvalidPatientDataError, match="blood_glucose_level"):
        validate_person_data(sample_person)


def test_validate_person_data_invalid_binary_string(sample_person):
    """Нечислове бінарне поле викликає InvalidPatientDataError."""
    sample_person["heart_disease"] = "maybe"

    with pytest.raises(InvalidPatientDataError, match="heart_disease"):
        validate_person_data(sample_person)


def test_parse_prediction_threshold_success():
    """parse_prediction_threshold перетворює відсотки у частку."""
    from diabetes.core.validators import parse_prediction_threshold

    assert parse_prediction_threshold(50) == 0.5
    assert parse_prediction_threshold("30") == 0.3


def test_parse_prediction_threshold_out_of_range():
    """parse_prediction_threshold відхиляє значення поза діапазоном."""
    from diabetes.core.validators import parse_prediction_threshold

    with pytest.raises(InvalidPatientDataError, match="Поріг"):
        parse_prediction_threshold(5)


def test_parse_prediction_threshold_empty_uses_default():
    """Порожній / None поріг повертає default."""
    from diabetes.core.validators import parse_prediction_threshold

    assert parse_prediction_threshold(None, default=0.4) == 0.4
    assert parse_prediction_threshold("", default=0.35) == 0.35
    assert parse_prediction_threshold("  ", default=0.5) == 0.5


def test_to_float_accepts_int_float_and_numeric_string():
    """_to_float нормалізує типові значення форми."""
    assert _to_float(54) == 54.0
    assert _to_float(27.5) == 27.5
    assert _to_float("  6.1  ") == 6.1


@pytest.mark.parametrize("bad", [True, False, None, [], {}, object()])
def test_to_float_rejects_unsupported_types(bad):
    """bool/None/контейнери не маскуються під числа."""
    with pytest.raises(TypeError):
        _to_float(bad)


@pytest.mark.parametrize("bad", ["", "abc", "nan", "inf", "-inf"])
def test_to_float_rejects_non_finite_or_empty(bad):
    """Порожній рядок, NaN і inf відхиляються."""
    with pytest.raises(ValueError):
        _to_float(bad)


def test_validate_person_data_rejects_bool_as_binary(sample_person):
    """True/False не є валідними 0/1 для hypertension."""
    sample_person["hypertension"] = True
    with pytest.raises(InvalidPatientDataError, match="hypertension"):
        validate_person_data(sample_person)


def test_validate_person_data_rejects_nan_age(sample_person):
    """NaN у віці → InvalidPatientDataError."""
    sample_person["age"] = float("nan")
    with pytest.raises(InvalidPatientDataError, match="age"):
        validate_person_data(sample_person)


def test_parse_prediction_threshold_rejects_non_finite():
    """inf у порозі не проходить."""
    with pytest.raises(InvalidPatientDataError, match="числом"):
        parse_prediction_threshold(float("inf"))


def test_parse_prediction_threshold_bad_default_raises():
    """Битий default при порожньому value теж ловиться."""
    with pytest.raises(InvalidPatientDataError, match="за замовчуванням"):
        parse_prediction_threshold(None, default="bad")  # type: ignore[arg-type]
