"""
Unit-тести для validators.py.
"""

import pytest

from exceptions import InvalidPatientDataError
from validators import validate_person_data


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
    from validators import parse_prediction_threshold

    assert parse_prediction_threshold(50) == 0.5
    assert parse_prediction_threshold("30") == 0.3


def test_parse_prediction_threshold_out_of_range():
    """parse_prediction_threshold відхиляє значення поза діапазоном."""
    from validators import parse_prediction_threshold

    with pytest.raises(InvalidPatientDataError, match="Поріг"):
        parse_prediction_threshold(5)
