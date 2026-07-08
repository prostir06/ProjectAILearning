"""
Валідація та нормалізація даних пацієнта перед передачею в модель.
"""

from config import (
    FEATURES,
    GENDERS,
    SMOKING_HISTORY_VALUES,
    VALID_RANGES,
)
from exceptions import InvalidPatientDataError


def _parse_binary_field(name: str, value) -> int:
    """
    Перетворює бінарне поле (0/1) у ціле число.

    Args:
        name: Назва поля (для повідомлення про помилку).
        value: Значення з форми або словника.

    Returns:
        0 або 1.

    Raises:
        InvalidPatientDataError: Якщо значення не 0 і не 1.
    """
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise InvalidPatientDataError(
            f"Поле «{name}» має бути 0 або 1."
        ) from exc

    if parsed not in (0, 1):
        raise InvalidPatientDataError(
            f"Поле «{name}» має бути 0 або 1."
        )

    return parsed


def _parse_float_field(name: str, value, min_val: float, max_val: float) -> float:
    """
    Перетворює числове поле у float і перевіряє діапазон.

    Args:
        name: Назва поля.
        value: Вхідне значення.
        min_val: Мінімально допустиме значення.
        max_val: Максимально допустиме значення.

    Returns:
        Валідне число з плаваючою крапкою.

    Raises:
        InvalidPatientDataError: Якщо значення не число або поза діапазоном.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidPatientDataError(
            f"Поле «{name}» має бути числом."
        ) from exc

    if not min_val <= parsed <= max_val:
        raise InvalidPatientDataError(
            f"Поле «{name}» має бути в діапазоні "
            f"{min_val}–{max_val}."
        )

    return parsed


def _parse_int_field(name: str, value, min_val: int, max_val: int) -> int:
    """
    Перетворює числове поле у int і перевіряє діапазон.

    Args:
        name: Назва поля.
        value: Вхідне значення.
        min_val: Мінімально допустиме значення.
        max_val: Максимально допустиме значення.

    Returns:
        Валідне ціле число.

    Raises:
        InvalidPatientDataError: Якщо значення не число або поза діапазоном.
    """
    try:
        parsed = int(float(value))
    except (TypeError, ValueError) as exc:
        raise InvalidPatientDataError(
            f"Поле «{name}» має бути цілим числом."
        ) from exc

    if not min_val <= parsed <= max_val:
        raise InvalidPatientDataError(
            f"Поле «{name}» має бути в діапазоні "
            f"{min_val}–{max_val}."
        )

    return parsed


def validate_person_data(data: dict) -> dict:
    """
    Перевіряє та нормалізує словник з даними пацієнта.

    Args:
        data: Словник із полями форми або API-запиту.

    Returns:
        Словник із типізованими значеннями, готовий для pandas DataFrame.

    Raises:
        InvalidPatientDataError: Якщо відсутні поля або значення некоректні.
    """
    if not isinstance(data, dict):
        raise InvalidPatientDataError("Дані пацієнта мають бути словником.")

    missing = [field for field in FEATURES if field not in data]
    if missing:
        raise InvalidPatientDataError(
            f"Відсутні обов'язкові поля: {', '.join(missing)}."
        )

    gender = str(data["gender"]).strip()
    if gender not in GENDERS:
        raise InvalidPatientDataError(
            f"Невідома стать: {gender!r}. Допустимо: {', '.join(GENDERS)}."
        )

    smoking_history = str(data["smoking_history"]).strip() or "No Info"
    if smoking_history not in SMOKING_HISTORY_VALUES:
        allowed = ", ".join(SMOKING_HISTORY_VALUES)
        raise InvalidPatientDataError(
            f"Невідома історія куріння: {smoking_history!r}. "
            f"Допустимо: {allowed}."
        )

    age_min, age_max = VALID_RANGES["age"]
    bmi_min, bmi_max = VALID_RANGES["bmi"]
    hba1c_min, hba1c_max = VALID_RANGES["HbA1c_level"]
    glucose_min, glucose_max = VALID_RANGES["blood_glucose_level"]

    return {
        "gender": gender,
        "age": _parse_float_field("age", data["age"], age_min, age_max),
        "hypertension": _parse_binary_field(
            "hypertension", data["hypertension"]
        ),
        "heart_disease": _parse_binary_field(
            "heart_disease", data["heart_disease"]
        ),
        "smoking_history": smoking_history,
        "bmi": _parse_float_field("bmi", data["bmi"], bmi_min, bmi_max),
        "HbA1c_level": _parse_float_field(
            "HbA1c_level", data["HbA1c_level"], hba1c_min, hba1c_max
        ),
        "blood_glucose_level": _parse_int_field(
            "blood_glucose_level",
            data["blood_glucose_level"],
            glucose_min,
            glucose_max,
        ),
    }


def parse_prediction_threshold(value, default: float | None = None) -> float:
    """
    Перетворює значення слайдера (відсотки) у поріг 0.0–1.0.

    Args:
        value: Значення з форми (наприклад, «50» для 50%).
        default: Поріг за замовчуванням; якщо None — PREDICTION_THRESHOLD.

    Returns:
        Валідний поріг у діапазоні THRESHOLD_MIN–THRESHOLD_MAX.

    Raises:
        InvalidPatientDataError: Якщо значення не число.
    """
    from config import (
        PREDICTION_THRESHOLD,
        THRESHOLD_MAX,
        THRESHOLD_MIN,
    )

    if default is None:
        default = PREDICTION_THRESHOLD

    try:
        percent = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidPatientDataError(
            "Поріг ймовірності має бути числом."
        ) from exc

    threshold = percent / 100.0
    if not THRESHOLD_MIN <= threshold <= THRESHOLD_MAX:
        raise InvalidPatientDataError(
            f"Поріг має бути в діапазоні "
            f"{int(THRESHOLD_MIN * 100)}–{int(THRESHOLD_MAX * 100)}%."
        )

    return round(threshold, 2)
