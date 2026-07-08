"""
Unit-тести для exceptions.py — ієрархія користувацьких винятків.
"""

from exceptions import (
    DataLoadError,
    DiabetesProjectError,
    InvalidPatientDataError,
    ModelNotFoundError,
    PredictionError,
)


def test_diabetes_project_error_is_base():
    """Усі винятки проєкту успадковують DiabetesProjectError."""
    for error_class in (
        ModelNotFoundError,
        DataLoadError,
        InvalidPatientDataError,
        PredictionError,
    ):
        assert issubclass(error_class, DiabetesProjectError)


def test_model_not_found_is_file_not_found():
    """ModelNotFoundError сумісний із FileNotFoundError."""
    assert issubclass(ModelNotFoundError, FileNotFoundError)


def test_invalid_patient_data_is_value_error():
    """InvalidPatientDataError сумісний із ValueError."""
    assert issubclass(InvalidPatientDataError, ValueError)


def test_exception_messages_preserved():
    """Повідомлення винятку доступне через str()."""
    message = "Тестове повідомлення"
    assert str(InvalidPatientDataError(message)) == message
    assert str(PredictionError(message)) == message
