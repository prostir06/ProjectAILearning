"""
Unit-тести для Flask-додатку app.py.
"""

from unittest.mock import patch

import pytest

from app import app, format_metrics_for_display, get_error_message, load_metrics_rows, parse_form, parse_threshold_from_form
from exceptions import InvalidPatientDataError, ModelNotFoundError, PredictionError


@pytest.fixture
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_index_get_returns_form(client):
    """GET / повертає форму з усіма полями."""
    response = client.get("/")

    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Передбачення діабету" in html
    assert 'name="gender"' in html
    assert 'name="HbA1c_level"' in html
    assert 'name="prediction_threshold"' in html
    assert "form-section" in html
    assert 'name="smoking_history"' in html


def test_index_post_success(client, sample_person):
    """Успішний POST показує блок результату."""
    form_data = {key: str(value) for key, value in sample_person.items()}
    mock_prediction = {
        "models": [
            {
                "model_key": "random_forest",
                "model_name": "Випадковий ліс",
                "diabetes": 0,
                "label": "Ні",
                "probability": 0.12,
                "error_rate": 0.027,
                "accuracy": 0.973,
            }
        ],
        "summary": {
            "model_name": "Загальний підсумок",
            "total_models": 1,
            "votes_yes": 0,
            "votes_no": 1,
            "votes_text": "0 з 1 алгоритмів — «Так», 1 з 1 — «Ні»",
            "probability": 0.12,
            "diabetes": 0,
            "label": "Ні",
        },
    }

    with patch("app.predict_with_summary", return_value=mock_prediction):
        response = client.post("/", data=form_data)

    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "12%" in html
    assert "Ні" in html
    assert "Випадковий ліс" in html
    assert "Загальний підсумок" in html


def test_index_post_validation_error(client):
    """Некоректні дані показують повідомлення про помилку."""
    response = client.post(
        "/",
        data={
            "gender": "Female",
            "age": "200",
            "hypertension": "0",
            "heart_disease": "0",
            "smoking_history": "never",
            "bmi": "27.0",
            "HbA1c_level": "5.7",
            "blood_glucose_level": "120",
        },
    )

    assert response.status_code == 200
    assert "alert-error" in response.data.decode("utf-8")


def test_index_post_model_not_found(client, sample_person):
    """Відсутня модель відображається як помилка на сторінці."""
    form_data = {key: str(value) for key, value in sample_person.items()}

    with patch(
        "app.predict_with_summary",
        side_effect=ModelNotFoundError("Моделі не знайдено."),
    ):
        response = client.post("/", data=form_data)

    html = response.data.decode("utf-8")
    assert "Моделі не знайдено" in html


def test_parse_form_returns_all_keys():
    """parse_form повертає всі ключі DEFAULT_FORM."""
    class FakeForm:
        def __init__(self, data):
            self.data = data

        def __contains__(self, key):
            return key in self.data

        def get(self, key, default=""):
            return self.data.get(key, default)

    parsed = parse_form(FakeForm({"gender": "Male", "age": "30"}))

    assert "gender" in parsed
    assert "blood_glucose_level" in parsed
    assert parsed["gender"] == "Male"
    assert parsed["smoking_history"] == "No Info"


def test_format_metrics_for_display_sorts_by_selection_score():
    """Метрики сортуються за selection_score (рейтинг) від найвищого."""
    metrics = {
        "a": {
            "label_uk": "A",
            "accuracy": 0.9,
            "error_rate": 0.1,
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
            "roc_auc": 0.95,
            "selection_score": 0.82,
            "is_best": False,
        },
        "b": {
            "label_uk": "B",
            "accuracy": 0.95,
            "error_rate": 0.05,
            "precision": 0.9,
            "recall": 0.8,
            "f1": 0.85,
            "roc_auc": 0.92,
            "selection_score": 0.88,
            "is_best": True,
        },
        "c": {
            "label_uk": "C",
            "accuracy": 0.93,
            "error_rate": 0.07,
            "precision": 0.85,
            "recall": 0.75,
            "f1": 0.8,
            "roc_auc": 0.98,
            "selection_score": 0.85,
            "is_best": False,
        },
    }

    rows = format_metrics_for_display(metrics)

    assert [row["model_key"] for row in rows] == ["b", "c", "a"]
    assert rows[0]["rank"] == 1
    assert rows[0]["is_best"] is True


def test_format_metrics_for_display_invalid_input():
    """Некоректний тип metrics повертає порожній список."""
    assert format_metrics_for_display(None) == []
    assert format_metrics_for_display("bad") == []


def test_load_metrics_rows_handles_errors():
    """load_metrics_rows не піднімає виняток при збої."""
    with patch("app.get_training_metrics", side_effect=RuntimeError("fail")):
        assert load_metrics_rows() == []


def test_get_error_message_types():
    """get_error_message повертає зрозумілі тексти для різних винятків."""
    assert "Поле" in get_error_message(
        InvalidPatientDataError("Поле «age» має бути числом.")
    )
    assert "Модель" in get_error_message(
        ModelNotFoundError("Модель не знайдено.")
    )
    assert "передбачення" in get_error_message(
        PredictionError("збій")
    )
    assert "непередбачена" in get_error_message(RuntimeError("boom"))


def test_parse_form_handles_bad_form_data():
    """parse_form повертає DEFAULT_FORM при некоректному об'єкті форми."""
    from config import DEFAULT_FORM

    assert parse_form(None) == DEFAULT_FORM
    assert parse_form(42) == DEFAULT_FORM


def test_index_post_unexpected_error(client, sample_person):
    """Несподівана помилка показує загальне повідомлення."""
    form_data = {key: str(value) for key, value in sample_person.items()}

    with patch(
        "app.predict_with_summary",
        side_effect=RuntimeError("unexpected"),
    ):
        response = client.post("/", data=form_data)

    html = response.data.decode("utf-8")
    assert "непередбачена" in html


def test_format_metrics_skips_invalid_model_entry():
    """Некоректний запис метрик пропускається без падіння."""
    metrics = {
        "good": {
            "label_uk": "OK",
            "accuracy": 0.9,
            "error_rate": 0.1,
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
            "roc_auc": 0.85,
        },
        "bad": "not a dict",
    }

    rows = format_metrics_for_display(metrics)
    assert len(rows) == 1
    assert rows[0]["model_key"] == "good"


def test_parse_threshold_from_form_default():
    """parse_threshold_from_form повертає 0.5 за замовчуванням."""
    assert parse_threshold_from_form({}) == 0.5


def test_parse_threshold_from_form_custom():
    """parse_threshold_from_form читає значення слайдера."""
    class FakeForm:
        def __contains__(self, key):
            return key == "prediction_threshold"

        def get(self, key, default=""):
            return "30"

    assert parse_threshold_from_form(FakeForm()) == 0.3


def test_index_post_custom_threshold(client, sample_person):
    """POST передає користувацький поріг у predict_with_summary."""
    form_data = {key: str(value) for key, value in sample_person.items()}
    form_data["prediction_threshold"] = "30"
    mock_prediction = {
        "models": [],
        "summary": {
            "model_name": "Загальний підсумок",
            "total_models": 1,
            "votes_yes": 1,
            "votes_no": 0,
            "votes_text": "1 з 1",
            "probability": 0.45,
            "diabetes": 1,
            "label": "Так",
        },
    }

    with patch("app.predict_with_summary", return_value=mock_prediction) as mock_predict:
        response = client.post("/", data=form_data)

    mock_predict.assert_called_once()
    assert mock_predict.call_args.kwargs["threshold"] == 0.3
    assert response.status_code == 200
    assert "30%" in response.data.decode("utf-8")
