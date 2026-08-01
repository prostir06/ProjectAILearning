"""
API-тести для Flask-додатку app.py.
"""

from unittest.mock import patch

import pytest

from app import app


@pytest.fixture
def client():
    """Flask test client для JSON API."""
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with patch("app.bootstrap_models.ensure_models_ready", return_value=True):
        with app.test_client() as test_client:
            yield test_client


def test_health_returns_ok(client):
    """GET /health повертає status=ok."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_api_predict_accepts_percent_threshold_and_best_mode(
    client,
    sample_person,
):
    """POST /api/predict приймає threshold у % і mode=best."""
    payload = dict(sample_person)
    payload["threshold"] = 30
    payload["mode"] = "best"
    mock_result = {
        "models": [
            {
                "model_key": "random_forest",
                "model_name": "Random Forest",
                "diabetes": 1,
                "label": "Так",
                "probability": 0.77,
                "is_best": True,
                "rank": 1,
            }
        ],
        "summary": {
            "model_name": "Загальний підсумок",
            "total_models": 1,
            "votes_yes": 1,
            "votes_no": 0,
            "votes_text": "1 з 1 алгоритмів — «Так», 0 з 1 — «Ні»",
            "probability": 0.77,
            "diabetes": 1,
            "label": "Так",
            "weighted": False,
        },
        "mode": "best",
    }

    with patch("app.validate_person_data", return_value=sample_person) as mock_validate:
        with patch("app.predict_with_summary", return_value=mock_result) as mock_predict:
            response = client.post("/api/predict", json=payload)

    assert response.status_code == 200
    assert response.get_json()["mode"] == "best"
    mock_validate.assert_called_once()
    mock_predict.assert_called_once_with(
        sample_person,
        threshold=0.3,
        mode="best",
    )
