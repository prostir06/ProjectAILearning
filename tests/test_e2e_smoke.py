"""
Мінімальний smoke-тест Flask health endpoint.
"""

from diabetes.web.app import app


def test_health_smoke():
    """Базовий smoke: /health відповідає 200."""
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False

    with app.test_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert "models_ready" in payload
