"""
Мінімальний smoke-тест Flask health endpoint.
"""

from diabetes.web.app import create_app


def test_health_smoke():
    """Базовий smoke: /health відповідає 200."""
    application = create_app()
    application.config["TESTING"] = True
    application.config["WTF_CSRF_ENABLED"] = False

    with application.test_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert "models_ready" in payload
