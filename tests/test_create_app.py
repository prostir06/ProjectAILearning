"""Smoke: Flask application factory."""

from unittest.mock import patch


def test_create_app_returns_flask_with_routes():
    with patch("diabetes.ml.bootstrap.ensure_models_ready", return_value=True):
        from diabetes.web.app import create_app

        application = create_app()
        application.config["TESTING"] = True
        application.config["WTF_CSRF_ENABLED"] = False

        client = application.test_client()
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["status"] == "ok"
        assert "models_ready" in payload
