"""Smoke: Flask application factory."""

import diabetes.web.app as app_module
from diabetes.web.app import create_app


def test_create_app_returns_flask_with_routes():
    application = create_app()
    application.config["TESTING"] = True
    application.config["WTF_CSRF_ENABLED"] = False

    client = application.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert "models_ready" in payload


def test_app_module_has_no_import_time_singleton():
    """Factory-only: модуль не створює app на імпорті."""
    assert not hasattr(app_module, "app")
