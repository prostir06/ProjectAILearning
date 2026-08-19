"""
Unit-тести для entrypoint ``run.py``.
"""

from unittest.mock import MagicMock, patch

import run as run_module


def test_resolve_port_default(monkeypatch):
    """Без PORT у середовищі — 5000."""
    monkeypatch.delenv("PORT", raising=False)
    assert run_module._resolve_port() == 5000


def test_resolve_port_invalid_falls_back(monkeypatch):
    """Нечисловий PORT → default."""
    monkeypatch.setenv("PORT", "abc")
    assert run_module._resolve_port(default=5000) == 5000


def test_resolve_port_out_of_range_falls_back(monkeypatch):
    """PORT поза 1–65535 → default."""
    monkeypatch.setenv("PORT", "70000")
    assert run_module._resolve_port(default=5000) == 5000


def test_resolve_port_valid(monkeypatch):
    """Валідний PORT повертається як int."""
    monkeypatch.setenv("PORT", "5001")
    assert run_module._resolve_port() == 5001


def test_main_debug_runs_flask(monkeypatch):
    """FLASK_DEBUG=1 → create_app().run(...), код 0."""
    monkeypatch.setenv("FLASK_DEBUG", "1")
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "5000")

    fake_app = MagicMock()
    with patch("diabetes.web.app.create_app", return_value=fake_app):
        assert run_module.main() == 0

    fake_app.run.assert_called_once_with(
        debug=True,
        host="127.0.0.1",
        port=5000,
    )


def test_main_prod_uses_waitress(monkeypatch):
    """FLASK_DEBUG=0 → waitress.serve, код 0."""
    monkeypatch.setenv("FLASK_DEBUG", "0")
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "5000")

    fake_app = MagicMock()
    serve = MagicMock()
    waitress_mod = MagicMock(serve=serve)

    with patch("diabetes.web.app.create_app", return_value=fake_app):
        with patch.dict("sys.modules", {"waitress": waitress_mod}):
            assert run_module.main() == 0

    serve.assert_called_once_with(fake_app, host="0.0.0.0", port=5000)


def test_main_returns_1_when_create_app_fails(monkeypatch):
    """Збій create_app у debug → код 1."""
    monkeypatch.setenv("FLASK_DEBUG", "1")
    monkeypatch.setenv("PORT", "5000")

    with patch(
        "diabetes.web.app.create_app",
        side_effect=RuntimeError("broken"),
    ):
        assert run_module.main() == 1


def test_main_waitress_oserror_returns_1(monkeypatch):
    """Address already in use → код 1."""
    monkeypatch.setenv("FLASK_DEBUG", "0")
    monkeypatch.setenv("PORT", "5000")

    fake_app = MagicMock()
    serve = MagicMock(side_effect=OSError("address in use"))
    waitress_mod = MagicMock(serve=serve)

    with patch("diabetes.web.app.create_app", return_value=fake_app):
        with patch.dict("sys.modules", {"waitress": waitress_mod}):
            assert run_module.main() == 1
