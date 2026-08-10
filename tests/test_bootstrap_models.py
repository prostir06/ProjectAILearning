"""
Unit-тести для diabetes.ml.bootstrap.
"""

from unittest.mock import patch

import pytest

import diabetes.ml.bootstrap as bootstrap_models


def test_resolve_max_rows_uses_config_when_positive(monkeypatch):
    """QUICK_TRAIN_MAX_ROWS з config має пріоритет."""
    monkeypatch.setattr(bootstrap_models, "QUICK_TRAIN_MAX_ROWS", 5000)
    assert bootstrap_models._resolve_max_rows() == 5000


def test_resolve_max_rows_falls_back_on_bad_env(monkeypatch):
    """Некоректний env → 20000."""
    monkeypatch.setattr(bootstrap_models, "QUICK_TRAIN_MAX_ROWS", 0)
    monkeypatch.setenv("QUICK_TRAIN_MAX_ROWS", "abc")
    assert bootstrap_models._resolve_max_rows() == 20000


def test_ensure_models_ready_when_bundle_exists(tmp_path, monkeypatch):
    """Якщо joblib є — навчання не запускається."""
    bundle = tmp_path / "diabetes_models.joblib"
    bundle.write_bytes(b"ok")
    monkeypatch.setattr(bootstrap_models, "MODELS_BUNDLE_PATH", bundle)

    with patch("diabetes.ml.train.train_all_models") as train_mock:
        assert bootstrap_models.ensure_models_ready() is True
        train_mock.assert_not_called()


def test_ensure_models_ready_raises_runtime_error(tmp_path, monkeypatch):
    """Помилка навчання → RuntimeError."""
    missing = tmp_path / "missing.joblib"
    monkeypatch.setattr(bootstrap_models, "MODELS_BUNDLE_PATH", missing)
    monkeypatch.setattr(
        bootstrap_models,
        "BEST_MODELS_BUNDLE_PATH",
        tmp_path / "best.joblib",
    )

    with patch(
        "diabetes.ml.train.train_all_models",
        side_effect=OSError("disk"),
    ):
        with pytest.raises(RuntimeError, match="першому запуску"):
            bootstrap_models.ensure_models_ready()
