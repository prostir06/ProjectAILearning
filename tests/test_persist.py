"""
Unit-тести для diabetes.ml.persist (метадані бандла, JSON, best-only).
"""

from unittest.mock import patch

import pytest

from diabetes.ml.persist import (
    build_bundle_metadata,
    save_feature_importance,
    save_metrics_json,
    save_models_bundle,
)


def test_build_bundle_metadata_contains_expected_keys():
    """Метадані завжди мають стабільний набір ключів."""
    meta = build_bundle_metadata(0.48)

    assert set(meta) >= {
        "trained_at",
        "sklearn_version",
        "xgboost_version",
        "data_checksum",
        "optimal_threshold",
    }
    assert meta["optimal_threshold"] == 0.48


def test_build_bundle_metadata_survives_checksum_failure():
    """Збій checksum не зупиняє збір метаданих."""
    with patch(
        "diabetes.ml.persist.compute_data_checksum",
        side_effect=OSError("disk"),
    ):
        meta = build_bundle_metadata(None)

    assert meta["data_checksum"] is None
    assert "sklearn_version" in meta


def test_save_metrics_json_rejects_unserializable(tmp_path):
    """Об'єкт, який json не вміє серіалізувати → OSError."""
    metrics_file = tmp_path / "metrics.json"

    with pytest.raises(OSError, match="метрики"):
        save_metrics_json({"rf": {"pipe": object()}}, metrics_file)


def test_save_feature_importance_rejects_unserializable(tmp_path):
    """Несеріалізована importance → OSError."""
    importance_file = tmp_path / "imp.json"

    with pytest.raises(OSError, match="важливість"):
        save_feature_importance([{"feature": object()}], importance_file)


def test_save_models_bundle_missing_best_key_raises(trained_pipeline, tmp_path):
    """also_save_best без ключа best_model → OSError."""
    bundle_path = tmp_path / "models.joblib"

    with pytest.raises(OSError, match="відсутня"):
        save_models_bundle(
            models={"random_forest": trained_pipeline},
            metrics={},
            best_model_key="xgboost",
            feature_importance=[],
            bundle_path=bundle_path,
            also_save_best=True,
        )
