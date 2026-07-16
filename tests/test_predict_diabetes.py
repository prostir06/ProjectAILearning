"""
Unit-тести для predict_diabetes.py.
"""

from unittest.mock import patch

import pytest

from exceptions import ModelNotFoundError, PredictionError
from predict_diabetes import (
    apply_threshold_to_results,
    build_prediction_summary,
    get_training_metrics,
    predict,
    predict_all,
    predict_with_summary,
    reset_pipeline_cache,
)


def test_predict_returns_expected_keys(sample_person, trained_pipeline):
    """Успішне передбачення повертає diabetes, label і probability."""
    with patch("predict_diabetes._get_pipeline", return_value=trained_pipeline):
        result = predict(sample_person)

    assert set(result.keys()) == {"diabetes", "label", "probability"}
    assert result["label"] in ("Так", "Ні")
    assert result["diabetes"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_all_returns_all_models(sample_person, trained_pipeline):
    """predict_all повертає результат для кожної моделі в пакеті."""
    bundle = {
        "models": {
            "random_forest": trained_pipeline,
            "decision_tree": trained_pipeline,
        },
        "metrics": {
            "random_forest": {"error_rate": 0.03, "accuracy": 0.97},
            "decision_tree": {"error_rate": 0.05, "accuracy": 0.95},
        },
        "model_labels": {
            "random_forest": "Random Forest",
            "decision_tree": "Decision Tree",
        },
    }

    with patch("predict_diabetes._get_bundle", return_value=bundle):
        with patch(
            "predict_diabetes.get_training_metrics",
            return_value=bundle["metrics"],
        ):
            results = predict_all(sample_person)

    assert len(results) == 2
    assert results[0]["model_key"] in ("random_forest", "decision_tree")
    assert "probability" in results[0]
    assert results[0]["error_rate"] == 0.03


def test_predict_all_sorted_by_selection_score(sample_person, trained_pipeline):
    """predict_all повертає моделі відсортовані за рейтингом."""
    bundle = {
        "models": {
            "low": trained_pipeline,
            "high": trained_pipeline,
        },
        "metrics": {
            "low": {"error_rate": 0.05, "selection_score": 0.80, "roc_auc": 0.9, "recall": 0.7, "f1": 0.7},
            "high": {"error_rate": 0.03, "selection_score": 0.90, "roc_auc": 0.95, "recall": 0.8, "f1": 0.85},
        },
        "model_labels": {"low": "Low", "high": "High"},
    }

    with patch("predict_diabetes._get_bundle", return_value=bundle):
        with patch(
            "predict_diabetes.get_training_metrics",
            return_value=bundle["metrics"],
        ):
            results = predict_all(sample_person)

    assert results[0]["model_key"] == "high"
    assert results[0]["rank"] == 1
    assert results[1]["rank"] == 2


def test_predict_label_matches_class(sample_person, trained_pipeline):
    """Текстова мітка відповідає числовому класу."""
    with patch("predict_diabetes._get_pipeline", return_value=trained_pipeline):
        result = predict(sample_person)

    if result["diabetes"] == 1:
        assert result["label"] == "Так"
    else:
        assert result["label"] == "Ні"


def test_predict_model_not_found(sample_person):
    """Відсутня модель викликає ModelNotFoundError."""
    with patch("predict_diabetes.MODELS_BUNDLE_PATH") as mock_path:
        mock_path.exists.return_value = False

        with pytest.raises(ModelNotFoundError, match="Моделі не знайдено"):
            predict(sample_person)


def test_predict_invalid_data_raises():
    """Некоректні дані викликають InvalidPatientDataError."""
    from exceptions import InvalidPatientDataError

    with pytest.raises(InvalidPatientDataError):
        predict({"gender": "Female"})


def test_reset_pipeline_cache():
    """Скидання кешу дозволяє повторно завантажити модель."""
    import predict_diabetes as module

    module._bundle = object()
    reset_pipeline_cache()
    assert module._bundle is None


def test_predict_with_summary_returns_models_and_summary(
    sample_person,
    trained_pipeline,
):
    """predict_with_summary повертає список моделей і загальний підсумок."""
    bundle = {
        "models": {"random_forest": trained_pipeline},
        "metrics": {"random_forest": {"error_rate": 0.03}},
        "model_labels": {"random_forest": "Random Forest"},
    }

    with patch("predict_diabetes._get_bundle", return_value=bundle):
        with patch(
            "predict_diabetes.get_training_metrics",
            return_value=bundle["metrics"],
        ):
            result = predict_with_summary(sample_person)

    assert "models" in result
    assert "summary" in result
    assert len(result["models"]) == 1
    assert result["summary"]["total_models"] == 1


def test_get_bundle_empty_models_raises(tmp_path):
    """Порожній пакет моделей викликає PredictionError."""
    import joblib
    from predict_diabetes import _get_bundle

    reset_pipeline_cache()
    empty_bundle = tmp_path / "empty.joblib"
    joblib.dump({"models": {}}, empty_bundle)

    with patch("predict_diabetes.MODELS_BUNDLE_PATH", empty_bundle):
        with pytest.raises(PredictionError, match="не містить"):
            _get_bundle()


def test_get_training_metrics_invalid_json(tmp_path):
    """Пошкоджений JSON метрик не ламає завантаження."""
    bad_json = tmp_path / "metrics.json"
    bad_json.write_text("{invalid", encoding="utf-8")

    with patch("predict_diabetes.METRICS_PATH", bad_json):
        with patch("predict_diabetes._get_bundle", side_effect=ModelNotFoundError):
            assert get_training_metrics() == {}


def test_build_prediction_summary_average_and_votes():
    """Загальний підсумок рахує середню ймовірність і голоси."""
    results = [
        {"diabetes": 0, "label": "Ні", "probability": 0.1},
        {"diabetes": 0, "label": "Ні", "probability": 0.2},
        {"diabetes": 1, "label": "Так", "probability": 0.8},
        {"diabetes": 0, "label": "Ні", "probability": 0.3},
    ]

    summary = build_prediction_summary(results, threshold=0.5)

    assert summary["total_models"] == 4
    assert summary["votes_yes"] == 1
    assert summary["votes_no"] == 3
    assert summary["probability"] == 0.35
    assert summary["label"] == "Ні"
    assert "1 з 4" in summary["votes_text"]


def test_build_prediction_summary_positive_when_average_above_threshold():
    """Середня ймовірність вище порогу дає «Так»."""
    results = [
        {"diabetes": 1, "label": "Так", "probability": 0.7},
        {"diabetes": 1, "label": "Так", "probability": 0.6},
    ]

    summary = build_prediction_summary(results, threshold=0.5)

    assert summary["label"] == "Так"
    assert summary["probability"] == 0.65


def test_build_prediction_summary_empty_results():
    """Порожній список викликає PredictionError."""
    with pytest.raises(PredictionError, match="підсумку"):
        build_prediction_summary([])


def test_apply_threshold_to_results():
    """apply_threshold_to_results змінює мітки за порогом."""
    results = [
        {"probability": 0.45, "diabetes": 0, "label": "Ні"},
        {"probability": 0.55, "diabetes": 0, "label": "Ні"},
    ]

    apply_threshold_to_results(results, threshold=0.5)

    assert results[0]["label"] == "Ні"
    assert results[1]["label"] == "Так"


def test_predict_with_summary_respects_custom_threshold(
    sample_person,
    trained_pipeline,
):
    """predict_with_summary застосовує користувацький поріг."""
    bundle = {
        "models": {"random_forest": trained_pipeline},
        "metrics": {"random_forest": {"error_rate": 0.03, "selection_score": 0.9}},
        "model_labels": {"random_forest": "Random Forest"},
    }

    with patch("predict_diabetes._get_bundle", return_value=bundle):
        with patch(
            "predict_diabetes.get_training_metrics",
            return_value=bundle["metrics"],
        ):
            result = predict_with_summary(sample_person, threshold=0.99)

    assert "summary" in result
    assert result["summary"]["label"] in ("Так", "Ні")


def test_get_bundle_corrupted_file(tmp_path):
    """Пошкоджений файл моделей викликає PredictionError."""
    from predict_diabetes import _get_bundle

    bad_model = tmp_path / "bad_models.joblib"
    bad_model.write_text("not a real model", encoding="utf-8")

    with patch("predict_diabetes.MODELS_BUNDLE_PATH", bad_model):
        with pytest.raises(PredictionError, match="завантажити моделі"):
            _get_bundle()


def test_get_pipeline_unknown_model(sample_person, trained_pipeline):
    """Невідомий ключ моделі викликає PredictionError."""
    from predict_diabetes import _get_pipeline

    bundle = {
        "models": {"random_forest": trained_pipeline},
        "default_model": "random_forest",
    }

    with patch("predict_diabetes._get_bundle", return_value=bundle):
        with pytest.raises(PredictionError, match="не знайдено"):
            _get_pipeline("unknown_model")


def test_get_feature_importance_from_json(tmp_path):
    """get_feature_importance читає валідний JSON-файл."""
    from predict_diabetes import get_feature_importance

    importance_file = tmp_path / "importance.json"
    importance_file.write_text(
        '[{"feature": "age", "label_uk": "Вік", "importance": 0.5}]',
        encoding="utf-8",
    )

    with patch("predict_diabetes.FEATURE_IMPORTANCE_PATH", importance_file):
        result = get_feature_importance()

    assert len(result) == 1
    assert result[0]["feature"] == "age"


def test_get_feature_importance_invalid_json_type(tmp_path):
    """JSON не-список ігнорується, повертається fallback."""
    from predict_diabetes import get_feature_importance

    bad_file = tmp_path / "importance.json"
    bad_file.write_text('{"not": "a list"}', encoding="utf-8")

    with patch("predict_diabetes.FEATURE_IMPORTANCE_PATH", bad_file):
        with patch("predict_diabetes._get_bundle", side_effect=ModelNotFoundError):
            assert get_feature_importance() == []


def test_get_training_metrics_invalid_type(tmp_path):
    """JSON метрик не-словник ігнорується."""
    bad_file = tmp_path / "metrics.json"
    bad_file.write_text("[1, 2, 3]", encoding="utf-8")

    with patch("predict_diabetes.METRICS_PATH", bad_file):
        with patch("predict_diabetes._get_bundle", side_effect=ModelNotFoundError):
            assert get_training_metrics() == {}


def test_get_selection_score_handles_invalid_values():
    """_get_selection_score стійкий до некоректних типів."""
    from predict_diabetes import _get_selection_score

    assert _get_selection_score(None) == 0.0
    assert _get_selection_score("bad") == 0.0
    assert _get_selection_score({"selection_score": "x"}) == 0.0
    assert _get_selection_score(
        {"roc_auc": 1.0, "recall": 1.0, "f1": 1.0}
    ) == 1.0
