"""
Unit-тести для train_diabetes_model.py.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from exceptions import DataLoadError
import train_diabetes_model
from train_diabetes_model import (
    build_pipeline,
    compute_selection_score,
    evaluate_model,
    load_data,
    save_model,
    select_best_model_key,
    train_and_evaluate,
)


def test_build_pipeline_structure():
    """Pipeline містить кроки preprocessor, smote і classifier."""
    pipeline = build_pipeline()

    assert "preprocessor" in pipeline.named_steps
    assert "smote" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_load_data_success():
    """Реальний CSV завантажується без помилок."""
    dataframe = load_data()

    assert not dataframe.empty
    assert "diabetes" in dataframe.columns


def test_load_data_missing_file(tmp_path):
    """Відсутній файл даних викликає DataLoadError."""
    missing = tmp_path / "missing.csv"

    with patch("train_diabetes_model.DATA_PATH", missing):
        with pytest.raises(DataLoadError, match="не знайдено"):
            load_data()


def test_load_data_missing_columns(tmp_path):
    """CSV без потрібних стовпців викликає DataLoadError."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("gender,age\nFemale,30\n", encoding="utf-8")

    with patch("train_diabetes_model.DATA_PATH", bad_csv):
        with pytest.raises(DataLoadError, match="відсутні стовпці"):
            load_data()


def test_load_data_empty_after_dropna(tmp_path):
    """Повністю порожні рядки після dropna викликають DataLoadError."""
    empty_csv = tmp_path / "empty.csv"
    header = (
        "gender,age,hypertension,heart_disease,smoking_history,"
        "bmi,HbA1c_level,blood_glucose_level,diabetes\n"
    )
    empty_csv.write_text(header, encoding="utf-8")

    with patch("train_diabetes_model.DATA_PATH", empty_csv):
        with pytest.raises(DataLoadError, match="порожнім"):
            load_data()


def test_train_and_evaluate_with_tiny_data(tiny_dataframe):
    """Навчання на малому датасеті завершується успішно."""
    with patch("train_diabetes_model.load_data", return_value=tiny_dataframe):
        models, metrics, best_key, _ = train_diabetes_model.train_all_models(
            enable_tuning=False,
        )

    assert "random_forest" in models
    assert "random_forest" in metrics
    assert "roc_auc" in metrics["random_forest"]
    assert best_key in models


def test_train_all_models_returns_metrics(tiny_dataframe):
    """Кожен алгоритм має метрики accuracy та roc_auc."""
    with patch("train_diabetes_model.load_data", return_value=tiny_dataframe):
        _, metrics, _, _ = train_diabetes_model.train_all_models(
            enable_tuning=False,
        )

    for model_key, model_metrics in metrics.items():
        assert 0.0 <= model_metrics["accuracy"] <= 1.0
        assert 0.0 <= model_metrics["roc_auc"] <= 1.0
        assert "selection_score" in model_metrics


def test_compute_selection_score_weights():
    """Композитний бал враховує ROC-AUC, recall і F1."""
    metrics = {"roc_auc": 0.9, "recall": 0.8, "f1": 0.7}
    score = compute_selection_score(metrics)

    assert score == round(0.5 * 0.9 + 0.3 * 0.8 + 0.2 * 0.7, 4)


def test_compute_selection_score_missing_keys_returns_zero():
    """Неповні метрики дають 0.0 замість винятку."""
    assert compute_selection_score({"roc_auc": 0.9}) == 0.0
    assert compute_selection_score({}) == 0.0


def test_select_best_model_key():
    """Обирається модель із найвищим selection_score."""
    metrics = {
        "a": {"selection_score": 0.7},
        "b": {"selection_score": 0.85},
    }

    assert select_best_model_key(metrics) == "b"


def test_select_best_model_key_empty_raises():
    """Порожній словник метрик викликає DataLoadError."""
    with pytest.raises(DataLoadError, match="Немає навчених моделей"):
        select_best_model_key({})


def test_train_and_evaluate_single_class(tiny_dataframe):
    """Один клас у цільовій змінній викликає DataLoadError."""
    single_class = tiny_dataframe.copy()
    single_class["diabetes"] = 0

    with patch("train_diabetes_model.load_data", return_value=single_class):
        with pytest.raises(DataLoadError, match="менше двох класів"):
            train_and_evaluate()


def test_save_model_success(trained_pipeline, tmp_path):
    """Модель успішно зберігається на диск."""
    model_file = tmp_path / "model.joblib"
    save_model(trained_pipeline, model_file)

    assert model_file.exists()


def test_save_model_write_error(trained_pipeline, tmp_path):
    """Помилка запису піднімає OSError."""
    model_dir = tmp_path / "blocked"
    model_dir.mkdir()
    model_file = model_dir / "model.joblib"

    with patch("train_diabetes_model.joblib.dump", side_effect=OSError("denied")):
        with pytest.raises(OSError, match="зберегти модель"):
            save_model(trained_pipeline, model_file)


def test_evaluate_model_returns_metrics(trained_pipeline, tiny_dataframe):
    """evaluate_model повертає усі ключі метрик."""
    from config import FEATURES
    from train_diabetes_model import evaluate_model

    x_test = tiny_dataframe[FEATURES].iloc[:4]
    y_test = tiny_dataframe["diabetes"].iloc[:4]

    metrics = evaluate_model(trained_pipeline, x_test, y_test)

    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_extract_feature_importance(trained_pipeline):
    """extract_feature_importance повертає список ознак для tree-based моделі."""
    from train_diabetes_model import extract_feature_importance

    result = extract_feature_importance(trained_pipeline, top_n=3)

    assert isinstance(result, list)
    if result:
        assert "feature" in result[0]
        assert "importance" in result[0]


def test_build_pipeline_unknown_key():
    """Невідомий алгоритм викликає ValueError."""
    with pytest.raises(ValueError, match="Невідомий алгоритм"):
        build_pipeline("nonexistent_algorithm")


def test_save_metrics_json_success(tmp_path):
    """save_metrics_json записує файл на диск."""
    from train_diabetes_model import save_metrics_json

    metrics_file = tmp_path / "metrics.json"
    metrics = {"rf": {"accuracy": 0.9, "roc_auc": 0.85}}

    save_metrics_json(metrics, metrics_file)

    assert metrics_file.exists()
    assert "rf" in metrics_file.read_text(encoding="utf-8")


def test_save_feature_importance_success(tmp_path):
    """save_feature_importance записує JSON-файл."""
    from train_diabetes_model import save_feature_importance

    importance_file = tmp_path / "importance.json"
    data = [{"feature": "age", "label_uk": "Вік", "importance": 0.4}]

    save_feature_importance(data, importance_file)

    assert importance_file.exists()


def test_main_returns_error_on_data_load_failure():
    """main() повертає код 1 при DataLoadError."""
    from train_diabetes_model import main

    with patch(
        "train_diabetes_model.train_all_models",
        side_effect=DataLoadError("fail"),
    ):
        assert main() == 1


def test_load_data_corrupted_csv(tmp_path):
    """Пошкоджений CSV викликає DataLoadError."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_bytes(b"\xff\xfe invalid binary")

    with patch("train_diabetes_model.DATA_PATH", bad_csv):
        with pytest.raises(DataLoadError):
            load_data()
