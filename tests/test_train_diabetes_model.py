"""
Unit-тести для diabetes.ml.train.
"""

import math
from unittest.mock import patch

import joblib
import pytest

import diabetes.ml.train as train_module
from diabetes.core.exceptions import DataLoadError
from diabetes.ml.train import (
    build_pipeline,
    evaluate_model,
    load_data,
    save_model,
    select_best_model_key,
    train_and_evaluate,
)


def test_build_pipeline_structure():
    """Default RF: preprocessor + classifier, без SMOTE."""
    pipeline = build_pipeline()

    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    assert "smote" not in pipeline.named_steps


def test_build_pipeline_logistic_uses_smote():
    """Логістична регресія включає крок SMOTE."""
    pipeline = build_pipeline("logistic_regression")

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

    with patch("diabetes.ml.data.DATA_PATH", missing):
        with pytest.raises(DataLoadError, match="не знайдено"):
            load_data()


def test_load_data_missing_columns(tmp_path):
    """CSV без потрібних стовпців викликає DataLoadError."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("gender,age\nFemale,30\n", encoding="utf-8")

    with patch("diabetes.ml.data.DATA_PATH", bad_csv):
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

    with patch("diabetes.ml.data.DATA_PATH", empty_csv):
        with pytest.raises(DataLoadError, match="порожнім"):
            load_data()


def test_train_and_evaluate_with_tiny_data(tiny_dataframe):
    """Навчання на малому датасеті завершується успішно."""
    with patch("diabetes.ml.train.load_data", return_value=tiny_dataframe):
        models, metrics, best_key, _, optimal_threshold = (
            train_module.train_all_models(
            enable_tuning=False,
        )
        )

    assert "random_forest" in models
    assert "random_forest" in metrics
    assert "roc_auc" in metrics["random_forest"]
    assert best_key in models
    assert 0.0 <= optimal_threshold <= 1.0


def test_train_all_models_returns_metrics(tiny_dataframe):
    """Кожен алгоритм має метрики accuracy та roc_auc."""
    with patch("diabetes.ml.train.load_data", return_value=tiny_dataframe):
        _, metrics, _, _, _ = train_module.train_all_models(
            enable_tuning=False,
        )

    for model_key, model_metrics in metrics.items():
        if model_key == "_meta":
            continue
        assert 0.0 <= model_metrics["accuracy"] <= 1.0
        assert 0.0 <= model_metrics["roc_auc"] <= 1.0
        assert 0.0 <= model_metrics["pr_auc"] <= 1.0
        assert "selection_score" in model_metrics

    assert "optimal_threshold" in metrics["_meta"]


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

    with patch("diabetes.ml.train.load_data", return_value=single_class):
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

    with patch("diabetes.ml.persist.joblib.dump", side_effect=OSError("denied")):
        with pytest.raises(OSError, match="зберегти модель"):
            save_model(trained_pipeline, model_file)


def test_evaluate_model_returns_metrics(trained_pipeline, tiny_dataframe):
    """evaluate_model повертає усі ключі метрик."""
    from diabetes.core.config import FEATURES

    x_test = tiny_dataframe[FEATURES].iloc[:4]
    y_test = tiny_dataframe["diabetes"].iloc[:4]

    metrics = evaluate_model(trained_pipeline, x_test, y_test)

    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_save_models_bundle_writes_metadata_and_best_only_file(
    trained_pipeline,
    tmp_path,
):
    """save_models_bundle зберігає metadata та окремий best-only bundle."""
    from diabetes.ml.train import save_models_bundle

    bundle_path = tmp_path / "models.joblib"
    best_bundle_path = tmp_path / "best.joblib"
    metrics = {
        "random_forest": {
            "accuracy": 0.9,
            "roc_auc": 0.88,
            "pr_auc": 0.84,
            "recall": 0.8,
            "f1": 0.79,
            "selection_score": 0.85,
            "is_best": True,
        }
    }

    with patch(
        "diabetes.ml.persist.BEST_MODELS_BUNDLE_PATH",
        best_bundle_path,
    ):
        save_models_bundle(
            models={"random_forest": trained_pipeline},
            metrics=metrics,
            best_model_key="random_forest",
            feature_importance=[],
            optimal_threshold=0.42,
            bundle_path=bundle_path,
        )

    bundle = joblib.load(bundle_path)
    best_bundle = joblib.load(best_bundle_path)

    assert bundle["metadata"]["optimal_threshold"] == 0.42
    assert "trained_at" in bundle["metadata"]
    assert "data_checksum" in bundle["metadata"]
    assert bundle["best_model"] == "random_forest"
    assert set(best_bundle["models"].keys()) == {"random_forest"}


def test_extract_feature_importance(trained_pipeline):
    """extract_feature_importance повертає список ознак для tree-based моделі."""
    from diabetes.ml.train import extract_feature_importance

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
    from diabetes.ml.train import save_metrics_json

    metrics_file = tmp_path / "metrics.json"
    metrics = {"rf": {"accuracy": 0.9, "roc_auc": 0.85}}

    save_metrics_json(metrics, metrics_file)

    assert metrics_file.exists()
    assert "rf" in metrics_file.read_text(encoding="utf-8")


def test_save_feature_importance_success(tmp_path):
    """save_feature_importance записує JSON-файл."""
    from diabetes.ml.train import save_feature_importance

    importance_file = tmp_path / "importance.json"
    data = [{"feature": "age", "label_uk": "Вік", "importance": 0.4}]

    save_feature_importance(data, importance_file)

    assert importance_file.exists()


def test_main_returns_error_on_data_load_failure():
    """main() повертає код 1 при DataLoadError."""
    from diabetes.ml.train import main

    with patch(
        "diabetes.ml.train.train_all_models",
        side_effect=DataLoadError("fail"),
    ):
        assert main() == 1


def test_load_data_corrupted_csv(tmp_path):
    """Пошкоджений CSV викликає DataLoadError."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_bytes(b"\xff\xfe invalid binary")

    with patch("diabetes.ml.data.DATA_PATH", bad_csv):
        with pytest.raises(DataLoadError):
            load_data()


def test_resolve_scale_pos_weight_uses_train_labels(tiny_dataframe):
    """_resolve_scale_pos_weight обчислює ratio з реальних міток train."""
    from diabetes.ml.registry import DEFAULT_SCALE_POS_WEIGHT

    y_train = tiny_dataframe["diabetes"]
    weight = train_module._resolve_scale_pos_weight(y_train)

    assert weight >= DEFAULT_SCALE_POS_WEIGHT
    assert math.isfinite(weight)


def test_resolve_scale_pos_weight_fallback_on_compute_error():
    """Несподіваний збій compute_scale_pos_weight не зупиняє навчання."""
    from diabetes.ml.registry import DEFAULT_SCALE_POS_WEIGHT

    with patch(
        "diabetes.ml.registry.compute_scale_pos_weight",
        side_effect=RuntimeError("unexpected"),
    ):
        weight = train_module._resolve_scale_pos_weight([0, 1, 0])

    assert weight == DEFAULT_SCALE_POS_WEIGHT


def test_train_all_models_passes_scale_pos_weight_to_pipelines(tiny_dataframe):
    """train_all_models передає обчислений scale_pos_weight у get_model_pipelines."""
    with patch("diabetes.ml.train.load_data", return_value=tiny_dataframe):
        with patch("diabetes.ml.train.get_model_pipelines") as get_pipelines:
            get_pipelines.return_value = {}
            with pytest.raises(DataLoadError, match="Немає моделей"):
                train_module.train_all_models(
                    enable_tuning=False,
                    model_keys=["xgboost"],
                )

    _, kwargs = get_pipelines.call_args
    assert "scale_pos_weight" in kwargs
    assert kwargs["scale_pos_weight"] >= 1.0


def test_split_dataset_returns_three_partitions(tiny_dataframe):
    """3-way split зберігає всі рядки."""
    from diabetes.core.config import FEATURES
    from diabetes.ml.data import split_dataset

    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        tiny_dataframe[FEATURES],
        tiny_dataframe["diabetes"],
    )
    total = len(x_train) + len(x_val) + len(x_test)
    assert total == len(tiny_dataframe)
    assert len(y_train) == len(x_train)
    assert len(y_val) == len(x_val)
    assert len(y_test) == len(x_test)


def test_find_optimal_threshold_in_bounds(trained_pipeline, tiny_dataframe):
    """Youden-поріг лежить у дозволеному діапазоні."""
    from diabetes.core.config import FEATURES, THRESHOLD_MAX, THRESHOLD_MIN
    from diabetes.ml.evaluate import find_optimal_threshold

    x_val = tiny_dataframe[FEATURES]
    y_val = tiny_dataframe["diabetes"]
    threshold = find_optimal_threshold(trained_pipeline, x_val, y_val)
    assert THRESHOLD_MIN <= threshold <= THRESHOLD_MAX
