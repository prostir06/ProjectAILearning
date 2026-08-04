"""Smoke: package layout and BASE_DIR point at repo root."""

from pathlib import Path


def test_base_dir_is_repo_root():
    from diabetes.core.config import BASE_DIR, DATA_PATH, MODELS_BUNDLE_PATH

    repo_root = Path(__file__).resolve().parents[1]
    assert BASE_DIR == repo_root
    assert DATA_PATH == repo_root / "diabetes_prediction_dataset.csv"
    assert MODELS_BUNDLE_PATH == repo_root / "diabetes_models.joblib"


def test_core_package_importable():
    import diabetes.core  # noqa: F401
