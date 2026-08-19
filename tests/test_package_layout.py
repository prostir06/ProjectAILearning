"""Smoke: package layout and BASE_DIR point at repo root."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_base_dir_is_repo_root():
    from diabetes.core.config import BASE_DIR, DATA_PATH, MODELS_BUNDLE_PATH

    assert BASE_DIR == REPO_ROOT
    assert DATA_PATH == REPO_ROOT / "diabetes_prediction_dataset.csv"
    assert MODELS_BUNDLE_PATH == REPO_ROOT / "diabetes_models.joblib"


def test_core_package_importable():
    import diabetes.core  # noqa: F401


def _imported_modules(py_file: Path) -> list[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_core_does_not_import_ml_or_web():
    """Шар core не залежить від ml/web."""
    core_dir = REPO_ROOT / "diabetes" / "core"
    for py_file in core_dir.glob("*.py"):
        for name in _imported_modules(py_file):
            assert not name.startswith("diabetes.ml")
            assert not name.startswith("diabetes.web")


def test_ml_does_not_import_web():
    """Шар ml не залежить від web."""
    ml_dir = REPO_ROOT / "diabetes" / "ml"
    for py_file in ml_dir.glob("*.py"):
        for name in _imported_modules(py_file):
            assert not name.startswith("diabetes.web")
