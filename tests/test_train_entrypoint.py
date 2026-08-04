"""
Unit-тести для CLI entrypoint ``train.py``.
"""

from unittest.mock import patch

import train as train_entrypoint


def test_train_main_delegates_to_ml_train():
    """Успішний виклик diabetes.ml.train.main → його код виходу."""
    with patch("diabetes.ml.train.main", return_value=0) as train_main:
        assert train_entrypoint.main(["--help"]) == 0
        train_main.assert_called_once_with(["--help"])


def test_train_main_returns_1_on_train_failure():
    """Виняток під час навчання → код 1."""
    with patch(
        "diabetes.ml.train.main",
        side_effect=RuntimeError("fail"),
    ):
        assert train_entrypoint.main([]) == 1


def test_train_main_maps_system_exit_code():
    """SystemExit(2) з argparse → 2."""
    with patch(
        "diabetes.ml.train.main",
        side_effect=SystemExit(2),
    ):
        assert train_entrypoint.main([]) == 2


def test_train_main_system_exit_none_is_zero():
    """SystemExit() без коду → 0."""
    with patch(
        "diabetes.ml.train.main",
        side_effect=SystemExit(),
    ):
        assert train_entrypoint.main([]) == 0
