"""
CLI навчання моделей передбачення діабету.
"""

from __future__ import annotations

import argparse
import sys

from diabetes.core.config import BEST_MODELS_BUNDLE_PATH
from diabetes.core.exceptions import DataLoadError
from diabetes.ml.persist import (
    save_feature_importance,
    save_metrics_json,
    save_models_bundle,
)
from diabetes.ml.registry import MODEL_LABELS_UK

MODEL_SHORTCUTS = {
    "rf": "random_forest",
    "xgb": "xgboost",
    "lr": "logistic_regression",
    "hgb": "hist_gradient_boosting",
    "dt": "decision_tree",
    "ada": "adaboost",
}


def parse_model_keys(raw_models: str | None) -> list[str] | None:
    """Мапить короткі CLI-аліаси моделей у внутрішні ключі."""
    if not raw_models:
        return None

    resolved_keys: list[str] = []
    for raw_key in raw_models.split(","):
        model_key = raw_key.strip()
        if not model_key:
            continue
        model_key = MODEL_SHORTCUTS.get(model_key, model_key)
        if model_key not in MODEL_LABELS_UK:
            raise ValueError(f"Невідомий ключ моделі: {raw_key.strip()}")
        if model_key not in resolved_keys:
            resolved_keys.append(model_key)

    return resolved_keys or None


def build_arg_parser() -> argparse.ArgumentParser:
    """Створює CLI parser для скрипта навчання."""
    parser = argparse.ArgumentParser(
        description="Навчання моделей передбачення діабету."
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Вимкнути RandomizedSearchCV для топ-моделей.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Список моделей через кому: rf,xgb,lr,hgb,dt,ada.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Обмежити кількість рядків для швидкого навчання.",
    )
    parser.add_argument(
        "--serve-best-only",
        action="store_true",
        help="Зберегти лише best-only bundle.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Точка входу скрипта навчання.

    Returns:
        0 при успіху, 1 при помилці.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv if argv is not None else [])

    # Лінивий імпорт: уникаємо циклу cli ↔ train.
    from diabetes.ml.train import train_all_models

    try:
        selected_model_keys = parse_model_keys(args.models)
        (
            models,
            metrics,
            best_key,
            importance,
            optimal_threshold,
        ) = train_all_models(
            enable_tuning=not args.no_tune,
            max_rows=args.sample,
            model_keys=selected_model_keys,
        )

        if args.serve_best_only:
            save_models_bundle(
                models={best_key: models[best_key]},
                metrics={
                    best_key: metrics[best_key],
                    "_meta": metrics.get("_meta", {}),
                },
                best_model_key=best_key,
                feature_importance=importance,
                bundle_path=BEST_MODELS_BUNDLE_PATH,
                also_save_best=False,
                optimal_threshold=optimal_threshold,
            )
        else:
            save_models_bundle(
                models,
                metrics,
                best_key,
                importance,
                optimal_threshold=optimal_threshold,
            )

        save_metrics_json(metrics)
        save_feature_importance(importance)
    except (DataLoadError, ValueError, OSError) as exc:
        print(f"Помилка навчання: {exc}", file=sys.stderr)
        return 1

    return 0
