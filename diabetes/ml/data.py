"""
Завантаження датасету та train/validation/test split.
"""

from __future__ import annotations

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from diabetes.core.config import (
    DATA_PATH,
    FEATURES,
    QUICK_TRAIN_MAX_ROWS,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)
from diabetes.core.exceptions import DataLoadError


def load_data() -> pd.DataFrame:
    """
    Завантажує та очищує навчальний датасет.

    Returns:
        DataFrame без рядків із пропущеними значеннями в ознаках або цілі.

    Raises:
        DataLoadError: Якщо файл відсутній, порожній або пошкоджений.
    """
    if not DATA_PATH.exists():
        raise DataLoadError(f"Файл даних не знайдено: {DATA_PATH}")

    try:
        dataframe = pd.read_csv(DATA_PATH)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DataLoadError(f"Не вдалося прочитати CSV: {DATA_PATH}") from exc
    except UnicodeDecodeError as exc:
        raise DataLoadError(
            f"CSV має некоректне кодування: {DATA_PATH}"
        ) from exc
    except OSError as exc:
        raise DataLoadError(
            f"Помилка доступу до файлу даних: {DATA_PATH}"
        ) from exc

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise DataLoadError(
            "У CSV відсутні стовпці: "
            f"{', '.join(missing_columns)}."
        )

    cleaned = dataframe.dropna(subset=required_columns)
    if cleaned.empty:
        raise DataLoadError(
            "Після видалення пропусків датасет став порожнім."
        )

    return cleaned


def subsample_dataframe(
    dataframe: pd.DataFrame,
    max_rows: int,
) -> pd.DataFrame:
    """За потреби зменшує датасет зі stratify по цілі."""
    effective_max_rows = max(max_rows, QUICK_TRAIN_MAX_ROWS)
    if effective_max_rows <= 0 or len(dataframe) <= effective_max_rows:
        return dataframe

    sampled, _ = train_test_split(
        dataframe,
        train_size=effective_max_rows,
        random_state=42,
        stratify=dataframe[TARGET],
    )
    return sampled.reset_index(drop=True)


def split_dataset(features, target):
    """Робить 3-way split: train / validation / test."""
    try:
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            features,
            target,
            test_size=TEST_SIZE,
            random_state=42,
            stratify=target,
        )
        validation_ratio = VAL_SIZE / (1 - TEST_SIZE)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val,
            y_train_val,
            test_size=validation_ratio,
            random_state=42,
            stratify=y_train_val,
        )
    except ValueError as exc:
        raise DataLoadError(
            "Недостатньо даних для stratified train/val/test split."
        ) from exc

    return x_train, x_val, x_test, y_train, y_val, y_test


def compute_data_checksum() -> str | None:
    """Повертає SHA256 датасету, якщо файл доступний."""
    if not DATA_PATH.exists():
        return None

    digest = hashlib.sha256()
    try:
        with DATA_PATH.open("rb") as data_file:
            for chunk in iter(lambda: data_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None

    return digest.hexdigest()
