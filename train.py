"""
CLI entrypoint для навчання моделей.

Запуск: ``python train.py`` (ті самі прапорці, що в ``diabetes.ml.train.main``).
"""

from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """
    Делегує в ``diabetes.ml.train.main`` з обробкою імпорту/збоїв.

    Returns:
        Код виходу CLI (0 — успіх).
    """
    try:
        from diabetes.ml.train import main as train_main
    except Exception as exc:  # noqa: BLE001
        logger.exception("Не вдалося імпортувати модуль навчання: %s", exc)
        return 1

    try:
        return int(train_main(argv))
    except SystemExit as exc:
        # argparse / явний sys.exit усередині train.main
        code = exc.code
        if code is None:
            return 0
        try:
            return int(code)
        except (TypeError, ValueError):
            return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Навчання завершилось з помилкою: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
