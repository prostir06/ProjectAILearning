"""
Серверний entrypoint: Flask (debug) або Waitress (prod).

Запуск: ``python run.py``
Змінні середовища: ``HOST``, ``PORT``, ``FLASK_DEBUG``.
"""

from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_port(default: int = 5000) -> int:
    """
    Читає ``PORT`` з середовища.

    Некоректні значення (літери, порожній рядок після strip) → ``default``.
    """
    raw = os.environ.get("PORT", str(default))
    try:
        port = int(raw)
    except (TypeError, ValueError):
        logger.warning("Некоректний PORT=%r, використано %s", raw, default)
        return default

    # Порт має бути в припустимому діапазоні TCP.
    if not 1 <= port <= 65535:
        logger.warning("PORT поза діапазоном 1–65535 (%s), використано %s", port, default)
        return default
    return port


def main() -> int:
    """
    Піднімає HTTP-сервер.

    Returns:
        Код виходу процесу (0 — ок, 1 — фатальна помилка старту).
    """
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = os.environ.get("HOST", "127.0.0.1") or "127.0.0.1"
    port = _resolve_port()

    try:
        from diabetes.web.app import create_app
    except Exception as exc:  # noqa: BLE001 — старт без traceback у консолі користувача
        logger.exception("Не вдалося імпортувати Flask-додаток: %s", exc)
        return 1

    try:
        application = create_app()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Не вдалося створити Flask-додаток: %s", exc)
        return 1

    if not debug_mode:
        try:
            from waitress import serve
        except ImportError:
            logger.warning(
                "waitress не встановлено; використано Flask development server."
            )
        else:
            try:
                serve(application, host=host, port=port)
                return 0
            except OSError as exc:
                # Типово: Address already in use.
                logger.exception("Waitress не зміг зайняти %s:%s: %s", host, port, exc)
                return 1
            except Exception as exc:  # noqa: BLE001
                logger.exception("Несподівана помилка Waitress: %s", exc)
                return 1

    try:
        application.run(debug=debug_mode, host=host, port=port)
        return 0
    except OSError as exc:
        logger.exception("Flask не зміг зайняти %s:%s: %s", host, port, exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Несподівана помилка Flask server: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
