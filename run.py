"""Server entrypoint for Flask + Waitress."""

from __future__ import annotations

import logging
import os

from diabetes.web.app import app, create_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = os.environ.get("HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("PORT", "5000"))
    except ValueError:
        logger.warning("Некоректний PORT у середовищі, використано 5000")
        port = 5000

    application = create_app() if debug_mode else app

    if not debug_mode:
        try:
            from waitress import serve
        except ImportError:
            logger.warning(
                "waitress не встановлено; використано Flask dev server."
            )
        else:
            serve(application, host=host, port=port)
            return

    application.run(debug=debug_mode, host=host, port=port)


if __name__ == "__main__":
    main()
