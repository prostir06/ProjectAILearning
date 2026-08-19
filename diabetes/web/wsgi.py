"""
WSGI-точка входу для Waitress / gunicorn.

Створює додаток через factory один раз на процес.
Запуск: ``waitress-serve diabetes.web.wsgi:application``
"""

from diabetes.web.app import create_app

application = create_app()
