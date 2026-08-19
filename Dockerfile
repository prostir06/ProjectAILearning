# Образ Flask-додатка передбачення діабету.
# Збірка: docker compose build
# Запуск: docker compose up
FROM python:3.11-slim-bookworm

WORKDIR /app

# libgomp1 потрібен для XGBoost на Linux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system appuser \
    && useradd --system --gid appuser --create-home --home-dir /home/appuser appuser

# requirements-docker.txt: без pytest; xgboost<3 щоб уникнути великого CUDA wheel.
COPY requirements-docker.txt .
ENV PIP_DEFAULT_TIMEOUT=300
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

ENV HOST=0.0.0.0 \
    PORT=5000 \
    FLASK_DEBUG=0

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health')" || exit 1

CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "diabetes.web.wsgi:application"]
