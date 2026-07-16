# Образ додатка передбачення діабету (Streamlit за замовчуванням).
# Збірка: docker compose build
# Запуск: docker compose up
FROM python:3.11-slim-bookworm

WORKDIR /app

# libgomp1 потрібен для XGBoost на Linux (аналог packages.txt для Streamlit Cloud).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# requirements-docker.txt: без pytest; xgboost<3 щоб уникнути великого CUDA wheel.
COPY requirements-docker.txt .
ENV PIP_DEFAULT_TIMEOUT=300
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .

EXPOSE 8501

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Перевірка готовності Streamlit (ендпоінт _stcore/health).
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
