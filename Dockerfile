# --- Build stage: install deps into a virtualenv-like layer --------------
FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLFLOW_ALLOW_FILE_STORE=true

# System deps needed by psycopg2 / lightgbm / catboost at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libpq-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- App layer -------------------------------------------------------------
COPY src/ ./src/
COPY api/ ./api/
COPY data/ ./data/
COPY models/ ./models/
COPY artifacts/ ./artifacts/
COPY app_dashboard.py ./app_dashboard.py
COPY start_server.py ./start_server.py

# Train champion model inside container build layer so scikit-learn version matches
RUN python data/generate_data.py && python -m src.train

# Non-root user
RUN useradd -m appuser
RUN mkdir -p /app/models /app/artifacts /app/mlruns && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
