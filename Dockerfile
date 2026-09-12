# --- Build stage: install deps into a virtualenv-like layer --------------
FROM python:3.11-slim AS base

WORKDIR /app

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

# Verify model loader during build
RUN python -c "from api.model_loader import model_service; model_service.load()"

# Non-root user
RUN useradd -m appuser
RUN mkdir -p /app/models /app/artifacts /app/mlruns && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLFLOW_ALLOW_FILE_STORE=true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
