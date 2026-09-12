"""
FastAPI service exposing the credit-risk model.

Endpoints:
    GET  /health              liveness/readiness probe
    GET  /model/info          metadata about the currently-served model
    POST /predict              single loan application -> default probability
    POST /predict/batch        list of applications -> list of predictions
    POST /admin/reload         hot-reload the production model (after retrain)

Run locally:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.db import SessionLocal, init_db, log_prediction
from api.model_loader import model_service
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    LoanApplication,
    ModelInfoResponse,
    PredictionResponse,
)
from src.data_validation import validate_inference_row

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        model_service.load()
        logger.info(f"Loaded production model: {model_service.model_name} ({model_service.version})")
    except Exception as e:
        logger.error(f"Failed to load production model at startup: {e}")
    yield


app = FastAPI(
    title="Credit Risk / Loan Default Prediction API",
    description="Predicts probability of default for a loan application.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_response(proba: float) -> PredictionResponse:
    return PredictionResponse(
        default_probability=round(proba, 4),
        prediction=model_service.bucket(proba),
        model_version=model_service.version,
        model_name=model_service.model_name,
        threshold=model_service.threshold,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    if model_service.pipeline is None:
        return HealthResponse(status="degraded")
    return HealthResponse(
        status="ok", model_version=model_service.version, model_name=model_service.model_name
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    if model_service.pipeline is None:
        raise HTTPException(status_code=503, detail="No model loaded.")
    return ModelInfoResponse(
        production_version=model_service.version,
        model_name=model_service.model_name,
        metrics=model_service.metrics,
        threshold=model_service.threshold,
        created_at=model_service.created_at,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    if model_service.pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")

    try:
        record = application.model_dump()
        business_errors = validate_inference_row(record)
        if business_errors:
            raise HTTPException(status_code=422, detail=business_errors)

        t0 = time.time()
        proba = model_service.predict_one(record)
        latency_ms = (time.time() - t0) * 1000

        response = _to_response(proba)

        try:
            session = SessionLocal()
            try:
                log_prediction(
                    session,
                    model_name=response.model_name,
                    model_version=response.model_version,
                    default_probability=response.default_probability,
                    prediction=response.prediction,
                    application=record,
                    latency_ms=latency_ms,
                )
            finally:
                session.close()
        except Exception as e:
            logger.warning(f"Failed to log prediction to DB: {e}")

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    if model_service.pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")

    records = [app.model_dump() for app in request.applications]
    for record in records:
        errors = validate_inference_row(record)
        if errors:
            raise HTTPException(status_code=422, detail=errors)

    probas = model_service.predict_batch(records)
    responses = [_to_response(p) for p in probas]
    return BatchPredictionResponse(predictions=responses)


@app.post("/admin/reload")
def reload_model():
    """Hot-reloads whichever version is currently marked production in the
    registry, without restarting the API process (e.g. after a retrain +
    promote)."""
    try:
        model_service.load()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")
    return {
        "status": "reloaded",
        "model_name": model_service.model_name,
        "model_version": model_service.version,
    }
