"""
Model Registry stage.

MLflow's tracking server logs every experiment run (params/metrics/artifacts)
-- see src/train.py. This module is a small, dependency-light *registry* on
top of that: it decides which trained model is "production", versions it
(v1, v2, ...), and persists a pointer the API reads at startup. This mirrors
what MLflow Model Registry / SageMaker Model Registry give you, without
requiring a running tracking server just to serve predictions.

registry.json layout:
{
  "current_production_version": "v3",
  "versions": {
    "v1": {"model_name": "xgboost", "path": "models/v1_xgboost.joblib",
           "metrics": {...}, "created_at": "...", "mlflow_run_id": "..."},
    ...
  }
}
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.config import MODEL_VERSION_PREFIX, MODELS_DIR, REGISTRY_PATH


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"current_production_version": None, "versions": {}}


def _save_registry(registry: dict):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def _next_version(registry: dict) -> str:
    existing = [
        int(v.replace(MODEL_VERSION_PREFIX, ""))
        for v in registry["versions"].keys()
        if v.startswith(MODEL_VERSION_PREFIX)
    ]
    n = max(existing) + 1 if existing else 1
    return f"{MODEL_VERSION_PREFIX}{n}"


def register_model(
    fitted_pipeline,
    model_name: str,
    metrics: dict,
    mlflow_run_id: str | None = None,
    threshold: float = 0.5,
    promote_to_production: bool = False,
) -> str:
    """Persist a fitted sklearn Pipeline to disk and record it in the registry."""
    registry = _load_registry()
    version = _next_version(registry)

    model_path = MODELS_DIR / f"{version}_{model_name}.joblib"
    joblib.dump(fitted_pipeline, model_path)

    registry["versions"][version] = {
        "model_name": model_name,
        "path": str(model_path.relative_to(MODELS_DIR.parent)),
        "metrics": metrics,
        "threshold": threshold,
        "mlflow_run_id": mlflow_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if promote_to_production or registry["current_production_version"] is None:
        registry["current_production_version"] = version

    _save_registry(registry)
    return version


def promote(version: str):
    registry = _load_registry()
    if version not in registry["versions"]:
        raise ValueError(f"Unknown model version '{version}'")
    registry["current_production_version"] = version
    _save_registry(registry)


def get_production_version() -> str | None:
    return _load_registry()["current_production_version"]


def load_production_model():
    registry = _load_registry()
    version = registry["current_production_version"]
    if version is None:
        raise RuntimeError("No production model registered yet. Run `python -m src.train` first.")
    entry = registry["versions"][version]
    model_path = MODELS_DIR.parent / entry["path"]
    pipeline = joblib.load(model_path)
    return pipeline, version, entry


def load_model_version(version: str):
    registry = _load_registry()
    if version not in registry["versions"]:
        raise ValueError(f"Unknown model version '{version}'")
    entry = registry["versions"][version]
    model_path = MODELS_DIR.parent / entry["path"]
    return joblib.load(model_path), entry


def list_versions() -> dict:
    return _load_registry()["versions"]
