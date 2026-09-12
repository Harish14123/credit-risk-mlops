"""
Wraps the model registry with the exact inference-time transformation the
training pipeline used (feature engineering must be identical between train
and serve, or you get silent train/serve skew). Loaded once at API startup
and cached in memory.
"""
from __future__ import annotations

import pandas as pd

from src.config import RISK_BUCKETS
from src.feature_engineering import add_engineered_features
from src.registry import load_production_model


class ModelService:
    def __init__(self):
        self.pipeline = None
        self.version = None
        self.entry = None

    def load(self):
        self.pipeline, self.version, self.entry = load_production_model()
        return self

    @property
    def model_name(self) -> str:
        return self.entry["model_name"]

    @property
    def threshold(self) -> float:
        return self.entry.get("threshold", 0.5)

    @property
    def metrics(self) -> dict:
        return self.entry.get("metrics", {})

    @property
    def created_at(self) -> str:
        return self.entry.get("created_at", "")

    def predict_one(self, application: dict) -> float:
        df = pd.DataFrame([application])
        df = add_engineered_features(df)
        proba = self.pipeline.predict_proba(df)[:, 1]
        return float(proba[0])

    def predict_batch(self, applications: list[dict]) -> list[float]:
        df = pd.DataFrame(applications)
        df = add_engineered_features(df)
        proba = self.pipeline.predict_proba(df)[:, 1]
        return [float(p) for p in proba]

    @staticmethod
    def bucket(probability: float) -> str:
        for lo, hi, label in RISK_BUCKETS:
            if lo <= probability < hi:
                return label
        return RISK_BUCKETS[-1][2]


model_service = ModelService()
