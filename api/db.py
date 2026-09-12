"""
Persistence layer for prediction logging (the "Monitoring / Logs" box in the
architecture diagram). Every /predict call is written to Postgres (or SQLite
locally, see src/config.DATABASE_URL) so we can later measure prediction
volume, score drift, and eventually join back to actual outcomes for
model-performance monitoring in production.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import DATABASE_URL

Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    default_probability = Column(Float, nullable=False)
    prediction = Column(String, nullable=False)
    income = Column(Float)
    age = Column(Integer)
    loan_amount = Column(Float)
    employment_years = Column(Float)
    credit_score = Column(Integer)
    latency_ms = Column(Float)


engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    Base.metadata.create_all(bind=engine)


def log_prediction(session, *, model_name, model_version, default_probability,
                    prediction, application: dict, latency_ms: float):
    try:
        entry = PredictionLog(
            id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc),
            model_name=str(model_name),
            model_version=str(model_version),
            default_probability=float(default_probability),
            prediction=str(prediction),
            income=float(application["income"]) if application.get("income") is not None else None,
            age=int(application["age"]) if application.get("age") is not None else None,
            loan_amount=float(application["loan_amount"]) if application.get("loan_amount") is not None else None,
            employment_years=float(application["employment_years"]) if application.get("employment_years") is not None else None,
            credit_score=int(application["credit_score"]) if application.get("credit_score") is not None else None,
            latency_ms=float(latency_ms),
        )
        session.add(entry)
        session.commit()
        return entry
    except Exception as e:
        session.rollback()
        raise e
