"""
Pydantic schemas for the /predict API. Using Pydantic gives us automatic
request validation, OpenAPI docs, and type safety at the service boundary --
this is separate from (and complements) the offline data_validation.py used
in the training pipeline.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class LoanApplication(BaseModel):
    income: float = Field(..., gt=0, le=5_000_000, description="Annual gross income (USD)")
    age: int = Field(..., ge=18, le=100)
    loan_amount: float = Field(..., gt=0, le=2_000_000)
    employment_years: float = Field(..., ge=0, le=60)
    credit_score: int = Field(..., ge=300, le=850)
    existing_debt: float = Field(0.0, ge=0, le=5_000_000, description="Total current outstanding debt")
    num_credit_lines: int = Field(0, ge=0, le=100)
    num_delinquencies_2yr: int = Field(0, ge=0, le=50)
    loan_term_months: int = Field(36, ge=1, le=480)
    interest_rate: float = Field(..., ge=0, le=60)
    home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"] = "RENT"
    loan_purpose: Literal[
        "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
        "medical", "small_business", "car", "other",
    ] = "other"
    employment_type: Literal["salaried", "self_employed", "unemployed", "retired"] = "salaried"

    @field_validator("loan_amount")
    @classmethod
    def loan_amount_reasonable(cls, v, info):
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "income": 65000,
                "age": 32,
                "loan_amount": 250000,
                "employment_years": 6,
                "credit_score": 720,
                "existing_debt": 12000,
                "num_credit_lines": 5,
                "num_delinquencies_2yr": 0,
                "loan_term_months": 60,
                "interest_rate": 9.5,
                "home_ownership": "MORTGAGE",
                "loan_purpose": "home_improvement",
                "employment_type": "salaried",
            }
        }
    }


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    default_probability: float
    prediction: Literal["low_risk", "medium_risk", "high_risk"]
    model_version: str
    model_name: str
    threshold: float


class BatchPredictionRequest(BaseModel):
    applications: list[LoanApplication]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_version: Optional[str] = None
    model_name: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    production_version: str
    model_name: str
    metrics: dict
    threshold: float
    created_at: str
