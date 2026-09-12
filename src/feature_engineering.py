"""
Feature Engineering stage.

Adds domain-specific derived features that materially help credit-risk
models (these mirror real underwriting ratios used by lenders), and builds
the sklearn ColumnTransformer used to preprocess numeric + categorical
features consistently across train/val/test/inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RAW_CATEGORICAL_FEATURES, RAW_NUMERIC_FEATURES

ENGINEERED_NUMERIC_FEATURES = [
    "debt_to_income",
    "loan_to_income",
    "credit_utilization_proxy",
    "income_per_credit_line",
    "delinquency_rate",
]

ALL_NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
ALL_CATEGORICAL_FEATURES = RAW_CATEGORICAL_FEATURES


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    income_safe = X["income"].clip(lower=1)

    X["debt_to_income"] = X["existing_debt"] / income_safe
    X["loan_to_income"] = X["loan_amount"] / income_safe
    # Proxy for utilization: how much of the loan relative to total credit lines held
    X["credit_utilization_proxy"] = X["loan_amount"] / (X["num_credit_lines"].clip(lower=0) + 1)
    X["income_per_credit_line"] = income_safe / (X["num_credit_lines"].clip(lower=0) + 1)
    X["delinquency_rate"] = X["num_delinquencies_2yr"] / (X["employment_years"].clip(lower=0.5))

    # Cap extreme ratios (protects linear models from outlier leverage)
    for col in ["debt_to_income", "loan_to_income", "credit_utilization_proxy",
                "income_per_credit_line", "delinquency_rate"]:
        X[col] = X[col].clip(upper=X[col].quantile(0.995))

    return X


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, ALL_NUMERIC_FEATURES),
        ("cat", categorical_pipeline, ALL_CATEGORICAL_FEATURES),
    ])
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())
