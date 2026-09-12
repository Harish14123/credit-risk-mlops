"""
Data Preprocessing stage.

Cleans raw, validated data:
  - Drops exact duplicate rows
  - Clips/repairs out-of-range values (e.g. impossible ages) to NaN so they
    get imputed rather than silently corrupting the model
  - Leaves imputation of missing values to an sklearn Pipeline (fit only on
    train, applied to val/test/inference) to avoid train/test leakage
"""
from __future__ import annotations

import pandas as pd

from src.config import TARGET_COL
from src.data_validation import VALID_RANGES


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Coerce numeric columns, turning garbage into NaN (to be imputed later)
    for col in VALID_RANGES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            lo, hi = VALID_RANGES[col]
            out_of_range = (df[col] < lo) | (df[col] > hi)
            df.loc[out_of_range, col] = pd.NA

    # Normalize categorical text (trim/upper where relevant)
    if "home_ownership" in df.columns:
        df["home_ownership"] = df["home_ownership"].astype(str).str.strip().str.upper()
    for col in ("loan_purpose", "employment_type"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    return df


def split_features_target(df: pd.DataFrame):
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])
    return X, y
