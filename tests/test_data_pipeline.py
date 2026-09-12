import numpy as np
import pandas as pd
import pytest

from src.data_validation import validate_raw_data, validate_inference_row
from src.preprocessing import clean_raw_data, split_features_target
from src.feature_engineering import add_engineered_features, build_preprocessor
from src.config import RAW_NUMERIC_FEATURES, RAW_CATEGORICAL_FEATURES, TARGET_COL


def make_good_df(n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "income": rng.uniform(20000, 150000, n),
        "age": rng.integers(20, 65, n),
        "loan_amount": rng.uniform(1000, 50000, n),
        "employment_years": rng.uniform(0, 20, n),
        "credit_score": rng.integers(500, 820, n),
        "existing_debt": rng.uniform(0, 30000, n),
        "num_credit_lines": rng.integers(0, 10, n),
        "num_delinquencies_2yr": rng.integers(0, 3, n),
        "loan_term_months": rng.choice([12, 24, 36, 60], n),
        "interest_rate": rng.uniform(4, 25, n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
        "loan_purpose": rng.choice(["car", "medical", "other"], n),
        "employment_type": rng.choice(["salaried", "self_employed"], n),
        TARGET_COL: rng.integers(0, 2, n),
    })


class TestValidation:
    def test_valid_data_passes(self):
        df = make_good_df()
        report = validate_raw_data(df)
        assert report.is_valid

    def test_missing_columns_fails(self):
        df = make_good_df().drop(columns=["income"])
        report = validate_raw_data(df)
        assert not report.is_valid
        assert any("income" in e for e in report.errors)

    def test_bad_category_fails(self):
        df = make_good_df()
        df.loc[0, "home_ownership"] = "SPACESHIP"
        report = validate_raw_data(df)
        assert not report.is_valid

    def test_non_binary_target_fails(self):
        df = make_good_df()
        df.loc[0, TARGET_COL] = 5
        report = validate_raw_data(df)
        assert not report.is_valid

    def test_missingness_reported_as_warning_not_error(self):
        df = make_good_df()
        df.loc[0:3, "income"] = np.nan
        report = validate_raw_data(df)
        assert report.is_valid  # small amount of missingness is repairable
        assert "income" in report.missing_by_col

    def test_inference_row_out_of_range(self):
        errors = validate_inference_row({"credit_score": 999, "age": 30})
        assert len(errors) == 1
        assert "credit_score" in errors[0]

    def test_inference_row_valid(self):
        errors = validate_inference_row({"credit_score": 700, "age": 30, "home_ownership": "RENT"})
        assert errors == []


class TestPreprocessing:
    def test_clean_drops_duplicates(self):
        df = make_good_df(n=10)
        df_with_dupes = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
        cleaned = clean_raw_data(df_with_dupes)
        assert len(cleaned) == 10

    def test_clean_nullifies_out_of_range(self):
        df = make_good_df(n=5)
        df.loc[0, "age"] = 500  # impossible
        cleaned = clean_raw_data(df)
        assert pd.isna(cleaned.loc[0, "age"])

    def test_split_features_target(self):
        df = make_good_df()
        X, y = split_features_target(df)
        assert TARGET_COL not in X.columns
        assert set(y.unique()) <= {0, 1}


class TestFeatureEngineering:
    def test_adds_expected_columns(self):
        df = make_good_df()
        X, _ = split_features_target(df)
        X_fe = add_engineered_features(X)
        for col in ["debt_to_income", "loan_to_income", "credit_utilization_proxy",
                    "income_per_credit_line", "delinquency_rate"]:
            assert col in X_fe.columns

    def test_no_division_by_zero_errors(self):
        df = make_good_df(n=5)
        df["income"] = 0  # edge case: zero income
        df["num_credit_lines"] = 0
        X, _ = split_features_target(df)
        X_fe = add_engineered_features(X)
        assert np.isfinite(X_fe["debt_to_income"]).all()
        assert np.isfinite(X_fe["income_per_credit_line"]).all()

    def test_preprocessor_fits_and_transforms(self):
        df = make_good_df(n=30)
        X, _ = split_features_target(df)
        X_fe = add_engineered_features(X)
        preprocessor = build_preprocessor()
        transformed = preprocessor.fit_transform(X_fe)
        assert transformed.shape[0] == 30

    def test_preprocessor_handles_missing_values(self):
        df = make_good_df(n=30)
        df.loc[0, "income"] = np.nan
        X, _ = split_features_target(df)
        X_fe = add_engineered_features(X)
        preprocessor = build_preprocessor()
        transformed = preprocessor.fit_transform(X_fe)
        assert not np.isnan(transformed).any()
