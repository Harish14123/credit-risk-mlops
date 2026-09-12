"""
Generates a synthetic but realistic loan-application dataset.

Why synthetic data? This project is built in an offline/sandboxed environment
with no access to Kaggle/data hosting sites. Rather than fake a "real"
dataset, we generate one from an explicit, documented data-generating
process (DGP) that encodes real credit-risk relationships (higher DTI, lower
credit score, more delinquencies -> higher default probability), plus noise
and a few missing/dirty values so the validation & preprocessing stages have
real work to do. Swap this module out for a real data loader (e.g. Lending
Club, HMDA, or an internal warehouse table) without touching anything
downstream -- the pipeline only depends on the column schema in src/config.py.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N = 25000

OUT_PATH = Path(__file__).resolve().parent / "raw_loans.csv"


def generate(n: int = N) -> pd.DataFrame:
    age = RNG.integers(21, 70, size=n)
    income = np.clip(RNG.normal(65000, 30000, size=n), 12000, 400000).round(2)
    employment_years = np.clip(
        RNG.gamma(shape=2.0, scale=3.0, size=n), 0, 40
    ).round(1)
    credit_score = np.clip(RNG.normal(680, 75, size=n), 300, 850).round(0)
    existing_debt = np.clip(RNG.normal(15000, 12000, size=n), 0, 200000).round(2)
    loan_amount = np.clip(RNG.normal(18000, 15000, size=n), 500, 500000).round(2)
    num_credit_lines = RNG.integers(0, 20, size=n)
    num_delinquencies_2yr = RNG.poisson(0.4, size=n)
    loan_term_months = RNG.choice([12, 24, 36, 48, 60, 84], size=n)
    interest_rate = np.clip(
        RNG.normal(13, 5, size=n) + (850 - credit_score) / 40, 3, 36
    ).round(2)

    home_ownership = RNG.choice(
        ["RENT", "OWN", "MORTGAGE", "OTHER"], size=n, p=[0.4, 0.2, 0.35, 0.05]
    )
    loan_purpose = RNG.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
         "medical", "small_business", "car", "other"],
        size=n,
        p=[0.30, 0.20, 0.12, 0.10, 0.08, 0.08, 0.07, 0.05],
    )
    employment_type = RNG.choice(
        ["salaried", "self_employed", "unemployed", "retired"],
        size=n, p=[0.65, 0.20, 0.05, 0.10]
    )

    debt_to_income = existing_debt / np.maximum(income, 1)
    loan_to_income = loan_amount / np.maximum(income, 1)

    # --- Ground-truth data generating process for default probability -----
    logit = (
        -3.6
        + 2.6 * debt_to_income
        + 1.9 * loan_to_income
        - 0.012 * (credit_score - 650)
        - 0.05 * employment_years
        + 0.35 * num_delinquencies_2yr
        + 0.04 * (interest_rate - 13)
        - 0.01 * (age - 40) * 0.05
        + np.where(employment_type == "unemployed", 1.1, 0.0)
        + np.where(employment_type == "self_employed", 0.15, 0.0)
        + np.where(home_ownership == "RENT", 0.15, -0.05)
    )
    noise = RNG.normal(0, 0.5, size=n)
    prob_default = 1 / (1 + np.exp(-(logit + noise)))
    default = RNG.binomial(1, prob_default)

    df = pd.DataFrame({
        "income": income,
        "age": age,
        "loan_amount": loan_amount,
        "employment_years": employment_years,
        "credit_score": credit_score,
        "existing_debt": existing_debt,
        "num_credit_lines": num_credit_lines,
        "num_delinquencies_2yr": num_delinquencies_2yr,
        "loan_term_months": loan_term_months,
        "interest_rate": interest_rate,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "employment_type": employment_type,
        "default": default,
    })

    # --- Inject realistic data-quality issues for the validation stage ----
    dirty_idx = RNG.choice(n, size=int(n * 0.02), replace=False)
    df.loc[dirty_idx, "income"] = np.nan
    dirty_idx2 = RNG.choice(n, size=int(n * 0.01), replace=False)
    df.loc[dirty_idx2, "credit_score"] = np.nan
    dup_idx = RNG.choice(n, size=int(n * 0.005), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)  # duplicate rows
    outlier_idx = RNG.choice(len(df), size=20, replace=False)
    df.loc[outlier_idx, "age"] = RNG.integers(120, 200, size=20)  # bad ages

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")
    print(f"Default rate: {df['default'].mean():.3%}")
