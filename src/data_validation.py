"""
Data Validation stage (first box in the architecture diagram).

Responsibilities:
  - Enforce schema (required columns + dtypes)
  - Enforce business-rule ranges (e.g. age must be plausible, credit_score
    within 300-850)
  - Detect duplicate rows
  - Report missingness
  - Fail loudly (raise) on structural problems, but only WARN on things that
    preprocessing can safely repair (missing values, duplicates).

This is intentionally dependency-free (no great_expectations) so it runs
anywhere, but it mimics that pattern: a list of declarative "expectations"
each returning a pass/fail + message, aggregated into a validation report.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.config import RAW_CATEGORICAL_FEATURES, RAW_NUMERIC_FEATURES, TARGET_COL

logger = logging.getLogger(__name__)

VALID_RANGES = {
    "age": (18, 100),
    "income": (0, 5_000_000),
    "loan_amount": (0, 2_000_000),
    "employment_years": (0, 60),
    "credit_score": (300, 850),
    "existing_debt": (0, 5_000_000),
    "num_credit_lines": (0, 100),
    "num_delinquencies_2yr": (0, 50),
    "loan_term_months": (1, 480),
    "interest_rate": (0, 60),
}

VALID_CATEGORIES = {
    "home_ownership": {"RENT", "OWN", "MORTGAGE", "OTHER"},
    "loan_purpose": {
        "debt_consolidation", "credit_card", "home_improvement",
        "major_purchase", "medical", "small_business", "car", "other",
    },
    "employment_type": {"salaried", "self_employed", "unemployed", "retired"},
}


@dataclass
class ValidationReport:
    n_rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_by_col: dict = field(default_factory=dict)
    n_duplicates: int = 0
    n_out_of_range: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [f"ValidationReport(n_rows={self.n_rows}, valid={self.is_valid})"]
        if self.errors:
            lines.append("  ERRORS:")
            lines += [f"    - {e}" for e in self.errors]
        if self.warnings:
            lines.append("  WARNINGS:")
            lines += [f"    - {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_raw_data(df: pd.DataFrame, require_target: bool = True) -> ValidationReport:
    report = ValidationReport(n_rows=len(df))

    # 1. Schema check --------------------------------------------------
    required_cols = list(RAW_NUMERIC_FEATURES) + list(RAW_CATEGORICAL_FEATURES)
    if require_target:
        required_cols = required_cols + [TARGET_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        report.errors.append(f"Missing required columns: {missing_cols}")
        # Can't do further checks safely without the columns
        return report

    # 2. Empty dataset ---------------------------------------------------
    if len(df) == 0:
        report.errors.append("Dataset has zero rows.")
        return report

    # 3. Dtype sanity: numeric columns must be coercible to numeric --------
    for col in RAW_NUMERIC_FEATURES:
        coerced = pd.to_numeric(df[col], errors="coerce")
        bad = coerced.isna() & df[col].notna()
        if bad.sum() > 0:
            report.errors.append(
                f"Column '{col}' has {bad.sum()} non-numeric values."
            )

    # 4. Missingness report (warning only, preprocessing imputes) ----------
    missing = df[required_cols].isna().sum()
    report.missing_by_col = {k: int(v) for k, v in missing.items() if v > 0}
    for col, n_missing in report.missing_by_col.items():
        pct = n_missing / len(df)
        report.warnings.append(f"Column '{col}' has {n_missing} missing ({pct:.2%}).")
        if pct > 0.5:
            report.errors.append(f"Column '{col}' is >50% missing; unusable.")

    # 5. Duplicate rows --------------------------------------------------
    n_dupes = int(df.duplicated().sum())
    report.n_duplicates = n_dupes
    if n_dupes > 0:
        report.warnings.append(f"Found {n_dupes} exact duplicate rows.")

    # 6. Range checks ------------------------------------------------------
    for col, (lo, hi) in VALID_RANGES.items():
        series = pd.to_numeric(df[col], errors="coerce")
        out_of_range = ((series < lo) | (series > hi)) & series.notna()
        n_bad = int(out_of_range.sum())
        if n_bad > 0:
            report.n_out_of_range[col] = n_bad
            report.warnings.append(
                f"Column '{col}' has {n_bad} values outside plausible range [{lo}, {hi}]."
            )

    # 7. Categorical domain checks ------------------------------------------
    for col, allowed in VALID_CATEGORIES.items():
        bad_vals = set(df[col].dropna().unique()) - allowed
        if bad_vals:
            report.errors.append(f"Column '{col}' has unexpected categories: {bad_vals}")

    # 8. Target sanity -----------------------------------------------------
    if require_target:
        bad_target = set(df[TARGET_COL].dropna().unique()) - {0, 1}
        if bad_target:
            report.errors.append(f"Target column has non-binary values: {bad_target}")

    logger.info(report.summary())
    return report


def validate_inference_row(record: dict) -> list[str]:
    """Lightweight validation used at inference time (single record, no target)."""
    errors = []
    for col, (lo, hi) in VALID_RANGES.items():
        if col in record and record[col] is not None:
            val = record[col]
            if not (lo <= val <= hi):
                errors.append(f"'{col}'={val} is outside plausible range [{lo}, {hi}].")
    for col, allowed in VALID_CATEGORIES.items():
        if col in record and record[col] is not None and record[col] not in allowed:
            errors.append(f"'{col}'={record[col]!r} not in allowed set {allowed}.")
    return errors


if __name__ == "__main__":
    from src.config import RAW_DATA_PATH
    df = pd.read_csv(RAW_DATA_PATH)
    report = validate_raw_data(df)
    print(report.summary())
