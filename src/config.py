"""
Central configuration for the credit risk MLOps project.
All paths and constants live here so every module (training, API, tests)
references a single source of truth.
"""
import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw_loans.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_loans.csv"

MODELS_DIR = ROOT_DIR / "models"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REGISTRY_PATH = MODELS_DIR / "registry.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- MLflow --------------------------------------------------------------
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{ROOT_DIR / 'mlruns'}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit-risk-default-prediction")

# --- Database ------------------------------------------------------------
# Falls back to a local sqlite file if no Postgres DSN is provided, so the
# project runs out of the box without infra, while still being Postgres-ready
# for docker-compose / production.
raw_db_url = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{ROOT_DIR / 'artifacts' / 'predictions.db'}",
)
if raw_db_url.startswith("postgres://"):
    raw_db_url = raw_db_url.replace("postgres://", "postgresql://", 1)
DATABASE_URL = raw_db_url

# --- Modeling --------------------------------------------------------------
TARGET_COL = "default"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # taken out of the training split

RAW_NUMERIC_FEATURES = [
    "income",
    "age",
    "loan_amount",
    "employment_years",
    "credit_score",
    "existing_debt",
    "num_credit_lines",
    "num_delinquencies_2yr",
    "loan_term_months",
    "interest_rate",
]
RAW_CATEGORICAL_FEATURES = [
    "home_ownership",
    "loan_purpose",
    "employment_type",
]
RAW_FEATURES = RAW_NUMERIC_FEATURES + RAW_CATEGORICAL_FEATURES

# Risk bucket thresholds applied to predicted probability of default
RISK_BUCKETS = [
    (0.00, 0.10, "low_risk"),
    (0.10, 0.30, "medium_risk"),
    (0.30, 1.01, "high_risk"),
]

MODEL_VERSION_PREFIX = "v"
