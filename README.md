# Credit Risk / Loan Default Prediction — End-to-End ML System

Predicts the probability that a loan applicant will default, served as a
production-style API — not just a notebook model. This repo covers the full
lifecycle: data validation → feature engineering → multi-model training →
evaluation → experiment tracking → model registry → REST API → containerized
deployment → CI/CD.

```
Raw Dataset → Data Validation → Preprocessing → Feature Engineering
     → [Logistic Regression | Random Forest | XGBoost | LightGBM | CatBoost]
     → Model Evaluation → Model Registry → FastAPI → Docker → Cloud Deploy → Monitoring/Logs
```

## Why synthetic data?

This was built in a sandboxed environment with no access to Kaggle/data
hosting sites, so `data/generate_data.py` generates a dataset from an
**explicit, documented data-generating process**: default probability is a
logistic function of debt-to-income, loan-to-income, credit score,
employment history, and delinquencies, plus noise — with missing values,
duplicate rows, and bad values (impossible ages) injected on purpose so the
validation stage has real problems to catch. Swap this module for a real
loader (Lending Club, HMDA, an internal warehouse table) and nothing
downstream needs to change — every stage only depends on the column schema
in `src/config.py`.

## Results

5 models trained on the same train/val/test split (70/10/20, stratified),
threshold tuned per-model on the validation set to maximize F1, all metrics
below computed on the **held-out test set**:

| Model               | ROC-AUC | PR-AUC | F1     | Precision | Recall | Brier  |
|---------------------|---------|--------|--------|-----------|--------|--------|
| Logistic Regression | 0.8509  | 0.6524 | 0.5813 | 0.6265    | 0.5422 | 0.1501 |
| CatBoost             | 0.8507  | 0.6470 | 0.5755 | 0.6044    | 0.5492 | 0.1384 |
| XGBoost              | 0.8445  | 0.6414 | 0.5614 | 0.5086    | 0.6265 | 0.1318 |
| LightGBM             | 0.8401  | 0.6401 | 0.5607 | 0.5256    | 0.6007 | 0.1271 |
| Random Forest        | 0.8423  | 0.6295 | 0.5663 | 0.6013    | 0.5351 | 0.1398 |

**Champion selection: PR-AUC**, not accuracy or ROC-AUC. Defaults are the
rare, costly class (~17% base rate here); a model that predicts "no default"
for everyone would score ~83% accuracy while being useless, and ROC-AUC can
look deceptively strong under imbalance. PR-AUC is far more sensitive to how
well the model actually ranks the minority (default) class, which is what a
lender is paying for. On this run **Logistic Regression wins** — a reminder
that a well-regularized linear model with engineered ratio features is a
completely legitimate production choice, not just a "baseline to beat."
Gradient boosting models offer higher recall (useful if missed defaults are
costlier than rejected good customers) and better calibration (lower Brier
score), which is a real trade-off a risk team would weigh, not a bug.

Full confusion matrices and calibration curve plots per model are written to
`artifacts/` after training; run history (params, metrics, model artifacts)
lives in MLflow.

## Project layout

```
credit-risk-mlops/
├── data/
│   ├── generate_data.py       # synthetic data-generating process
│   └── raw_loans.csv          # generated dataset (git-ignored, regenerate it)
├── src/
│   ├── config.py               # single source of truth for paths/constants
│   ├── data_validation.py      # schema/range/category checks + report
│   ├── preprocessing.py        # cleaning (dedupe, coerce, repair)
│   ├── feature_engineering.py  # engineered ratios + sklearn ColumnTransformer
│   ├── models.py               # model factory (5 candidate pipelines)
│   ├── evaluate.py             # full credit-risk metric suite + plots
│   ├── registry.py             # local model registry (versioning + promotion)
│   └── train.py                # orchestrates the whole pipeline end-to-end
├── api/
│   ├── main.py                  # FastAPI app (/predict, /health, etc.)
│   ├── schemas.py                # Pydantic request/response models
│   ├── model_loader.py           # loads production model, applies feature eng.
│   └── db.py                     # Postgres/SQLite prediction logging
├── tests/                        # 27 pytest tests (pipeline + API)
├── .github/workflows/ci.yml       # lint → train → test → docker build
├── Dockerfile
├── docker-compose.yml              # postgres + mlflow + trainer + api
├── render.yaml                      # low-cost cloud deploy (Render.com)
└── requirements.txt
```

## Running it locally

```bash
pip install -r requirements.txt

# 1. Generate the dataset
python data/generate_data.py

# 2. Run the full pipeline: validate -> preprocess -> engineer features
#    -> train 5 models -> evaluate -> log to MLflow -> register + promote champion
python -m src.train

# 3. Serve the champion model
uvicorn api.main:app --reload --port 8000
```

Then:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "income": 65000, "age": 32, "loan_amount": 25000, "employment_years": 6,
  "credit_score": 720, "existing_debt": 8000, "num_credit_lines": 5,
  "num_delinquencies_2yr": 0, "loan_term_months": 60, "interest_rate": 9.5,
  "home_ownership": "MORTGAGE", "loan_purpose": "home_improvement", "employment_type": "salaried"
}'
# -> {"default_probability":0.018,"prediction":"low_risk","model_version":"v1","model_name":"logistic_regression","threshold":0.72}
```

Interactive API docs (Swagger UI) at `http://localhost:8000/docs`.

Inspect experiment runs:
```bash
mlflow ui --backend-store-uri file:./mlruns   # http://localhost:5000
```

## Running the tests

```bash
python -m src.train   # tests need a registered model first
pytest tests/ -v
```
27 tests covering: data validation (schema/range/category failures),
preprocessing (dedup, out-of-range repair), feature engineering
(no div-by-zero, imputation), evaluation metrics (perfect/random separation
sanity checks), and full API integration (valid predictions, 422s on bad
input, batch predictions, admin reload).

## Running the full stack with Docker

```bash
docker compose up -d postgres mlflow
docker compose run --rm trainer     # trains models into a shared volume
docker compose up -d api            # serves the trained champion
```
API on `http://localhost:8000`, MLflow UI on `http://localhost:5000`,
Postgres on `localhost:5432` (predictions logged to the `prediction_logs`
table).

## API reference

| Method | Path            | Description                                   |
|--------|-----------------|------------------------------------------------|
| GET    | `/health`       | Liveness probe, current model version          |
| GET    | `/model/info`   | Metadata + held-out metrics for the served model |
| POST   | `/predict`      | Single application → probability + risk bucket |
| POST   | `/predict/batch`| List of applications → list of predictions      |
| POST   | `/admin/reload` | Hot-reload the current production model (after a retrain + promote), no restart needed |

Risk buckets: `low_risk` (<10%), `medium_risk` (10–30%), `high_risk` (≥30%),
configurable in `src/config.py`.

## Deploying

The Docker image is portable to any container host. Two options included:

- **`docker-compose.yml`** — a VM/EC2 instance with Docker installed can run
  the whole stack (`docker compose up -d`) for near-zero cost.
- **`render.yaml`** — Render.com Blueprint: provisions a managed Postgres +
  a web service from the Dockerfile on their free/low-cost tier. The same
  file structure works with minor edits on AWS App Runner, Azure Container
  Apps, or GCP Cloud Run — the image doesn't care which cloud runs it.

Because free-tier web services don't guarantee persistent disk across
deploys, bake a trained model into the image for a genuinely stateless
deploy: run `python -m src.train` during the Docker build (or as a one-off
Render "Job"/AWS ECS task before first deploy) so `models/registry.json`
ships inside the image rather than depending on a volume.

## Design decisions worth calling out

- **PR-AUC over accuracy/ROC-AUC for champion selection** — see Results.
- **Per-model threshold tuned on a validation set, evaluated on a separate
  test set** — avoids picking a threshold that flatters the test metrics.
- **`class_weight="balanced"` / `scale_pos_weight`** on every model instead
  of resampling (SMOTE etc.) — keeps the training distribution honest and
  avoids synthetic-minority artifacts leaking into a risk model.
- **Feature engineering lives in one module** (`feature_engineering.py`)
  imported by both `src/train.py` and `api/model_loader.py` — this is the
  single most common source of train/serve skew in real ML systems (a
  transformation that differs between training and serving), so it's
  structurally impossible to drift here.
- **Local JSON model registry** (`src/registry.py`) instead of requiring a
  running MLflow Model Registry server just to serve predictions — MLflow
  still logs every run/artifact for experiment tracking, but the API can
  start cold with nothing but a file on disk.
- **Pydantic validation at the API boundary is separate from
  `data_validation.py`** used in training — the former is about rejecting
  malformed HTTP requests fast (422s), the latter is about catching data
  quality problems in a batch/offline dataset before they corrupt a model.
