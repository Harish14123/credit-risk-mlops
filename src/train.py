"""
Training orchestration: ties together every stage in the architecture
diagram from "Raw Dataset" through "Model Registry".

Run:
    python -m src.train

What it does:
  1. Load raw data -> validate -> clean (data_validation, preprocessing)
  2. Engineer features (feature_engineering)
  3. Train/val/test split (stratified, since classes are imbalanced)
  4. Train every model in models.get_model_registry_specs()
  5. Log params/metrics/artifacts to MLflow for every run
  6. Evaluate each model with the full credit-risk metric suite
  7. Pick the champion model by PR-AUC (the right metric under class
     imbalance -- see evaluate.py docstring) evaluated on the held-out test set
  8. Register every trained model, promote the champion to production
  9. Write a model comparison report to artifacts/model_comparison.json
"""
from __future__ import annotations

import json
import logging
import os
import time

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACTS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
)
from src.data_validation import validate_raw_data
from src.evaluate import (
    evaluate_predictions,
    find_best_threshold_for_f1,
    plot_calibration_curve,
    plot_confusion_matrix,
    save_metrics_json,
)
from src.feature_engineering import add_engineered_features
from src.models import get_model_registry_specs
from src.preprocessing import clean_raw_data, split_features_target
from src.registry import register_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_and_prepare_data():
    df = pd.read_csv(RAW_DATA_PATH)

    report = validate_raw_data(df)
    logger.info(report.summary())
    if not report.is_valid:
        raise RuntimeError(f"Raw data failed validation:\n{report.summary()}")

    df = clean_raw_data(df)
    X, y = split_features_target(df)
    X = add_engineered_features(X)
    return X, y


def make_splits(X, y):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_ratio, stratify=y_train_full,
        random_state=RANDOM_STATE,
    )
    logger.info(
        f"Split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)} "
        f"(default rate train={y_train.mean():.3f}, val={y_val.mean():.3f}, test={y_test.mean():.3f})"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X, y = load_and_prepare_data()
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    specs = get_model_registry_specs(scale_pos_weight=scale_pos_weight)
    logger.info(f"Training {len(specs)} models: {list(specs.keys())}")

    comparison = []
    for model_name, pipeline in specs.items():
        with mlflow.start_run(run_name=model_name) as run:
            t0 = time.time()
            pipeline.fit(X_train, y_train)
            train_seconds = time.time() - t0

            val_proba = pipeline.predict_proba(X_val)[:, 1]
            best_threshold = find_best_threshold_for_f1(y_val.values, val_proba)

            test_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate_predictions(model_name, y_test.values, test_proba, threshold=best_threshold)

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("threshold", best_threshold)
            mlflow.log_metric("train_seconds", train_seconds)
            for k, v in metrics.to_dict().items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            mlflow.sklearn.log_model(pipeline, artifact_path="model", serialization_format="cloudpickle")

            cm_path = ARTIFACTS_DIR / f"{model_name}_confusion_matrix.png"
            cal_path = ARTIFACTS_DIR / f"{model_name}_calibration.png"
            plot_confusion_matrix(metrics.confusion_matrix, model_name, cm_path)
            plot_calibration_curve(y_test.values, test_proba, model_name, cal_path)
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(cal_path))

            metrics_path = ARTIFACTS_DIR / f"{model_name}_metrics.json"
            save_metrics_json(metrics, metrics_path)
            mlflow.log_artifact(str(metrics_path))

            version = register_model(
                pipeline, model_name, metrics.to_dict(),
                mlflow_run_id=run.info.run_id, threshold=best_threshold,
            )

            logger.info(
                f"[{model_name}] registered as {version} | ROC-AUC={metrics.roc_auc:.4f} "
                f"PR-AUC={metrics.pr_auc:.4f} F1={metrics.f1:.4f} "
                f"Precision={metrics.precision:.4f} Recall={metrics.recall:.4f} "
                f"Brier={metrics.brier_score:.4f} (train_seconds={train_seconds:.1f}s)"
            )

            comparison.append({
                "version": version, "model_name": model_name,
                "mlflow_run_id": run.info.run_id, "train_seconds": train_seconds,
                **metrics.to_dict(),
            })

    # Champion selection: best PR-AUC on held-out test set. PR-AUC is the
    # right primary metric here because defaults are the rare, costly class
    # -- ROC-AUC can look deceptively good under imbalance.
    comparison_df = pd.DataFrame(comparison).sort_values("pr_auc", ascending=False)
    champion = comparison_df.iloc[0]
    from src.registry import promote
    promote(champion["version"])

    logger.info(f"\n{comparison_df[['model_name','version','roc_auc','pr_auc','f1','precision','recall']].to_string(index=False)}")
    logger.info(f"Champion model promoted to production: {champion['model_name']} ({champion['version']})")

    report_path = ARTIFACTS_DIR / "model_comparison.json"
    with open(report_path, "w") as f:
        json.dump(comparison_df.to_dict(orient="records"), f, indent=2, default=str)
    logger.info(f"Model comparison report written to {report_path}")


if __name__ == "__main__":
    main()
