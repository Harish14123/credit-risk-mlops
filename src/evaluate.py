"""
Model Evaluation stage.

Credit risk is a highly imbalanced classification problem, so accuracy is
close to useless (a model predicting "no default" for everyone scores ~83%
accuracy on this dataset while being completely unusable). We report the
full set of metrics that matter for a lending use case, plus calibration,
since the probability itself (not just the class) is used downstream for
pricing and risk-based decisions.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvalMetrics:
    model_name: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    confusion_matrix: list  # [[tn, fp], [fn, tp]]
    threshold: float

    def to_dict(self):
        return asdict(self)


def evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> EvalMetrics:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return EvalMetrics(
        model_name=model_name,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        brier_score=float(brier_score_loss(y_true, y_proba)),
        confusion_matrix=cm,
        threshold=threshold,
    )


def find_best_threshold_for_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Scan thresholds and return the one maximizing F1 (useful default
    operating point; production would instead pick a threshold from a
    cost-based analysis of false negatives (missed defaults) vs false
    positives (rejected good customers))."""
    best_thresh, best_f1 = 0.5, -1
    for t in np.linspace(0.01, 0.99, 99):
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return float(best_thresh)


def plot_calibration_curve(y_true, y_proba, model_name: str, out_path):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, marker="o", label=model_name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title(f"Calibration curve - {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm, model_name: str, out_path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No Default", "Default"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No Default", "Default"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_metrics_json(metrics: EvalMetrics, out_path):
    with open(out_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
