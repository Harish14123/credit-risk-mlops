import numpy as np

from src.evaluate import evaluate_predictions, find_best_threshold_for_f1


def test_evaluate_predictions_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.01, 0.02, 0.03, 0.9, 0.95, 0.99])
    metrics = evaluate_predictions("test_model", y_true, y_proba, threshold=0.5)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.confusion_matrix == [[3, 0], [0, 3]]


def test_evaluate_predictions_random_guess_roc_auc_near_half():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=2000)
    y_proba = rng.uniform(0, 1, size=2000)
    metrics = evaluate_predictions("random_model", y_true, y_proba)
    assert 0.4 < metrics.roc_auc < 0.6


def test_find_best_threshold_returns_valid_range():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=500)
    y_proba = np.clip(y_true * 0.6 + rng.normal(0, 0.2, size=500), 0, 1)
    threshold = find_best_threshold_for_f1(y_true, y_proba)
    assert 0.0 < threshold < 1.0


def test_metrics_serializable_to_dict():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.2, 0.8, 0.3, 0.7])
    metrics = evaluate_predictions("m", y_true, y_proba)
    d = metrics.to_dict()
    assert "precision" in d and "roc_auc" in d and "confusion_matrix" in d
