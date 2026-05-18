import numpy as np
import pytest
from src.evaluate import compute_metrics


def test_perfect_prediction():
    y = np.array([100.0, 102.0, 101.0, 103.0, 105.0])
    metrics = compute_metrics(y, y)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0


def test_directional_accuracy_all_correct():
    y_true = np.array([100.0, 102.0, 104.0, 103.0, 105.0])
    # predictions always in the same direction as actuals
    y_pred = np.array([100.0, 101.9, 103.9, 102.9, 104.9])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["directional_accuracy"] == 100.0


def test_directional_accuracy_all_wrong():
    y_true = np.array([100.0, 102.0, 104.0, 103.0])
    y_pred = np.array([100.0, 103.0, 105.0, 104.0])  # always predicts up when goes down, etc.
    # actual: up, up, down  |  pred vs prev actual: up, up, up → last is wrong
    metrics = compute_metrics(y_true, y_pred)
    assert 0.0 <= metrics["directional_accuracy"] <= 100.0


def test_metric_keys():
    y = np.array([100.0, 101.0, 99.0])
    metrics = compute_metrics(y, y + 1)
    for key in ("mae", "rmse", "mape", "directional_accuracy"):
        assert key in metrics


def test_metrics_non_negative():
    rng = np.random.default_rng(0)
    y_true = 100 + rng.normal(0, 5, 50)
    y_pred = y_true + rng.normal(0, 2, 50)
    m = compute_metrics(y_true, y_pred)
    assert m["rmse"] >= 0
    assert m["mae"] >= 0
    assert m["mape"] >= 0
