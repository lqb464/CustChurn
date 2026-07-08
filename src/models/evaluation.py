"""Thước đo kỹ thuật và thước đo phục vụ quyết định nghiệp vụ."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: object, y_pred: object) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    error = prediction - actual
    denominator = np.abs(actual).sum()
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(mean_squared_error(actual, prediction) ** 0.5),
        "r2": float(r2_score(actual, prediction)),
        "wape": float(np.abs(error).sum() / denominator) if denominator else np.nan,
        "within_10pct": float(np.mean(np.abs(error) <= 0.10 * np.maximum(np.abs(actual), 1))),
        "within_20pct": float(np.mean(np.abs(error) <= 0.20 * np.maximum(np.abs(actual), 1))),
        "bias": float(np.mean(error)),
        "over_prediction_rate": float(np.mean(error > 0)),
    }


def metric_table(y_true: object, predictions: dict[str, object]) -> pd.DataFrame:
    rows = [{"model": name, **regression_metrics(y_true, values)} for name, values in predictions.items()]
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def policy_baseline(request_amount: object, ratio: float = 0.70) -> np.ndarray:
    return np.asarray(request_amount, dtype=float) * ratio
