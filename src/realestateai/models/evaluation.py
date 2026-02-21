import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def calculate_regression_metrics(preds: np.ndarray, y: pd.Series) -> dict:
    preds = np.asarray(preds)
    y_true = np.asarray(y)

    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    return {
        "mae": float(mean_absolute_error(y_true, preds)),
        "mape": float(mean_absolute_percentage_error(y_true, preds)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, preds)),
        "median_ae": float(median_absolute_error(y_true, preds)),
    }
