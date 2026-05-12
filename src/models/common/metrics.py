from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _as_1d_float_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    if method == "pearson":
        return float(np.corrcoef(y_true, y_pred)[0, 1])

    # Spearman: rank then Pearson
    y_true_rank = np.argsort(np.argsort(y_true))
    y_pred_rank = np.argsort(np.argsort(y_pred))
    if np.std(y_true_rank) == 0 or np.std(y_pred_rank) == 0:
        return float("nan")
    return float(np.corrcoef(y_true_rank, y_pred_rank)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true_arr = _as_1d_float_array(y_true)
    y_pred_arr = _as_1d_float_array(y_pred)
    bias = float(np.mean(y_pred_arr - y_true_arr))

    out = {
        "rmse": rmse(y_true_arr, y_pred_arr),
        "mae": mae(y_true_arr, y_pred_arr),
        "r2": r2(y_true_arr, y_pred_arr),
        "pearson_r": _safe_corr(y_true_arr, y_pred_arr, "pearson"),
        "spearman_r": _safe_corr(y_true_arr, y_pred_arr, "spearman"),
        "bias": bias,
    }
    return out


def mean_regression_metrics(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "pearson_r": float("nan"),
            "spearman_r": float("nan"),
            "bias": float("nan"),
        }

    keys = ["rmse", "mae", "r2", "pearson_r", "spearman_r", "bias"]
    out = {}
    for key in keys:
        vals = [pd[key] for pd in metrics_list if key in pd]
        if not vals:
            out[key] = float("nan")
            continue
        arr = np.asarray(vals, dtype=float)
        out[key] = float(np.nanmean(arr)) if not np.isnan(arr).all() else float("nan")
    return out
