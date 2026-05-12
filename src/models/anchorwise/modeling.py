from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
except Exception:
    LGBMRegressor = None
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except Exception:
    XGBRegressor = None
    HAS_XGBOOST = False


class SafeSelectKBest(SelectKBest):
    """A SelectKBest variant that falls back to all features when k is too large."""

    def fit(self, X, y):
        if isinstance(self.k, int) and self.k > X.shape[1]:
            self.k = "all"
        return super().fit(X, y)


class SafeVarianceThreshold(BaseEstimator, TransformerMixin):
    """A VarianceThreshold-like transformer that keeps all features when all are low-variance."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)
        self.support_mask_ = None

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("SafeVarianceThreshold expects 2D input.")
        with np.errstate(invalid="ignore"):
            variances = np.nanvar(arr, axis=0)
        mask = variances > self.threshold
        if not np.any(mask):
            # Keep all columns when no feature passes threshold; avoids pipeline hard-fail.
            mask = np.ones(arr.shape[1], dtype=bool)
        self.support_mask_ = mask
        return self

    def transform(self, X):
        if self.support_mask_ is None:
            raise RuntimeError("SafeVarianceThreshold is not fitted.")
        arr = np.asarray(X)
        return arr[:, self.support_mask_]

    def get_support(self, indices: bool = False):
        if self.support_mask_ is None:
            raise RuntimeError("SafeVarianceThreshold is not fitted.")
        if indices:
            return np.where(self.support_mask_)[0]
        return self.support_mask_


class SafePLSRegression(PLSRegression):
    """Clamp n_components to a feasible value at fit time."""

    def fit(self, X, y):
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("SafePLSRegression expects 2D input.")
        max_components = max(1, min(arr.shape[0] - 1 if arr.shape[0] > 1 else 1, arr.shape[1]))
        self.n_components = int(max(1, min(int(self.n_components), max_components)))
        return super().fit(X, y)


def suggest_params(trial, predictor: str, model_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    model_cfg = model_cfg or {}
    if predictor == "ridge":
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True),
        }

    if predictor == "elasticnet":
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
        }

    if predictor == "lasso":
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        }

    if predictor == "pls":
        return {
            "n_components": trial.suggest_int("n_components", 2, 32),
            "scale": trial.suggest_categorical("scale", [True, False]),
        }

    if predictor == "extratrees":
        max_depth_choice = trial.suggest_categorical("max_depth", [None, 6, 10, 16, 24])
        return {
            "n_estimators": trial.suggest_int("n_estimators", 20, 60, step=10),
            "max_depth": max_depth_choice,
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
        }

    if predictor == "random_forest":
        max_depth_choice = trial.suggest_categorical("max_depth", [None, 6, 10, 16, 24])
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=25),
            "max_depth": max_depth_choice,
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
        }

    if predictor == "lightgbm":
        backend = str(model_cfg.get("lightgbm_backend", "sklearn_gbrt")).lower()
        if backend != "native":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 20, 80, step=10),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }

        return {
            "n_estimators": trial.suggest_int("n_estimators", 20, 80, step=10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 8]),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    if predictor == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 30, 120, step=15),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    raise ValueError(f"未知 predictor: {predictor}")


def _build_estimator(predictor: str, params: Dict[str, Any], seed: int, model_cfg: Dict[str, Any] | None = None):
    model_cfg = model_cfg or {}
    if predictor == "ridge":
        return Ridge(alpha=float(params["alpha"]), random_state=seed)

    if predictor == "elasticnet":
        return ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=5000,
            random_state=seed,
        )

    if predictor == "lasso":
        return Lasso(
            alpha=float(params["alpha"]),
            max_iter=5000,
            random_state=seed,
        )

    if predictor == "pls":
        return SafePLSRegression(
            n_components=int(params["n_components"]),
            scale=bool(params.get("scale", False)),
        )

    if predictor == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=seed,
            n_jobs=1,
        )

    if predictor == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=seed,
            n_jobs=1,
        )

    if predictor == "lightgbm":
        backend = str(model_cfg.get("lightgbm_backend", "sklearn_gbrt")).lower()
        if backend == "native":
            if not HAS_LIGHTGBM:
                raise RuntimeError("当前环境不可用 lightgbm 原生后端。")
            return LGBMRegressor(
                objective="regression",
                n_estimators=int(params["n_estimators"]),
                learning_rate=float(params["learning_rate"]),
                num_leaves=int(params["num_leaves"]),
                max_depth=int(params["max_depth"]),
                min_child_samples=int(params["min_child_samples"]),
                subsample=float(params["subsample"]),
                colsample_bytree=float(params["colsample_bytree"]),
                reg_alpha=float(params["reg_alpha"]),
                reg_lambda=float(params["reg_lambda"]),
                random_state=seed,
                n_jobs=1,
                verbosity=-1,
            )

        # fallback: sklearn GBDT，避免当前环境中 lightgbm 的 OMP SHM 崩溃
        return GradientBoostingRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            subsample=float(params["subsample"]),
            random_state=seed,
        )

    if predictor == "xgboost":
        if not HAS_XGBOOST:
            raise RuntimeError("当前环境不可用 xgboost。")
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )

    raise ValueError(f"未知 predictor: {predictor}")


def _selector_k_for_predictor(predictor: str, n_features: int, preprocess_cfg: Dict[str, Any]) -> int | None:
    selector_cfg = preprocess_cfg.get("select_k_best", {})
    if not selector_cfg.get("enabled", True):
        return None

    max_features_map = selector_cfg.get("max_features_by_model", {})
    k_limit = max_features_map.get(predictor)
    if k_limit is None:
        k_limit = max_features_map.get("default")
    if k_limit is None:
        return None

    k = int(min(max(1, int(k_limit)), int(n_features)))
    return k


def build_pipeline(
    predictor: str,
    params: Dict[str, Any],
    n_features: int,
    preprocess_cfg: Dict[str, Any],
    seed: int,
    model_cfg: Dict[str, Any] | None = None,
) -> Pipeline:
    steps = []

    imputer_cfg = preprocess_cfg.get("imputer", {})
    strategy = imputer_cfg.get("strategy", "median")
    add_indicator = bool(imputer_cfg.get("add_indicator", False))
    steps.append(("imputer", SimpleImputer(strategy=strategy, add_indicator=add_indicator)))

    var_cfg = preprocess_cfg.get("variance_threshold", {})
    if var_cfg.get("enabled", True):
        threshold = float(var_cfg.get("threshold", 0.0))
        steps.append(("variance", SafeVarianceThreshold(threshold=threshold)))

    k = _selector_k_for_predictor(predictor, n_features=n_features, preprocess_cfg=preprocess_cfg)
    if k is not None:
        steps.append(("select", SafeSelectKBest(score_func=f_regression, k=k)))

    if predictor in {"ridge", "elasticnet", "lasso", "pls"}:
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))

    estimator = _build_estimator(predictor, params, seed, model_cfg=model_cfg)
    steps.append(("model", estimator))

    return Pipeline(steps=steps)


def sanitize_feature_matrix(x):
    # 避免 inf 进入模型
    x = x.replace([np.inf, -np.inf], np.nan)
    return x
