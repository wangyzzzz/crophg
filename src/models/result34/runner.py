from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import warnings
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".tmp_mpl").resolve()))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState
from sklearn.exceptions import ConvergenceWarning

from models.anchorwise.data_loader import MultiTargetDataBundle, load_multitarget_model_inputs
from models.anchorwise.modeling import HAS_LIGHTGBM, build_pipeline, sanitize_feature_matrix, suggest_params
from models.common.io_utils import ensure_dir, get_git_commit_hash, now_iso, read_yaml, write_json, write_yaml
from models.common.metrics import mean_regression_metrics, regression_metrics
from models.common.optuna_reuse import normalize_outer_groups, resolve_optuna_reuse_policy, should_search_on_outer_fold
from models.result31.runner import (
    _append_progress,
    _build_outer_payload,
    _prepare_split_groups,
    _resolve_active_predictors,
    _resolve_output_dir,
    _resolve_scenarios_cfg,
    _resolve_split_groups_for_scenario,
    _resolve_timeline_dirs,
    _set_global_seed,
)
from models.result34.feature_engineering import (
    AnchorLocalConfig,
    PhaseStateConfig,
    attach_anchor_bin_prefix_to_full_h_spec,
    build_h_anchor_local_features,
    build_h_anchor_vi_features,
    build_h_full_features,
    build_h_phase_state_features,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Skipping features without any observed values")
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt", category=RuntimeWarning)


VALID_INPUT_VARIANTS = [
    "G",
    "H_FULL",
    "G+H_FULL",
    "H_ANCHOR_LOCAL",
    "G+H_ANCHOR_LOCAL",
    "H_ANCHOR_VI",
    "G+H_ANCHOR_VI",
    "H_ANCHOR_AUTO",
    "G+H_ANCHOR_AUTO",
    "G+FULLH",
]
ANCHOR_LOCAL_VARIANTS = {"H_ANCHOR_LOCAL", "G+H_ANCHOR_LOCAL"}
ANCHOR_VI_VARIANTS = {"H_ANCHOR_VI", "G+H_ANCHOR_VI"}
AUTO_WINDOW_VARIANTS = {"H_ANCHOR_AUTO", "G+H_ANCHOR_AUTO"}
REDUCED_H_VARIANTS = ANCHOR_LOCAL_VARIANTS | ANCHOR_VI_VARIANTS | AUTO_WINDOW_VARIANTS
G_ONLY_VARIANTS = {"G"}
G_H_FUSION_VARIANTS = {"G+H_FULL", "G+H_ANCHOR_LOCAL", "G+H_ANCHOR_VI", "G+H_ANCHOR_AUTO", "G+FULLH"}
HAS_RSCRIPT = shutil.which("Rscript") is not None
_GRM_ASSET_CACHE: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}


def _safe_std(s: pd.Series) -> float:
    return float(np.std(pd.to_numeric(s, errors="coerce"), ddof=0))


def _nan_metric_dict() -> dict[str, float]:
    return {
        "rmse": float("nan"),
        "mae": float("nan"),
        "r2": float("nan"),
        "pearson_r": float("nan"),
        "spearman_r": float("nan"),
        "bias": float("nan"),
    }


def _resolve_input_variants(exp_cfg: dict) -> list[str]:
    raw = exp_cfg.get("input_variants", VALID_INPUT_VARIANTS)
    variants: list[str] = []
    for item in raw:
        token = str(item).strip().upper()
        if token == "H_PHASE":
            token = "H_FULL"
        if token == "G+H_PHASE":
            token = "G+H_FULL"
        if token not in VALID_INPUT_VARIANTS:
            raise ValueError(f"不支持的 input_variant: {item}")
        if token not in variants:
            variants.append(token)
    if not variants:
        raise RuntimeError("没有可执行的 input_variant。")
    return variants


def _normalize_anchor_order_value(value: Any) -> int | None:
    if value in (None, "", "none", "None", "null", "NULL"):
        return None
    return int(value)


def _resolve_growth_prefix_cfg(exp_cfg: dict[str, Any]) -> dict[str, Any]:
    raw = exp_cfg.get("growth_prefix", exp_cfg.get("prefix_mode", {}))
    if raw in (None, False):
        raw = {}
    if raw is True:
        raw = {"enabled": True}
    if not isinstance(raw, dict):
        raise TypeError("experiment.growth_prefix 必须是字典或布尔值。")

    n_anchor_bins = int(exp_cfg.get("n_anchor_bins", exp_cfg.get("anchor_local", {}).get("n_anchor_bins", 20)))
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return {"enabled": False, "anchor_orders": [None], "n_anchor_bins": n_anchor_bins}

    raw_orders = raw.get("anchor_orders")
    if raw_orders in (None, "", []):
        orders = list(range(1, n_anchor_bins + 1))
    else:
        orders = sorted({int(x) for x in raw_orders})
    orders = [x for x in orders if 1 <= int(x) <= n_anchor_bins]
    if not orders:
        raise ValueError("experiment.growth_prefix.anchor_orders 为空或超出 n_anchor_bins 范围。")
    return {"enabled": True, "anchor_orders": orders, "n_anchor_bins": n_anchor_bins}


def _normalize_task_spec(spec: dict) -> tuple[str, str, str, str, str, int | None]:
    required = ["scenario", "timeline", "target", "input_variant", "predictor"]
    missing = [k for k in required if k not in spec]
    if missing:
        raise KeyError(f"task_spec 缺少字段: {missing}")
    input_variant = str(spec["input_variant"]).upper()
    if input_variant == "H_PHASE":
        input_variant = "H_FULL"
    if input_variant == "G+H_PHASE":
        input_variant = "G+H_FULL"
    return (
        str(spec["scenario"]),
        str(spec["timeline"]),
        str(spec["target"]),
        input_variant,
        str(spec["predictor"]).lower(),
        _normalize_anchor_order_value(spec.get("anchor_order")),
    )


def _resolve_task_filter_set(exp_cfg: dict) -> set[tuple[str, str, str, str, str, int | None]]:
    raw = None
    if isinstance(exp_cfg.get("task_filters"), dict):
        raw = exp_cfg.get("task_filters", {}).get("task_specs")
    if raw is None:
        raw = exp_cfg.get("task_specs")
    if raw in (None, "", []):
        return set()
    if not isinstance(raw, list):
        raise TypeError("experiment.task_filters.task_specs 必须是列表。")
    return {_normalize_task_spec(spec) for spec in raw}


def _resolve_anchor_window_radius_candidates(exp_cfg: dict[str, Any]) -> list[int]:
    raw = exp_cfg.get("anchor_window_search", {}).get("radius_candidates", [0])
    if raw in (None, "", []):
        return [0]
    if not isinstance(raw, list):
        raw = [raw]
    values = sorted({max(0, int(x)) for x in raw})
    return values or [0]


def enumerate_result34_tasks_from_config(cfg: dict) -> list[dict[str, Any]]:
    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})
    timeline_dirs = _resolve_timeline_dirs(data_cfg)
    scenarios_cfg = _resolve_scenarios_cfg(data_cfg)
    targets = [str(x) for x in exp_cfg.get("targets", ["ActualYD", "CM", "LM", "PHM", "Spike", "TKW"])]
    active_predictors = _resolve_active_predictors(exp_cfg)
    input_variants = _resolve_input_variants(exp_cfg)
    task_filter_set = _resolve_task_filter_set(exp_cfg)
    prefix_cfg = _resolve_growth_prefix_cfg(exp_cfg)
    prefix_orders = prefix_cfg["anchor_orders"]

    tasks: list[dict[str, Any]] = []
    for scenario_name in scenarios_cfg.keys():
        for timeline_name in timeline_dirs.keys():
            for target_col in targets:
                for input_variant in input_variants:
                    for predictor in active_predictors:
                        for anchor_order in prefix_orders:
                            task_key = (scenario_name, timeline_name, target_col, input_variant, predictor, anchor_order)
                            if task_filter_set and task_key not in task_filter_set:
                                continue
                            task = {
                                "scenario": scenario_name,
                                "timeline": timeline_name,
                                "target": target_col,
                                "input_variant": input_variant,
                                "predictor": predictor,
                            }
                            if anchor_order is not None:
                                task["anchor_order"] = int(anchor_order)
                            tasks.append(task)
    return tasks


def _build_input_variant_matrix(
    bundle: MultiTargetDataBundle,
    *,
    x_phase: pd.DataFrame,
    x_anchor_local: pd.DataFrame,
    x_anchor_vi: pd.DataFrame,
    input_variant: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    variant = str(input_variant).strip().upper()
    if variant == "H_PHASE":
        variant = "H_FULL"
    if variant == "G+H_PHASE":
        variant = "G+H_FULL"
    if variant == "G":
        x = sanitize_feature_matrix(bundle.x_genotype.copy())
        return x, {
            "input_variant": variant,
            "n_features_total": int(x.shape[1]),
            "n_features_h_full": 0,
            "n_features_h_anchor_local": 0,
            "n_features_h_phase": 0,
            "n_features_g": int(x.shape[1]),
        }
    if variant == "H_FULL":
        x = sanitize_feature_matrix(x_phase.copy())
        return x, {
            "input_variant": variant,
            "n_features_total": int(x.shape[1]),
            "n_features_h_full": int(x.shape[1]),
            "n_features_h_anchor_local": 0,
            "n_features_h_phase": int(x.shape[1]),
            "n_features_g": 0,
        }
    if variant == "G+H_FULL":
        x = pd.concat([x_phase, bundle.x_genotype], axis=1).loc[bundle.meta.index].copy()
        x = sanitize_feature_matrix(x)
        return x, {
            "input_variant": variant,
            "n_features_total": int(x.shape[1]),
            "n_features_h_full": int(x_phase.shape[1]),
            "n_features_h_anchor_local": 0,
            "n_features_h_phase": int(x_phase.shape[1]),
            "n_features_g": int(bundle.x_genotype.shape[1]),
        }
    if variant == "H_ANCHOR_LOCAL":
        x = sanitize_feature_matrix(x_anchor_local.copy())
        return x, {
            "input_variant": variant,
            "n_features_total": int(x.shape[1]),
            "n_features_h_full": 0,
            "n_features_h_anchor_local": int(x.shape[1]),
            "n_features_h_phase": 0,
            "n_features_g": 0,
        }
    if variant == "H_ANCHOR_AUTO":
        x = sanitize_feature_matrix(x_anchor_local.copy())
        return x, {
            "input_variant": variant,
            "n_features_total": int(x.shape[1]),
            "n_features_h_full": 0,
            "n_features_h_anchor_local": int(x.shape[1]),
            "n_features_h_anchor_vi": int(x.shape[1]),
            "n_features_h_anchor_auto": int(x.shape[1]),
            "n_features_h_phase": 0,
            "n_features_g": 0,
        }
    if variant == "G+H_ANCHOR_LOCAL":
        x = pd.concat([x_anchor_local, bundle.x_genotype], axis=1).loc[bundle.meta.index].copy()
        x = sanitize_feature_matrix(x)
        return x, {
            "input_variant": variant,
            "n_features_total": int(x.shape[1]),
            "n_features_h_full": 0,
            "n_features_h_anchor_local": int(x_anchor_local.shape[1]),
            "n_features_h_phase": 0,
            "n_features_g": int(bundle.x_genotype.shape[1]),
        }
    if variant == "G+H_ANCHOR_AUTO":
        x = pd.concat([x_anchor_local, bundle.x_genotype], axis=1).loc[bundle.meta.index].copy()
        x = sanitize_feature_matrix(x)
        return x, {
            "input_variant": variant,
            "n_features_total": int(x.shape[1]),
            "n_features_h_full": 0,
            "n_features_h_anchor_local": int(x_anchor_local.shape[1]),
            "n_features_h_anchor_vi": int(x_anchor_local.shape[1]),
            "n_features_h_anchor_auto": int(x_anchor_local.shape[1]),
            "n_features_h_phase": 0,
            "n_features_g": int(bundle.x_genotype.shape[1]),
        }
    if variant == "H_ANCHOR_VI":
        x = sanitize_feature_matrix(x_anchor_vi.copy())
        return x, {
            "input_variant": variant,
            "n_features_total": int(x.shape[1]),
            "n_features_h_full": 0,
            "n_features_h_anchor_local": 0,
            "n_features_h_anchor_vi": int(x.shape[1]),
            "n_features_h_phase": 0,
            "n_features_g": 0,
        }
    if variant == "G+H_ANCHOR_VI":
        x = pd.concat([x_anchor_vi, bundle.x_genotype], axis=1).loc[bundle.meta.index].copy()
        x = sanitize_feature_matrix(x)
        return x, {
            "input_variant": variant,
            "n_features_total": int(x.shape[1]),
            "n_features_h_full": 0,
            "n_features_h_anchor_local": 0,
            "n_features_h_anchor_vi": int(x_anchor_vi.shape[1]),
            "n_features_h_phase": 0,
            "n_features_g": int(bundle.x_genotype.shape[1]),
        }
    if variant == "G+FULLH":
        x = pd.concat([x_phase, bundle.x_genotype], axis=1).loc[bundle.meta.index].copy()
        x = sanitize_feature_matrix(x)
        return x, {
            "input_variant": variant,
            "n_features_total": int(x.shape[1]),
            "n_features_h_full": int(x_phase.shape[1]),
            "n_features_h_anchor_local": 0,
            "n_features_h_phase": 0,
            "n_features_g": int(bundle.x_genotype.shape[1]),
        }
    raise ValueError(f"不支持的 input_variant: {input_variant}")


def _anchor_prefix_metadata(anchor_order: int | None, anchor_feature_spec_df: pd.DataFrame) -> dict[str, Any]:
    if anchor_order is None:
        return {
            "anchor_order": np.nan,
            "anchor_idx": np.nan,
            "anchor_tb": np.nan,
            "anchor_phase": "",
            "anchor_token": "",
        }
    anchor_order_int = int(anchor_order)
    anchor_idx = anchor_order_int - 1
    meta = {
        "anchor_order": anchor_order_int,
        "anchor_idx": anchor_idx,
        "anchor_tb": np.nan,
        "anchor_phase": "",
        "anchor_token": "",
    }
    if anchor_feature_spec_df.empty or "anchor_idx" not in anchor_feature_spec_df.columns:
        return meta
    sub = anchor_feature_spec_df.loc[pd.to_numeric(anchor_feature_spec_df["anchor_idx"], errors="coerce") == anchor_idx].copy()
    if sub.empty:
        return meta
    for col in ["anchor_tb", "anchor_phase", "anchor_token"]:
        if col in sub.columns:
            vals = sub[col].dropna().astype(str).unique().tolist()
            if vals:
                meta[col] = vals[0]
    if "anchor_tb" in sub.columns:
        vals_num = pd.to_numeric(sub["anchor_tb"], errors="coerce").dropna().unique().tolist()
        if vals_num:
            meta["anchor_tb"] = int(vals_num[0])
    return meta


def _filter_feature_spec_to_columns(feature_spec_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if feature_spec_df.empty or "feature" not in feature_spec_df.columns:
        return feature_spec_df.copy()
    col_set = set(str(x) for x in columns)
    return feature_spec_df.loc[feature_spec_df["feature"].astype(str).isin(col_set)].copy()


def _limit_variant_to_anchor_prefix(
    *,
    x: pd.DataFrame,
    info: dict[str, Any],
    feature_spec_df: pd.DataFrame,
    input_variant: str,
    anchor_order: int | None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if anchor_order is None or feature_spec_df.empty or "feature" not in feature_spec_df.columns:
        return x, dict(info), feature_spec_df.copy()

    if "anchor_idx" not in feature_spec_df.columns:
        raise ValueError(f"{input_variant} 缺少 anchor_idx，无法执行 growth prefix 截断。")

    variant = str(input_variant).strip().upper()
    anchor_idx_limit = int(anchor_order) - 1
    anchor_idx_ok = pd.to_numeric(feature_spec_df["anchor_idx"], errors="coerce") <= anchor_idx_limit
    if "source_anchor_idx" in feature_spec_df.columns:
        source_anchor_idx_ok = (
            pd.to_numeric(feature_spec_df["source_anchor_idx"], errors="coerce") <= anchor_idx_limit
        )
        h_spec = feature_spec_df.loc[anchor_idx_ok & source_anchor_idx_ok].copy()
    else:
        h_spec = feature_spec_df.loc[anchor_idx_ok].copy()
    if variant in {"H_FULL", "G+H_FULL", "G+FULLH"}:
        missing_anchor = int(pd.to_numeric(feature_spec_df["anchor_idx"], errors="coerce").isna().sum())
        if missing_anchor:
            raise ValueError(
                f"{input_variant} 有 {missing_anchor} 个 H_FULL 特征缺少 anchor_idx；"
                "请确认 full-H prefix 使用连续 anchor-bin 映射，而不是 midpoint no-bin 映射。"
            )
    h_cols = [str(col) for col in h_spec["feature"].astype(str).tolist() if str(col) in x.columns]
    h_col_set = set(str(col) for col in feature_spec_df["feature"].astype(str).tolist())
    g_cols = [str(col) for col in x.columns if str(col) not in h_col_set]

    if variant == "G":
        keep_cols = list(x.columns)
    elif variant in G_H_FUSION_VARIANTS:
        keep_cols = [col for col in x.columns if col in set(g_cols) | set(h_cols)]
    else:
        keep_cols = [col for col in x.columns if col in set(h_cols)]

    x_prefix = x.loc[:, keep_cols].copy()
    info_prefix = dict(info)
    n_h = len(h_cols)
    n_g = len([col for col in keep_cols if col in set(g_cols)])
    info_prefix["n_features_total"] = int(x_prefix.shape[1])
    info_prefix["n_features_g"] = int(n_g)
    if variant in {"H_FULL", "G+H_FULL", "G+FULLH"}:
        info_prefix["n_features_h_full"] = int(n_h)
        info_prefix["n_features_h_phase"] = int(n_h)
    if variant in ANCHOR_LOCAL_VARIANTS or variant in ANCHOR_VI_VARIANTS or variant in AUTO_WINDOW_VARIANTS:
        info_prefix["n_features_h_anchor_local"] = int(n_h) if variant in ANCHOR_LOCAL_VARIANTS or variant in AUTO_WINDOW_VARIANTS else 0
        info_prefix["n_features_h_anchor_vi"] = int(n_h) if variant in ANCHOR_VI_VARIANTS or variant in AUTO_WINDOW_VARIANTS else 0
        info_prefix["n_features_h_anchor_auto"] = int(n_h) if variant in AUTO_WINDOW_VARIANTS else int(info_prefix.get("n_features_h_anchor_auto", 0))
        info_prefix["n_features_h_phase"] = 0
        info_prefix["n_features_h_full"] = 0
    info_prefix["growth_prefix_enabled"] = True
    info_prefix["anchor_order"] = int(anchor_order)
    info_prefix["anchor_idx"] = int(anchor_idx_limit)
    return x_prefix, info_prefix, h_spec


def _build_auto_input_variant_matrix(
    bundle: MultiTargetDataBundle,
    *,
    x_h: pd.DataFrame,
    input_variant: str,
    radius: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    variant = str(input_variant).strip().upper()
    radius = max(0, int(radius))
    if variant == "H_ANCHOR_AUTO":
        x = sanitize_feature_matrix(x_h.copy())
        return x, {
            "input_variant": variant,
            "n_features_total": int(x.shape[1]),
            "n_features_h_full": 0,
            "n_features_h_anchor_local": int(x_h.shape[1]) if radius > 0 else 0,
            "n_features_h_anchor_vi": int(x_h.shape[1]) if radius == 0 else 0,
            "n_features_h_anchor_auto": int(x_h.shape[1]),
            "n_features_h_phase": 0,
            "n_features_g": 0,
            "anchor_window_radius": int(radius),
            "anchor_window_left_context": int(radius),
            "anchor_window_right_context": int(radius),
        }
    if variant == "G+H_ANCHOR_AUTO":
        x = pd.concat([x_h, bundle.x_genotype], axis=1).loc[bundle.meta.index].copy()
        x = sanitize_feature_matrix(x)
        return x, {
            "input_variant": variant,
            "n_features_total": int(x.shape[1]),
            "n_features_h_full": 0,
            "n_features_h_anchor_local": int(x_h.shape[1]) if radius > 0 else 0,
            "n_features_h_anchor_vi": int(x_h.shape[1]) if radius == 0 else 0,
            "n_features_h_anchor_auto": int(x_h.shape[1]),
            "n_features_h_phase": 0,
            "n_features_g": int(bundle.x_genotype.shape[1]),
            "anchor_window_radius": int(radius),
            "anchor_window_left_context": int(radius),
            "anchor_window_right_context": int(radius),
        }
    raise ValueError(f"不支持的 auto input_variant: {input_variant}")


def _resolve_feature_spec_df_for_variant(
    *,
    timeline_phase_info: dict[str, dict[str, Any]],
    timeline_anchor_local_info: dict[str, dict[str, Any]],
    timeline_anchor_vi_info: dict[str, dict[str, Any]],
    timeline_name: str,
    input_variant: str,
) -> pd.DataFrame:
    variant = str(input_variant).strip().upper()
    if variant in {"H_FULL", "G+H_FULL", "G+FULLH"}:
        return timeline_phase_info[timeline_name]["feature_spec_df"]
    if variant in ANCHOR_LOCAL_VARIANTS or variant in AUTO_WINDOW_VARIANTS:
        return timeline_anchor_local_info[timeline_name]["feature_spec_df"]
    if variant in ANCHOR_VI_VARIANTS:
        return timeline_anchor_vi_info[timeline_name]["feature_spec_df"]
    return pd.DataFrame()


def _resolve_auto_variant_base(variant: str) -> str:
    token = str(variant).strip().upper()
    if token == "H_ANCHOR_AUTO":
        return "H_ANCHOR"
    if token == "G+H_ANCHOR_AUTO":
        return "G+H_ANCHOR"
    return token


def _resolve_auto_feature_spec_df(
    *,
    timeline_anchor_feature_info_by_radius: dict[str, dict[int, dict[str, Any]]],
    timeline_name: str,
    input_variant: str,
    radius: int,
) -> pd.DataFrame:
    variant = str(input_variant).strip().upper()
    if variant not in AUTO_WINDOW_VARIANTS:
        return pd.DataFrame()
    info = timeline_anchor_feature_info_by_radius.get(timeline_name, {}).get(int(radius), {})
    return info.get("feature_spec_df", pd.DataFrame()).copy()


def _availability_row(timeline: str, input_variant: str, x: pd.DataFrame, info: dict[str, Any]) -> dict[str, Any]:
    miss = x.isna()
    row_miss = miss.mean(axis=1)
    overall_missing_ratio = float(miss.values.mean()) if x.size > 0 else float("nan")
    return {
        "timeline": timeline,
        "input_variant": input_variant,
        "n_samples": int(x.shape[0]),
        "n_features_total": int(info["n_features_total"]),
        "n_features_h_phase": int(info.get("n_features_h_phase", info.get("n_features_h_full", 0))),
        "n_features_h_anchor_local": int(info.get("n_features_h_anchor_local", 0)),
        "n_features_h_anchor_vi": int(info.get("n_features_h_anchor_vi", 0)),
        "n_features_h_anchor_auto": int(info.get("n_features_h_anchor_auto", 0)),
        "n_features_h_full": int(info.get("n_features_h_full", 0)),
        "n_features_g": int(info["n_features_g"]),
        "overall_missing_ratio": overall_missing_ratio,
        "row_missing_ratio_mean": float(row_miss.mean()),
        "row_missing_ratio_std": float(row_miss.std(ddof=0)),
        "fully_observed_sample_count": int((row_miss == 0).sum()),
        "partially_missing_sample_count": int(((row_miss > 0) & (row_miss <= 0.5)).sum()),
        "severely_missing_sample_count": int((row_miss > 0.5).sum()),
    }


def _availability_row_auto(
    timeline: str,
    input_variant: str,
    candidate_frames: dict[int, pd.DataFrame],
    candidate_infos: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    radius_candidates = sorted(candidate_infos.keys())
    if not radius_candidates:
        return {"timeline": timeline, "input_variant": input_variant, "auto_window_dynamic": True}
    first_radius = radius_candidates[0]
    n_samples = int(candidate_frames[first_radius].shape[0])
    total_by_radius = {int(r): int(candidate_infos[r]["n_features_total"]) for r in radius_candidates}
    h_auto_by_radius = {int(r): int(candidate_infos[r].get("n_features_h_anchor_auto", 0)) for r in radius_candidates}
    return {
        "timeline": timeline,
        "input_variant": input_variant,
        "n_samples": n_samples,
        "n_features_total": float("nan"),
        "n_features_h_phase": 0,
        "n_features_h_anchor_local": float("nan"),
        "n_features_h_anchor_vi": float("nan"),
        "n_features_h_anchor_auto": float("nan"),
        "n_features_h_full": 0,
        "n_features_g": int(candidate_infos[first_radius].get("n_features_g", 0)),
        "overall_missing_ratio": float("nan"),
        "row_missing_ratio_mean": float("nan"),
        "row_missing_ratio_std": float("nan"),
        "fully_observed_sample_count": float("nan"),
        "partially_missing_sample_count": float("nan"),
        "severely_missing_sample_count": float("nan"),
        "auto_window_dynamic": True,
        "window_radius_candidates_json": json.dumps(radius_candidates, ensure_ascii=False),
        "n_features_total_min": int(min(total_by_radius.values())),
        "n_features_total_max": int(max(total_by_radius.values())),
        "n_features_total_by_radius_json": json.dumps(total_by_radius, ensure_ascii=False, sort_keys=True),
        "n_features_h_anchor_auto_min": int(min(h_auto_by_radius.values())),
        "n_features_h_anchor_auto_max": int(max(h_auto_by_radius.values())),
        "n_features_h_anchor_auto_by_radius_json": json.dumps(h_auto_by_radius, ensure_ascii=False, sort_keys=True),
    }


def _safe_abs_corr(x: pd.Series, y: pd.Series) -> float:
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 3:
        return 0.0
    x_use = x_arr[mask]
    y_use = y_arr[mask]
    if float(np.std(x_use, ddof=0)) <= 1.0e-12 or float(np.std(y_use, ddof=0)) <= 1.0e-12:
        return 0.0
    corr = np.corrcoef(x_use, y_use)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(abs(corr))


def _resolve_anchor_local_score_target(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    g_cols: list[str],
    input_variant: str,
    pruning_cfg: dict[str, Any],
    g_pred_train: np.ndarray | None = None,
) -> tuple[pd.Series, str]:
    y_direct = y_train.astype(float).copy()
    blend = float(pruning_cfg.get("g_aware_score_blend", 1.0))
    if abs(blend - 1.0) > 1.0e-12:
        raise ValueError(
            "正式 Result 3.3B/3.4A 口径要求 anchor_local_pruning.g_aware_score_blend=1.0，"
            "所有 H group 打分必须基于原始 y。"
        )
    return y_direct, "y_direct_g_aware_blend_1.00"


def _split_h_g_columns(
    x: pd.DataFrame,
    *,
    feature_spec_df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    if feature_spec_df.empty:
        return [], list(x.columns)
    h_cols = [str(col) for col in feature_spec_df["feature"].astype(str).tolist() if str(col) in x.columns]
    h_col_set = set(h_cols)
    g_cols = [str(col) for col in x.columns if str(col) not in h_col_set]
    return h_cols, g_cols


def _meta_signature(meta_df: pd.DataFrame) -> str:
    ids = [str(x) for x in meta_df.index.astype(str).tolist()]
    raw = "\x1f".join(ids).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def _fit_predict_single_branch(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_apply: pd.DataFrame,
    predictor: str,
    params: dict[str, Any],
    preprocess_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    seed: int,
) -> np.ndarray:
    pipe = build_pipeline(
        predictor=predictor,
        params=params,
        n_features=x_train.shape[1],
        preprocess_cfg=preprocess_cfg,
        seed=seed,
        model_cfg=model_cfg,
    )
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_apply)
    return np.asarray(pred, dtype=float)


def _load_grm_assets(repo_root: Path) -> tuple[np.ndarray, pd.DataFrame]:
    cache_key = str(repo_root.resolve())
    cached = _GRM_ASSET_CACHE.get(cache_key)
    if cached is not None:
        return cached

    grm_matrix_path = repo_root / "data" / "processed" / "grm.npy"
    grm_index_path = repo_root / "data" / "processed" / "grm_index.parquet"
    if not grm_matrix_path.exists():
        raise FileNotFoundError(f"缺少 GRM 矩阵: {grm_matrix_path}")
    if not grm_index_path.exists():
        raise FileNotFoundError(f"缺少 GRM 索引: {grm_index_path}")

    grm = np.load(grm_matrix_path)
    grm_index_df = pd.read_parquet(grm_index_path)
    if "genotype_id" not in grm_index_df.columns:
        raise ValueError("grm_index.parquet 缺少 genotype_id 列。")
    _GRM_ASSET_CACHE[cache_key] = (grm, grm_index_df)
    return grm, grm_index_df


def _write_grm_csv(grm: np.ndarray, grm_index_df: pd.DataFrame, out_path: Path) -> None:
    ids = [str(x) for x in grm_index_df["genotype_id"].astype(str).tolist()]
    grm_df = pd.DataFrame(grm, index=ids, columns=ids)
    grm_df.index.name = "genotype_id"
    grm_df.to_csv(out_path)


def _fit_predict_gblup_sommer(
    *,
    train_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    y_train: pd.Series,
    repo_root: Path,
    seed: int,
) -> dict[str, Any]:
    if not HAS_RSCRIPT:
        raise RuntimeError("当前环境不可用 Rscript，无法运行 sommer GBLUP。")

    grm, grm_index_df = _load_grm_assets(repo_root)
    grm_ids = [str(x) for x in grm_index_df["genotype_id"].astype(str).tolist()]

    work_rows: list[dict[str, Any]] = []
    for meta_df, y_series, use_y in [
        (train_meta, y_train, True),
        (val_meta, None, False),
        (test_meta, None, False),
    ]:
        for plot_id in meta_df.index.astype(str).tolist():
            row = {
                "plot_id": str(plot_id),
                "genotype_id": str(meta_df.loc[plot_id, "genotype_id"]),
                "y": float(y_series.loc[plot_id]) if use_y else float("nan"),
            }
            work_rows.append(row)

    fit_df = pd.DataFrame(work_rows)
    fit_df["genotype_id"] = fit_df["genotype_id"].astype(str)

    with tempfile.TemporaryDirectory(prefix="result34_gblup_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        fit_csv = tmp_path / "fit_data.csv"
        grm_csv = tmp_path / "grm.csv"
        out_csv = tmp_path / "predictions.csv"
        script_path = repo_root / "scripts" / "sommer_gblup_predict.R"

        fit_df.to_csv(fit_csv, index=False)
        _write_grm_csv(grm, grm_index_df, grm_csv)

        cmd = [
            "Rscript",
            str(script_path),
            str(fit_csv),
            str(grm_csv),
            str(out_csv),
        ]
        env = os.environ.copy()
        env.setdefault("R_DEFAULT_PACKAGES", "stats,graphics,grDevices,utils,methods")
        env.setdefault("LC_ALL", "C")
        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                "sommer GBLUP 运行失败。\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        if not out_csv.exists():
            raise RuntimeError("sommer GBLUP 未生成预测文件。")

        pred_df = pd.read_csv(out_csv)

    geno_col = None
    for candidate in ["genotype_id", "ID", "Name", "id"]:
        if candidate in pred_df.columns:
            geno_col = candidate
            break
    if geno_col is None:
        object_cols = [c for c in pred_df.columns if pred_df[c].dtype == object]
        if not object_cols:
            raise RuntimeError(f"无法识别 GBLUP 输出中的基因型列: {pred_df.columns.tolist()}")
        geno_col = object_cols[0]

    pred_col = None
    for candidate in ["predicted.value", "predicted_value", "prediction", "pred"]:
        if candidate in pred_df.columns:
            pred_col = candidate
            break
    if pred_col is None:
        numeric_cols = [c for c in pred_df.columns if c != geno_col and pd.api.types.is_numeric_dtype(pred_df[c])]
        if not numeric_cols:
            raise RuntimeError(f"无法识别 GBLUP 输出中的预测列: {pred_df.columns.tolist()}")
        pred_col = numeric_cols[0]

    pred_df = pred_df[[geno_col, pred_col]].copy()
    pred_df[geno_col] = pred_df[geno_col].astype(str)
    pred_map = dict(zip(pred_df[geno_col].tolist(), pred_df[pred_col].astype(float).tolist()))
    missing = sorted(set(str(x) for x in fit_df["genotype_id"].unique().tolist()) - set(pred_map.keys()))
    if missing:
        raise RuntimeError(f"sommer GBLUP 预测缺少 genotype: {missing[:5]}")

    def _map_meta(meta_df: pd.DataFrame) -> np.ndarray:
        return np.asarray([float(pred_map[str(meta_df.loc[plot_id, "genotype_id"])]) for plot_id in meta_df.index.astype(str)], dtype=float)

    return {
        "pred_train": _map_meta(train_meta),
        "pred_val": _map_meta(val_meta),
        "pred_test": _map_meta(test_meta),
        "pred_map": pred_map,
        "grm_ids": grm_ids,
    }


def _fit_predict_g_branch(
    *,
    train_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    y_train: pd.Series,
    x_train_g: pd.DataFrame,
    x_val_g: pd.DataFrame,
    x_test_g: pd.DataFrame,
    predictor: str,
    params: dict[str, Any],
    preprocess_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    seed: int,
    repo_root: Path,
    use_gblup: bool,
) -> dict[str, np.ndarray]:
    if use_gblup:
        return _fit_predict_gblup_sommer(
            train_meta=train_meta,
            val_meta=val_meta,
            test_meta=test_meta,
            y_train=y_train,
            repo_root=repo_root,
            seed=seed,
        )
    pred_train = _fit_predict_single_branch(
        x_train=x_train_g,
        y_train=y_train,
        x_apply=x_train_g,
        predictor=predictor,
        params=params,
        preprocess_cfg=preprocess_cfg,
        model_cfg=model_cfg,
        seed=seed,
    )
    pred_val = _fit_predict_single_branch(
        x_train=x_train_g,
        y_train=y_train,
        x_apply=x_val_g,
        predictor=predictor,
        params=params,
        preprocess_cfg=preprocess_cfg,
        model_cfg=model_cfg,
        seed=seed + 1,
    )
    pred_test = _fit_predict_single_branch(
        x_train=x_train_g,
        y_train=y_train,
        x_apply=x_test_g,
        predictor=predictor,
        params=params,
        preprocess_cfg=preprocess_cfg,
        model_cfg=model_cfg,
        seed=seed + 2,
    )
    return {
        "pred_train": np.asarray(pred_train, dtype=float),
        "pred_val": np.asarray(pred_val, dtype=float),
        "pred_test": np.asarray(pred_test, dtype=float),
    }


def _get_or_fit_cached_g_branch(
    *,
    inner_ctx: dict[str, Any],
    test_meta: pd.DataFrame,
    y_train: pd.Series,
    x_train_g: pd.DataFrame,
    x_val_g: pd.DataFrame,
    x_test_g: pd.DataFrame,
    predictor: str,
    params: dict[str, Any],
    preprocess_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    seed: int,
    repo_root: Path,
    use_gblup: bool,
) -> dict[str, np.ndarray]:
    cache_key = ""
    if use_gblup:
        cache_store = inner_ctx.setdefault("_g_branch_cache", {})
        cache_key = f"gblup::{_meta_signature(test_meta)}"
        cached = cache_store.get(cache_key)
        if cached is not None:
            return cached

    result = _fit_predict_g_branch(
        train_meta=inner_ctx["meta_train"],
        val_meta=inner_ctx["meta_val"],
        test_meta=test_meta,
        y_train=y_train,
        x_train_g=x_train_g,
        x_val_g=x_val_g,
        x_test_g=x_test_g,
        predictor=predictor,
        params=params,
        preprocess_cfg=preprocess_cfg,
        model_cfg=model_cfg,
        seed=seed,
        repo_root=repo_root,
        use_gblup=use_gblup,
    )
    if use_gblup:
        cache_store[cache_key] = result
    return result


def _fit_least_squares_fusion_weights(
    *,
    y_val: pd.Series,
    pred_val_h: np.ndarray,
    pred_val_g: np.ndarray,
) -> tuple[float, float, float] | None:
    y_arr = pd.to_numeric(y_val, errors="coerce").to_numpy(dtype=float)
    h_arr = np.asarray(pred_val_h, dtype=float)
    g_arr = np.asarray(pred_val_g, dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(h_arr) & np.isfinite(g_arr)
    if int(mask.sum()) < 3:
        return None
    design = np.column_stack([np.ones(int(mask.sum())), h_arr[mask], g_arr[mask]])
    try:
        coef, *_ = np.linalg.lstsq(design, y_arr[mask], rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coef)):
        return None
    intercept, weight_h, weight_g = [float(x) for x in coef.tolist()]
    return intercept, weight_h, weight_g


def _apply_fusion_weights(
    *,
    pred_h: np.ndarray,
    pred_g: np.ndarray,
    intercept: float,
    weight_h: float,
    weight_g: float,
) -> np.ndarray:
    return (
        float(intercept)
        + float(weight_h) * np.asarray(pred_h, dtype=float)
        + float(weight_g) * np.asarray(pred_g, dtype=float)
    )


def _mode_or_empty(values: list[str]) -> str:
    if not values:
        return ""
    return str(pd.Series(values).mode().iloc[0])


def _mode_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(pd.Series(values).mode().iloc[0])


def _fit_predict_safe_fusion(
    *,
    inner_payload: list[dict[str, Any]],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    meta_test: pd.DataFrame,
    predictor: str,
    params: dict[str, Any],
    preprocess_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    feature_spec_df: pd.DataFrame,
    seed: int,
    fusion_cfg: dict[str, Any],
    repo_root: Path,
    use_gblup_for_g: bool,
) -> dict[str, Any]:
    metric_name = str(fusion_cfg.get("selection_metric", "pearson"))
    tie_tolerance = float(fusion_cfg.get("candidate_tie_tolerance", 1.0e-9))

    train_metrics_list: list[dict[str, Any]] = []
    val_h_metrics_list: list[dict[str, Any]] = []
    val_g_metrics_list: list[dict[str, Any]] = []
    test_inner_preds: list[np.ndarray] = []
    val_metrics_list: list[dict[str, Any]] = []
    candidate_selected_list: list[str] = []
    weight_h_list: list[float] = []
    weight_g_list: list[float] = []
    intercept_list: list[float] = []

    for step_i, inner in enumerate(inner_payload):
        h_cols, g_cols = _split_h_g_columns(inner["x_train"], feature_spec_df=feature_spec_df)
        if h_cols:
            x_train_h = inner["x_train"].loc[:, h_cols].copy()
            x_val_h = inner["x_val"].loc[:, h_cols].copy()
            x_test_h = x_test.loc[:, h_cols].copy()

            pred_train_h = _fit_predict_single_branch(
                x_train=x_train_h,
                y_train=inner["y_train"],
                x_apply=x_train_h,
                predictor=predictor,
                params=params,
                preprocess_cfg=preprocess_cfg,
                model_cfg=model_cfg,
                seed=seed + step_i * 101 + 1,
            )
            pred_val_h = _fit_predict_single_branch(
                x_train=x_train_h,
                y_train=inner["y_train"],
                x_apply=x_val_h,
                predictor=predictor,
                params=params,
                preprocess_cfg=preprocess_cfg,
                model_cfg=model_cfg,
                seed=seed + step_i * 101 + 2,
            )
            pred_test_h = _fit_predict_single_branch(
                x_train=x_train_h,
                y_train=inner["y_train"],
                x_apply=x_test_h,
                predictor=predictor,
                params=params,
                preprocess_cfg=preprocess_cfg,
                model_cfg=model_cfg,
                seed=seed + step_i * 101 + 3,
            )
            h_metrics = regression_metrics(inner["y_val"].to_numpy(dtype=float), pred_val_h)
            val_h_metrics_list.append(h_metrics)
        else:
            pred_train_h = None
            pred_val_h = None
            pred_test_h = None
            h_metrics = _nan_metric_dict()
            val_h_metrics_list.append(h_metrics)

        if not g_cols:
            if pred_train_h is None or pred_val_h is None or pred_test_h is None:
                raise RuntimeError("safe fusion 在 H/G 两个分支同时为空时无法运行。")
            selected_candidate = "h_only"
            intercept_selected = 0.0
            weight_h_selected = 1.0
            weight_g_selected = 0.0
            pred_train = pred_train_h
            pred_val = pred_val_h
            pred_test = pred_test_h
            val_g_metrics_list.append(_nan_metric_dict())
        else:
            x_train_g = inner["x_train"].loc[:, g_cols].copy()
            x_val_g = inner["x_val"].loc[:, g_cols].copy()
            x_test_g = x_test.loc[:, g_cols].copy()
            g_branch = _get_or_fit_cached_g_branch(
                inner_ctx=inner,
                test_meta=meta_test,
                y_train=inner["y_train"],
                x_train_g=x_train_g,
                x_val_g=x_val_g,
                x_test_g=x_test_g,
                predictor=predictor,
                params=params,
                preprocess_cfg=preprocess_cfg,
                model_cfg=model_cfg,
                seed=seed + step_i * 101 + 4,
                repo_root=repo_root,
                use_gblup=use_gblup_for_g,
            )
            pred_train_g = g_branch["pred_train"]
            pred_val_g = g_branch["pred_val"]
            pred_test_g = g_branch["pred_test"]

            g_metrics = regression_metrics(inner["y_val"].to_numpy(dtype=float), pred_val_g)
            val_g_metrics_list.append(g_metrics)
            if pred_train_h is None or pred_val_h is None or pred_test_h is None:
                selected_candidate = "g_only"
                intercept_selected = 0.0
                weight_h_selected = 0.0
                weight_g_selected = 1.0
                pred_train = pred_train_g
                pred_val = pred_val_g
                pred_test = pred_test_g
            else:
                candidate_rows: list[dict[str, Any]] = [
                    {
                        "name": "g_only",
                        "priority": 0,
                        "intercept": 0.0,
                        "weight_h": 0.0,
                        "weight_g": 1.0,
                        "pred_train": pred_train_g,
                        "pred_val": pred_val_g,
                        "pred_test": pred_test_g,
                        "metrics": g_metrics,
                    },
                    {
                        "name": "h_only",
                        "priority": 1,
                        "intercept": 0.0,
                        "weight_h": 1.0,
                        "weight_g": 0.0,
                        "pred_train": pred_train_h,
                        "pred_val": pred_val_h,
                        "pred_test": pred_test_h,
                        "metrics": h_metrics,
                    },
                ]
                ols_weights = _fit_least_squares_fusion_weights(
                    y_val=inner["y_val"],
                    pred_val_h=pred_val_h,
                    pred_val_g=pred_val_g,
                )
                if ols_weights is not None:
                    intercept_ols, weight_h_ols, weight_g_ols = ols_weights
                    pred_train_ols = _apply_fusion_weights(
                        pred_h=pred_train_h,
                        pred_g=pred_train_g,
                        intercept=intercept_ols,
                        weight_h=weight_h_ols,
                        weight_g=weight_g_ols,
                    )
                    pred_val_ols = _apply_fusion_weights(
                        pred_h=pred_val_h,
                        pred_g=pred_val_g,
                        intercept=intercept_ols,
                        weight_h=weight_h_ols,
                        weight_g=weight_g_ols,
                    )
                    pred_test_ols = _apply_fusion_weights(
                        pred_h=pred_test_h,
                        pred_g=pred_test_g,
                        intercept=intercept_ols,
                        weight_h=weight_h_ols,
                        weight_g=weight_g_ols,
                    )
                    candidate_rows.append(
                        {
                            "name": "ols_h_g",
                            "priority": 2,
                            "intercept": intercept_ols,
                            "weight_h": weight_h_ols,
                            "weight_g": weight_g_ols,
                            "pred_train": pred_train_ols,
                            "pred_val": pred_val_ols,
                            "pred_test": pred_test_ols,
                            "metrics": regression_metrics(
                                inner["y_val"].to_numpy(dtype=float),
                                pred_val_ols,
                            ),
                        }
                    )

                candidate_rows = sorted(
                    candidate_rows,
                    key=lambda row: (
                        -float(_extract_selection_score(row["metrics"], metric_name)),
                        int(row["priority"]),
                    ),
                )
                best_row = candidate_rows[0]
                best_score = float(_extract_selection_score(best_row["metrics"], metric_name))
                for cand in candidate_rows[1:]:
                    cand_score = float(_extract_selection_score(cand["metrics"], metric_name))
                    if abs(cand_score - best_score) <= tie_tolerance and int(cand["priority"]) < int(best_row["priority"]):
                        best_row = cand
                        best_score = cand_score

                selected_candidate = str(best_row["name"])
                intercept_selected = float(best_row["intercept"])
                weight_h_selected = float(best_row["weight_h"])
                weight_g_selected = float(best_row["weight_g"])
                pred_train = np.asarray(best_row["pred_train"], dtype=float)
                pred_val = np.asarray(best_row["pred_val"], dtype=float)
                pred_test = np.asarray(best_row["pred_test"], dtype=float)

        candidate_selected_list.append(str(selected_candidate))
        intercept_list.append(float(intercept_selected))
        weight_h_list.append(float(weight_h_selected))
        weight_g_list.append(float(weight_g_selected))
        train_metrics_list.append(regression_metrics(inner["y_train"].to_numpy(dtype=float), pred_train))
        val_metrics_list.append(regression_metrics(inner["y_val"].to_numpy(dtype=float), pred_val))
        test_inner_preds.append(np.asarray(pred_test, dtype=float))

    pred_mat = np.column_stack(test_inner_preds)
    pred_ens = pred_mat.mean(axis=1)
    test_metrics = regression_metrics(y_test.to_numpy(dtype=float), pred_ens)
    mean_train_metrics = mean_regression_metrics(train_metrics_list)
    mean_val_metrics = mean_regression_metrics(val_metrics_list)
    mean_val_h_metrics = mean_regression_metrics(val_h_metrics_list)
    mean_val_g_metrics = mean_regression_metrics(val_g_metrics_list)
    return {
        "pred_mat": pred_mat,
        "pred_ens": pred_ens,
        "test_metrics": test_metrics,
        "mean_train_metrics": mean_train_metrics,
        "mean_val_metrics": mean_val_metrics,
        "mean_val_h_metrics": mean_val_h_metrics,
        "mean_val_g_metrics": mean_val_g_metrics,
        "candidate_selected_list": candidate_selected_list,
        "candidate_selected_mode": _mode_or_empty(candidate_selected_list),
        "intercept_list": intercept_list,
        "intercept_mean": float(np.mean(intercept_list)) if intercept_list else float("nan"),
        "intercept_mode": _mode_or_nan(intercept_list),
        "weight_h_list": weight_h_list,
        "weight_h_mean": float(np.mean(weight_h_list)) if weight_h_list else float("nan"),
        "weight_h_mode": _mode_or_nan(weight_h_list),
        "weight_g_list": weight_g_list,
        "weight_g_mean": float(np.mean(weight_g_list)) if weight_g_list else float("nan"),
        "weight_g_mode": _mode_or_nan(weight_g_list),
    }


def _default_params_for_predictor(
    predictor: str,
    *,
    n_features: int,
    model_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_cfg = model_cfg or {}
    token = str(predictor).strip().lower()
    if token == "ridge":
        return {"alpha": 1.0}
    if token == "lasso":
        return {"alpha": 0.1}
    if token == "elasticnet":
        return {"alpha": 0.1, "l1_ratio": 0.5}
    if token == "pls":
        return {"n_components": int(max(1, min(8, n_features))), "scale": False}
    if token == "extratrees":
        return {"n_estimators": 40, "max_depth": 10, "min_samples_leaf": 2, "max_features": "sqrt"}
    if token == "random_forest":
        return {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 2, "max_features": "sqrt"}
    if token == "lightgbm":
        backend = str(model_cfg.get("lightgbm_backend", "sklearn_gbrt")).lower()
        if backend == "native":
            return {
                "n_estimators": 50,
                "learning_rate": 0.05,
                "num_leaves": 15,
                "max_depth": 4,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 1.0e-4,
                "reg_lambda": 1.0e-2,
            }
        return {"n_estimators": 50, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.8}
    if token == "xgboost":
        return {
            "n_estimators": 60,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_child_weight": 2.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 1.0e-4,
            "reg_lambda": 1.0e-2,
        }
    raise ValueError(f"未知 predictor: {predictor}")


def _aggregate_group_scores(score_df: pd.DataFrame, group_col: str, agg: str) -> pd.DataFrame:
    if score_df.empty:
        return pd.DataFrame(columns=[group_col, "score"])
    agg_name = str(agg).strip().lower()
    if agg_name not in {"mean", "max", "median"} and not agg_name.startswith("top"):
        raise ValueError(f"不支持的 group score 聚合方式: {agg}")
    if agg_name.startswith("top"):
        raw_k = agg_name.replace("top", "").replace("_mean", "").strip()
        k = max(1, int(raw_k or "3"))
        grouped = (
            score_df.sort_values(["score"], ascending=[False])
            .groupby(group_col, as_index=False)
            .head(k)
            .groupby(group_col, as_index=False)["score"]
            .mean()
        )
    else:
        grouped = score_df.groupby(group_col, as_index=False)["score"].agg(agg_name)
    return grouped.sort_values(["score", group_col], ascending=[False, True]).reset_index(drop=True)


def _apply_group_quota_floor(
    group_scores: pd.DataFrame,
    *,
    group_col: str,
    quota_col: str,
    min_per_quota: int,
) -> list[str]:
    if min_per_quota <= 0 or group_scores.empty:
        return []
    keep_groups: list[str] = []
    for _, sub in group_scores.groupby(quota_col, dropna=False):
        picked = (
            sub.sort_values(["score", group_col], ascending=[False, True])
            .head(min_per_quota)[group_col]
            .astype(str)
            .tolist()
        )
        keep_groups.extend(picked)
    return list(dict.fromkeys(keep_groups))


def _effective_min_per_phase_for_k(
    group_scores: pd.DataFrame,
    *,
    requested_min_per_phase: int,
    n_keep: int,
) -> int:
    requested = max(0, int(requested_min_per_phase))
    n_keep = max(0, int(n_keep))
    if requested <= 0 or n_keep <= 0 or group_scores.empty or "phase" not in group_scores.columns:
        return 0
    n_phases_present = int(group_scores["phase"].fillna("na_phase").astype(str).nunique())
    if n_phases_present <= 0:
        return 0
    # Respect the requested per-phase floor when feasible, but do not let the quota
    # exceed the current K budget under early/prefix-restricted settings.
    return min(requested, n_keep // n_phases_present)


def _select_anchor_vi_groups_by_count(
    group_scores: pd.DataFrame,
    *,
    group_col: str,
    n_keep: int,
    min_per_phase: int,
) -> set[str]:
    if group_scores.empty:
        return set()

    effective_min_per_phase = _effective_min_per_phase_for_k(
        group_scores,
        requested_min_per_phase=min_per_phase,
        n_keep=n_keep,
    )
    base_keep = _apply_group_quota_floor(
        group_scores,
        group_col=group_col,
        quota_col="phase",
        min_per_quota=effective_min_per_phase,
    )
    target_keep = max(0, max(int(n_keep), len(base_keep)))
    if target_keep == 0:
        return set()
    ranked_groups = group_scores.sort_values(["score", group_col], ascending=[False, True])[group_col].astype(str).tolist()
    keep_unique = list(dict.fromkeys(base_keep))
    if len(keep_unique) >= target_keep:
        return set(keep_unique[:target_keep])
    for group_id in ranked_groups:
        if group_id not in keep_unique:
            keep_unique.append(group_id)
        if len(keep_unique) >= target_keep:
            break
    return set(keep_unique[:target_keep])


def _group_feature_map(h_spec: pd.DataFrame) -> dict[str, list[str]]:
    if h_spec.empty:
        return {}
    grouped = (
        h_spec[["anchor_vi_group", "feature"]]
        .dropna()
        .drop_duplicates()
        .groupby("anchor_vi_group")["feature"]
        .apply(lambda s: [str(x) for x in s.tolist()])
    )
    return {str(group_id): features for group_id, features in grouped.items()}


def _extend_groups_to_feature_floor(
    *,
    keep_groups: set[str],
    h_spec: pd.DataFrame,
    ranked_groups: list[str],
    min_h_features: int,
) -> tuple[set[str], list[str]]:
    group_to_features = _group_feature_map(h_spec)
    keep_group_order = [str(g) for g in ranked_groups if str(g) in set(str(x) for x in keep_groups)]
    if not keep_group_order:
        keep_group_order = [str(g) for g in keep_groups if str(g) in group_to_features]
    keep_group_order = list(dict.fromkeys(keep_group_order))
    keep_group_set = set(keep_group_order)

    keep_h_cols: list[str] = []
    for group_id in keep_group_order:
        keep_h_cols.extend(group_to_features.get(group_id, []))
    keep_h_cols = list(dict.fromkeys(keep_h_cols))

    if len(keep_h_cols) >= int(min_h_features):
        return keep_group_set, keep_h_cols

    for group_id in ranked_groups:
        group_id = str(group_id)
        if group_id in keep_group_set or group_id not in group_to_features:
            continue
        keep_group_set.add(group_id)
        keep_group_order.append(group_id)
        keep_h_cols.extend(group_to_features[group_id])
        keep_h_cols = list(dict.fromkeys(keep_h_cols))
        if len(keep_h_cols) >= int(min_h_features):
            break
    return keep_group_set, keep_h_cols


def _extract_selection_score(metric_dict: dict[str, Any], metric_name: str) -> float:
    token = str(metric_name).strip().lower()
    if token == "pearson":
        return float(metric_dict.get("pearson_r", float("-inf")))
    if token == "r2":
        return float(metric_dict.get("r2", float("-inf")))
    if token == "rmse":
        return -float(metric_dict.get("rmse", float("inf")))
    if token == "mae":
        return -float(metric_dict.get("mae", float("inf")))
    raise ValueError(f"不支持的 group_selection_metric: {metric_name}")


def _resolve_ridge_alpha_grid_cfg(optuna_cfg: dict[str, Any]) -> dict[str, Any] | None:
    raw = optuna_cfg.get("ridge_alpha_grid", None)
    if raw in (None, False):
        return None
    if raw is True:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("optuna.ridge_alpha_grid 必须是字典或布尔值。")
    if not bool(raw.get("enabled", False)):
        return None

    values_raw = raw.get("values")
    if values_raw in (None, "", []):
        min_alpha = float(raw.get("min_alpha", 1.0e-3))
        max_alpha = float(raw.get("max_alpha", 1.0e2))
        num = int(raw.get("num", 21))
        if min_alpha <= 0 or max_alpha <= 0 or max_alpha <= min_alpha:
            raise ValueError("optuna.ridge_alpha_grid 的 min_alpha/max_alpha 必须为正且 max_alpha > min_alpha。")
        if num < 2:
            raise ValueError("optuna.ridge_alpha_grid.num 至少为 2。")
        values = np.logspace(np.log10(min_alpha), np.log10(max_alpha), num=num).tolist()
    else:
        values = [float(x) for x in values_raw]

    values = sorted({float(x) for x in values if float(x) > 0})
    if not values:
        raise ValueError("optuna.ridge_alpha_grid 没有可用 alpha 值。")
    return {
        "enabled": True,
        "values": values,
        "selection_metric": str(raw.get("selection_metric", "pearson")).strip().lower(),
        "tie_tolerance": float(raw.get("tie_tolerance", 0.005)),
        "fusion_tie_tolerance": float(raw.get("fusion_tie_tolerance", raw.get("tie_tolerance", 0.005))),
        "std_tie_tolerance": float(raw.get("std_tie_tolerance", 1.0e-12)),
        "prefer_larger_alpha_on_tie": bool(raw.get("prefer_larger_alpha_on_tie", True)),
        "fusion_prefer_larger_alpha_on_tie": bool(
            raw.get("fusion_prefer_larger_alpha_on_tie", raw.get("prefer_larger_alpha_on_tie", True))
        ),
    }


def _trial_attr_float(trial: optuna.trial.FrozenTrial, name: str, default: float) -> float:
    try:
        value = float(trial.user_attrs.get(name, default))
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)


def _select_best_ridge_grid_trial(
    complete_trials: list[optuna.trial.FrozenTrial],
    *,
    grid_cfg: dict[str, Any],
    input_variant: str,
) -> tuple[optuna.trial.FrozenTrial, list[optuna.trial.FrozenTrial]]:
    metric = str(grid_cfg.get("selection_metric", "pearson")).strip().lower()
    if metric != "pearson":
        raise ValueError("当前 ridge_alpha_grid 只支持 selection_metric=pearson。")
    is_fusion_variant = str(input_variant).strip().upper() in G_H_FUSION_VARIANTS
    tie_key = "fusion_tie_tolerance" if is_fusion_variant else "tie_tolerance"
    prefer_key = "fusion_prefer_larger_alpha_on_tie" if is_fusion_variant else "prefer_larger_alpha_on_tie"
    tie_tolerance = max(0.0, float(grid_cfg.get(tie_key, grid_cfg.get("tie_tolerance", 0.005))))
    std_tie_tolerance = max(0.0, float(grid_cfg.get("std_tie_tolerance", 1.0e-12)))
    prefer_larger_alpha = bool(grid_cfg.get(prefer_key, grid_cfg.get("prefer_larger_alpha_on_tie", True)))

    def score(t: optuna.trial.FrozenTrial) -> float:
        return _trial_attr_float(t, "mean_val_pearson", float("-inf"))

    def std_score(t: optuna.trial.FrozenTrial) -> float:
        return _trial_attr_float(t, "std_val_pearson", float("inf"))

    def alpha_value(t: optuna.trial.FrozenTrial) -> float:
        try:
            return float(t.params.get("alpha", float("nan")))
        except Exception:
            return float("nan")

    max_score = max(score(t) for t in complete_trials)
    near_best = [t for t in complete_trials if max_score - score(t) <= tie_tolerance]
    min_std = min(std_score(t) for t in near_best)
    stable = [t for t in near_best if std_score(t) - min_std <= std_tie_tolerance]
    if prefer_larger_alpha:
        best = max(stable, key=lambda t: alpha_value(t))
    else:
        best = min(stable, key=lambda t: alpha_value(t))

    remaining = [t for t in complete_trials if t.number != best.number]
    remaining = sorted(
        remaining,
        key=lambda t: (
            -score(t),
            std_score(t),
            -alpha_value(t) if prefer_larger_alpha else alpha_value(t),
            int(t.number),
        ),
    )
    return best, [best, *remaining]


def _evaluate_group_count_candidates(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    input_variant: str,
    feature_spec_df: pd.DataFrame,
    pruning_cfg: dict[str, Any],
    predictor: str,
    preprocess_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    seed: int,
    inner_ctx: dict[str, Any] | None = None,
    repo_root: Path | None = None,
    fusion_cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[int, float], dict[int, set[str]], dict[int, list[str]], pd.DataFrame]:
    variant = str(input_variant).strip().upper()
    if variant not in REDUCED_H_VARIANTS:
        return pd.DataFrame(), {}, {}, {}, pd.DataFrame()

    h_spec = feature_spec_df[feature_spec_df["feature"].isin(x_train.columns)].copy()
    if h_spec.empty:
        return pd.DataFrame(), {}, {}, {}, pd.DataFrame()

    h_spec["anchor_vi_group"] = (
        h_spec["anchor_token"].astype(str).fillna("na_anchor") + "||" + h_spec["vi_name"].astype(str).fillna("na_vi")
    )
    h_cols = h_spec["feature"].astype(str).tolist()
    h_col_set = set(h_cols)
    g_cols = [col for col in x_train.columns if col not in h_col_set]
    fusion_cfg = fusion_cfg or {}
    score_target, score_target_mode = _resolve_anchor_local_score_target(
        x_train=x_train,
        y_train=y_train,
        g_cols=g_cols,
        input_variant=variant,
        pruning_cfg=pruning_cfg,
    )

    feature_scores = [{"feature": col, "score": _safe_abs_corr(x_train[col], score_target)} for col in h_cols]
    score_df = pd.DataFrame(feature_scores).merge(
        h_spec[["feature", "anchor_token", "vi_name", "phase", "family", "anchor_vi_group"]],
        on="feature",
        how="left",
    )
    group_scores = _aggregate_group_scores(
        score_df,
        group_col="anchor_vi_group",
        agg=str(pruning_cfg.get("group_score_agg", pruning_cfg.get("anchor_vi_score_agg", "top3_mean"))),
    )
    group_meta = (
        h_spec[["anchor_vi_group", "anchor_token", "vi_name", "phase", "family"]]
        .drop_duplicates(subset=["anchor_vi_group"])
        .copy()
    )
    group_scores = group_scores.merge(group_meta, on="anchor_vi_group", how="left")
    if group_scores.empty:
        return group_scores, {}, {}, {}, h_spec

    max_group_candidates = int(len(group_scores))
    max_groups = max(0, min(int(pruning_cfg.get("max_groups", max_group_candidates)), max_group_candidates))
    min_groups = max(0, min(int(pruning_cfg.get("min_groups", max(0, min(4, max_groups)))), max_groups))
    min_per_phase = max(0, int(pruning_cfg.get("group_min_per_phase", pruning_cfg.get("anchor_min_per_phase", 0))))
    min_h_features = max(0, int(pruning_cfg.get("min_h_features", 0)))
    step = max(1, int(pruning_cfg.get("group_k_step", 1)))
    metric_name = str(pruning_cfg.get("group_selection_metric", "pearson"))
    tie_tolerance = float(pruning_cfg.get("group_selection_tie_tolerance", 1.0e-6))

    candidate_ks = list(range(min_groups, max_groups + 1, step))
    if candidate_ks[-1] != max_groups:
        candidate_ks.append(max_groups)
    candidate_ks = sorted(set(candidate_ks))

    selection_predictor = str(pruning_cfg.get("group_selection_predictor", predictor)).strip().lower()
    params = _default_params_for_predictor(
        selection_predictor,
        n_features=max(1, int(x_train.shape[1])),
        model_cfg=model_cfg,
    )

    scores_by_k: dict[int, float] = {}
    groups_by_k: dict[int, set[str]] = {}
    cols_by_k: dict[int, list[str]] = {}
    ranked_groups = group_scores.sort_values(["score", "anchor_vi_group"], ascending=[False, True])[
        "anchor_vi_group"
    ].astype(str).tolist()

    x_val_use = sanitize_feature_matrix(x_val.copy())
    x_train_use = sanitize_feature_matrix(x_train.copy())
    use_safe_fusion = _use_safe_fusion_for_variant(input_variant=variant, fusion_cfg=fusion_cfg)
    use_gblup_for_g = bool(fusion_cfg.get("g_use_gblup", False))
    if use_safe_fusion and use_gblup_for_g and inner_ctx is not None:
        inner_ctx.setdefault("_g_branch_cache", {})

    for k in candidate_ks:
        keep_groups = _select_anchor_vi_groups_by_count(
            group_scores,
            group_col="anchor_vi_group",
            n_keep=int(k),
            min_per_phase=min_per_phase,
        )
        keep_groups, keep_h_cols = _extend_groups_to_feature_floor(
            keep_groups=keep_groups,
            h_spec=h_spec,
            ranked_groups=ranked_groups,
            min_h_features=min_h_features,
        )
        keep_set = set(g_cols) | set(keep_h_cols)
        keep_cols = [col for col in x_train_use.columns if col in keep_set]
        if not keep_cols:
            continue
        if use_safe_fusion and inner_ctx is not None and repo_root is not None:
            inner_for_k = dict(inner_ctx)
            if use_gblup_for_g:
                inner_for_k["_g_branch_cache"] = inner_ctx.setdefault("_g_branch_cache", {})
            inner_for_k["x_train"] = x_train.loc[:, keep_cols].copy()
            inner_for_k["x_val"] = x_val.loc[:, keep_cols].copy()
            safe_inner = _fit_predict_safe_fusion(
                inner_payload=[inner_for_k],
                x_test=x_val.loc[:, keep_cols].copy(),
                y_test=y_val,
                meta_test=inner_ctx["meta_val"],
                predictor=selection_predictor,
                params=params,
                preprocess_cfg=preprocess_cfg,
                model_cfg=model_cfg,
                feature_spec_df=feature_spec_df,
                seed=int(seed) + int(k),
                fusion_cfg=fusion_cfg,
                repo_root=repo_root,
                use_gblup_for_g=use_gblup_for_g,
            )
            val_metrics = safe_inner["mean_val_metrics"]
        else:
            pipe = build_pipeline(
                predictor=selection_predictor,
                params=params,
                n_features=len(keep_cols),
                preprocess_cfg=preprocess_cfg,
                seed=int(seed) + int(k),
                model_cfg=model_cfg,
            )
            pipe.fit(x_train_use.loc[:, keep_cols], y_train)
            pred_val = pipe.predict(x_val_use.loc[:, keep_cols])
            val_metrics = regression_metrics(y_val.to_numpy(dtype=float), pred_val)
        score_value = _extract_selection_score(val_metrics, metric_name)
        scores_by_k[int(k)] = float(score_value)
        groups_by_k[int(k)] = set(keep_groups)
        cols_by_k[int(k)] = list(keep_cols)

    return group_scores, scores_by_k, groups_by_k, cols_by_k, h_spec


def _apply_anchor_local_pruning(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_frames_apply: list[pd.DataFrame],
    input_variant: str,
    feature_spec_df: pd.DataFrame,
    pruning_cfg: dict[str, Any],
    stability_tag: str = "single_split",
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, Any]]:
    variant = str(input_variant).strip().upper()
    if variant not in REDUCED_H_VARIANTS:
        return x_train, x_frames_apply, {}
    if not bool(pruning_cfg.get("enabled", True)):
        return x_train, x_frames_apply, {"pruning_enabled": False}

    if feature_spec_df.empty:
        return x_train, x_frames_apply, {"pruning_enabled": False, "reason": "empty_feature_spec"}

    h_spec = feature_spec_df[feature_spec_df["feature"].isin(x_train.columns)].copy()
    if h_spec.empty:
        return x_train, x_frames_apply, {"pruning_enabled": False, "reason": "no_anchor_local_columns"}

    h_cols = h_spec["feature"].astype(str).tolist()
    h_col_set = set(h_cols)
    g_cols = [col for col in x_train.columns if col not in h_col_set]
    h_spec["anchor_vi_group"] = (
        h_spec["anchor_token"].astype(str).fillna("na_anchor") + "||" + h_spec["vi_name"].astype(str).fillna("na_vi")
    )

    score_target, score_target_mode = _resolve_anchor_local_score_target(
        x_train=x_train,
        y_train=y_train,
        g_cols=g_cols,
        input_variant=variant,
        pruning_cfg=pruning_cfg,
    )

    feature_scores = []
    for col in h_cols:
        feature_scores.append({"feature": col, "score": _safe_abs_corr(x_train[col], score_target)})
    score_df = pd.DataFrame(feature_scores).merge(
        h_spec[["feature", "anchor_token", "vi_name", "phase", "family", "anchor_vi_group"]],
        on="feature",
        how="left",
    )

    group_scores = _aggregate_group_scores(
        score_df,
        group_col="anchor_vi_group",
        agg=str(pruning_cfg.get("group_score_agg", pruning_cfg.get("anchor_vi_score_agg", "top3_mean"))),
    )
    group_meta = (
        h_spec[["anchor_vi_group", "anchor_token", "vi_name", "phase", "family"]]
        .drop_duplicates(subset=["anchor_vi_group"])
        .copy()
    )
    group_scores = group_scores.merge(group_meta, on="anchor_vi_group", how="left")

    max_group_candidates = int(len(group_scores))
    max_groups = int(pruning_cfg.get("max_groups", max_group_candidates))
    max_groups = max(0, min(max_groups, max_group_candidates))
    min_groups = int(pruning_cfg.get("min_groups", max(0, min(4, max_groups))))
    min_groups = max(0, min(min_groups, max_groups))
    requested_min_per_phase = max(0, int(pruning_cfg.get("group_min_per_phase", pruning_cfg.get("anchor_min_per_phase", 0))))
    effective_min_per_phase = _effective_min_per_phase_for_k(
        group_scores,
        requested_min_per_phase=requested_min_per_phase,
        n_keep=min_groups,
    )

    keep_groups = _select_anchor_vi_groups_by_count(
        group_scores,
        group_col="anchor_vi_group",
        n_keep=int(min_groups),
        min_per_phase=requested_min_per_phase,
    )
    min_h_features = max(0, int(pruning_cfg.get("min_h_features", 0)))
    ranked_groups = group_scores.sort_values(["score", "anchor_vi_group"], ascending=[False, True])[
        "anchor_vi_group"
    ].astype(str).tolist()
    keep_groups, keep_h_cols = _extend_groups_to_feature_floor(
        keep_groups=keep_groups,
        h_spec=h_spec,
        ranked_groups=ranked_groups,
        min_h_features=min_h_features,
    )
    kept_h_spec = h_spec[h_spec["anchor_vi_group"].astype(str).isin(keep_groups)].copy()

    keep_set = set(g_cols) | set(keep_h_cols)
    keep_cols = [col for col in x_train.columns if col in keep_set]
    x_train_kept = x_train.loc[:, keep_cols].copy()
    x_apply_kept = [frame.loc[:, keep_cols].copy() for frame in x_frames_apply]

    keep_anchor_count = int(kept_h_spec["anchor_token"].astype(str).nunique()) if not kept_h_spec.empty else 0
    keep_vi_count = int(kept_h_spec["vi_name"].astype(str).nunique()) if not kept_h_spec.empty else 0

    kept_groups_sorted = sorted(str(x) for x in keep_groups)
    kept_anchor_tokens_sorted = (
        sorted(kept_h_spec["anchor_token"].dropna().astype(str).unique().tolist()) if not kept_h_spec.empty else []
    )
    kept_vi_names_sorted = (
        sorted(kept_h_spec["vi_name"].dropna().astype(str).unique().tolist()) if not kept_h_spec.empty else []
    )

    stats = {
        "pruning_enabled": True,
        "score_target": score_target_mode,
        "stability_tag": stability_tag,
        "selection_unit": "anchor_x_vi_group",
        "n_anchor_candidates": int(h_spec["anchor_token"].astype(str).nunique()),
        "n_vi_candidates": int(h_spec["vi_name"].astype(str).nunique()),
        "n_group_candidates": int(len(group_scores)),
        "n_group_kept": int(len(keep_groups)),
        "n_anchor_kept": int(keep_anchor_count),
        "n_vi_kept": int(keep_vi_count),
        "n_h_features_before": int(len(h_cols)),
        "n_h_features_after": int(len(keep_h_cols)),
        "n_total_features_after": int(len(keep_cols)),
        "max_groups": int(max_groups),
        "min_groups": int(min_groups),
        "group_min_per_phase_requested": int(requested_min_per_phase),
        "group_min_per_phase_effective": int(effective_min_per_phase),
        "kept_group_ids_json": json.dumps(kept_groups_sorted, ensure_ascii=False),
        "kept_anchor_tokens_json": json.dumps(kept_anchor_tokens_sorted, ensure_ascii=False),
        "kept_vi_names_json": json.dumps(kept_vi_names_sorted, ensure_ascii=False),
    }
    return x_train_kept, x_apply_kept, stats


def _stable_anchor_local_pruning(
    *,
    payload: dict[str, Any],
    input_variant: str,
    feature_spec_df: pd.DataFrame,
    pruning_cfg: dict[str, Any],
    predictor: str,
    preprocess_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    seed: int,
    repo_root: Path | None = None,
    fusion_cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    variant = str(input_variant).strip().upper()
    if variant not in REDUCED_H_VARIANTS:
        return payload, {}

    inner_payload = payload["inner_payload"]
    x_test = payload["x_test"]
    if not inner_payload:
        return payload, {"pruning_enabled": False, "reason": "empty_inner_payload"}

    stability_enabled = bool(pruning_cfg.get("stability_enabled", True))
    if not stability_enabled or len(inner_payload) <= 1:
        return _prune_outer_payload_for_anchor_local(
            payload=payload,
            input_variant=variant,
            feature_spec_df=feature_spec_df,
            pruning_cfg=pruning_cfg,
            predictor=predictor,
            preprocess_cfg=preprocess_cfg,
            model_cfg=model_cfg,
            seed=seed,
        )

    group_counter: dict[str, int] = {}
    selected_k_counter: dict[int, int] = {}
    per_split_stats: list[dict[str, Any]] = []

    for inner in inner_payload:
        _, scores_by_k, groups_by_k, _, _ = _evaluate_group_count_candidates(
            x_train=inner["x_train"],
            y_train=inner["y_train"],
            x_val=inner["x_val"],
            y_val=inner["y_val"],
            input_variant=variant,
            feature_spec_df=feature_spec_df,
            pruning_cfg=pruning_cfg,
            predictor=predictor,
            preprocess_cfg=preprocess_cfg,
            model_cfg=model_cfg,
            seed=int(seed) + int(inner["inner_fold"]),
            inner_ctx=inner,
            repo_root=repo_root,
            fusion_cfg=fusion_cfg,
        )
        if not scores_by_k:
            continue
        ranked_k = sorted(scores_by_k.items(), key=lambda x: (-x[1], x[0]))
        best_k, best_score = ranked_k[0]
        tie_tolerance = float(pruning_cfg.get("group_selection_tie_tolerance", 1.0e-6))
        for cand_k, cand_score in ranked_k[1:]:
            if abs(cand_score - best_score) <= tie_tolerance and cand_k < best_k:
                best_k = cand_k
        inner_h_spec = feature_spec_df[feature_spec_df["feature"].isin(inner["x_train"].columns)].copy()
        if not inner_h_spec.empty:
            inner_h_spec["anchor_vi_group"] = (
                inner_h_spec["anchor_token"].astype(str).fillna("na_anchor")
                + "||"
                + inner_h_spec["vi_name"].astype(str).fillna("na_vi")
            )
        inner_group_meta = inner_h_spec[["anchor_vi_group", "phase"]].drop_duplicates() if not inner_h_spec.empty else pd.DataFrame(columns=["anchor_vi_group", "phase"])
        requested_min_per_phase = max(
            0,
            int(pruning_cfg.get("group_min_per_phase", pruning_cfg.get("anchor_min_per_phase", 0))),
        )
        effective_min_per_phase = _effective_min_per_phase_for_k(
            inner_group_meta,
            requested_min_per_phase=requested_min_per_phase,
            n_keep=int(best_k),
        )
        selected_k_counter[int(best_k)] = selected_k_counter.get(int(best_k), 0) + 1
        keep_groups = groups_by_k[int(best_k)]
        for group_id in sorted(keep_groups):
            group_counter[group_id] = group_counter.get(group_id, 0) + 1
        per_split_stats.append(
            {
                "inner_fold": int(inner["inner_fold"]),
                "selected_k": int(best_k),
                "selected_score": float(best_score),
                "group_min_per_phase_requested": int(requested_min_per_phase),
                "group_min_per_phase_effective": int(effective_min_per_phase),
                "candidate_scores_json": json.dumps(scores_by_k, ensure_ascii=False, sort_keys=True),
                "n_group_kept": int(len(keep_groups)),
            }
        )

    min_group_freq = max(1, int(pruning_cfg.get("group_min_frequency", max(1, len(inner_payload) // 2))))
    keep_groups = {group_id for group_id, cnt in group_counter.items() if int(cnt) >= min_group_freq}

    h_spec = feature_spec_df[feature_spec_df["feature"].isin(payload["x_test"].columns)].copy()
    if not h_spec.empty:
        h_spec["anchor_vi_group"] = (
            h_spec["anchor_token"].astype(str).fillna("na_anchor") + "||" + h_spec["vi_name"].astype(str).fillna("na_vi")
        )
    min_h_features = max(0, int(pruning_cfg.get("min_h_features", 32)))
    ranked_groups = [str(group_id) for group_id, _ in sorted(group_counter.items(), key=lambda x: (-x[1], x[0]))]
    keep_groups, keep_h_cols = _extend_groups_to_feature_floor(
        keep_groups=keep_groups,
        h_spec=h_spec,
        ranked_groups=ranked_groups,
        min_h_features=min_h_features,
    )
    kept_h_spec = h_spec[h_spec["anchor_vi_group"].astype(str).isin(keep_groups)].copy()

    h_col_set = set(h_spec["feature"].astype(str).tolist())
    g_cols = [col for col in payload["x_test"].columns if col not in h_col_set]
    keep_set = set(g_cols) | set(keep_h_cols)
    keep_cols = [col for col in payload["x_test"].columns if col in keep_set]

    new_inner_payload = []
    for inner in inner_payload:
        new_inner = dict(inner)
        new_inner["x_train"] = inner["x_train"].loc[:, keep_cols].copy()
        new_inner["x_val"] = inner["x_val"].loc[:, keep_cols].copy()
        new_inner_payload.append(new_inner)

    new_payload = dict(payload)
    new_payload["x_test"] = payload["x_test"].loc[:, keep_cols].copy()
    new_payload["inner_payload"] = new_inner_payload

    keep_anchor_count = int(kept_h_spec["anchor_token"].astype(str).nunique()) if not kept_h_spec.empty else 0
    keep_vi_count = int(kept_h_spec["vi_name"].astype(str).nunique()) if not kept_h_spec.empty else 0
    kept_groups_sorted = sorted(str(x) for x in keep_groups)
    kept_anchor_tokens_sorted = (
        sorted(kept_h_spec["anchor_token"].dropna().astype(str).unique().tolist()) if not kept_h_spec.empty else []
    )
    kept_vi_names_sorted = (
        sorted(kept_h_spec["vi_name"].dropna().astype(str).unique().tolist()) if not kept_h_spec.empty else []
    )
    requested_min_per_phase = max(
        0,
        int(pruning_cfg.get("group_min_per_phase", pruning_cfg.get("anchor_min_per_phase", 0))),
    )
    selected_k_mode = int(sorted(selected_k_counter.items(), key=lambda x: (-x[1], x[0]))[0][0]) if selected_k_counter else 0
    group_phase_meta = h_spec[["anchor_vi_group", "phase"]].drop_duplicates() if not h_spec.empty else pd.DataFrame(columns=["anchor_vi_group", "phase"])
    effective_min_per_phase_mode = _effective_min_per_phase_for_k(
        group_phase_meta,
        requested_min_per_phase=requested_min_per_phase,
        n_keep=selected_k_mode,
    )

    stats = {
        "pruning_enabled": True,
        "stability_tag": "outer_train_stability_selection",
        "score_target": "y_direct_g_aware_blend_1.00",
        "selection_unit": "anchor_x_vi_group",
        "n_anchor_candidates": int(h_spec["anchor_token"].astype(str).nunique()),
        "n_vi_candidates": int(h_spec["vi_name"].astype(str).nunique()),
        "n_group_candidates": int(h_spec["anchor_vi_group"].astype(str).nunique()),
        "n_group_kept": int(len(keep_groups)),
        "n_anchor_kept": int(keep_anchor_count),
        "n_vi_kept": int(keep_vi_count),
        "n_h_features_before": int(len(h_col_set)),
        "n_h_features_after": int(len(keep_h_cols)),
        "n_total_features_after": int(len(keep_cols)),
        "group_min_frequency": int(min_group_freq),
        "inner_selection_runs": int(len(inner_payload)),
        "selected_k_mode": int(selected_k_mode),
        "group_min_per_phase_requested": int(requested_min_per_phase),
        "group_min_per_phase_effective_mode": int(effective_min_per_phase_mode),
        "selected_k_counter_json": json.dumps(selected_k_counter, ensure_ascii=False, sort_keys=True),
        "per_split_stats_json": json.dumps(per_split_stats, ensure_ascii=False),
        "candidate_scores_json": json.dumps({}, ensure_ascii=False),
        "kept_group_ids_json": json.dumps(kept_groups_sorted, ensure_ascii=False),
        "kept_anchor_tokens_json": json.dumps(kept_anchor_tokens_sorted, ensure_ascii=False),
        "kept_vi_names_json": json.dumps(kept_vi_names_sorted, ensure_ascii=False),
        "kept_columns": keep_cols,
    }
    return new_payload, stats


def _select_best_auto_window_payload(
    *,
    payloads_by_radius: dict[int, dict[str, Any]],
    feature_specs_by_radius: dict[int, pd.DataFrame],
    input_variant: str,
    pruning_cfg: dict[str, Any],
    predictor: str,
    preprocess_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    seed: int,
    repo_root: Path | None = None,
    fusion_cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    variant = str(input_variant).strip().upper()
    if variant not in AUTO_WINDOW_VARIANTS:
        raise ValueError(f"非 AUTO variant 不应调用自动窗口选择: {input_variant}")

    tie_tolerance = float(pruning_cfg.get("window_selection_tie_tolerance", 1.0e-6))
    metric_name = str(pruning_cfg.get("group_selection_metric", "pearson"))
    candidate_rows: list[dict[str, Any]] = []
    shared_g_cache_by_inner: dict[str, dict[str, Any]] = {}
    if bool((fusion_cfg or {}).get("enabled", False)) and bool((fusion_cfg or {}).get("g_use_gblup", False)):
        for payload in payloads_by_radius.values():
            for inner in payload.get("inner_payload", []):
                cache_key = str(inner.get("split_name", inner.get("inner_fold", "")))
                inner["_g_branch_cache"] = shared_g_cache_by_inner.setdefault(cache_key, {})

    for radius in sorted(payloads_by_radius.keys()):
        payload = payloads_by_radius[int(radius)]
        feature_spec_df = feature_specs_by_radius[int(radius)]
        pruned_payload, pruning_stats = _stable_anchor_local_pruning(
            payload=payload,
            input_variant="G+H_ANCHOR_VI" if variant == "G+H_ANCHOR_AUTO" and int(radius) == 0 else (
                "H_ANCHOR_VI" if variant == "H_ANCHOR_AUTO" and int(radius) == 0 else (
                    "G+H_ANCHOR_LOCAL" if variant == "G+H_ANCHOR_AUTO" else "H_ANCHOR_LOCAL"
                )
            ),
            feature_spec_df=feature_spec_df,
            pruning_cfg=pruning_cfg,
            predictor=predictor,
            preprocess_cfg=preprocess_cfg,
            model_cfg=model_cfg,
            seed=int(seed) + int(radius) * 1000,
            repo_root=repo_root,
            fusion_cfg=fusion_cfg,
        )
        inner_payload = pruned_payload["inner_payload"]
        if not inner_payload:
            continue

        use_safe_fusion = _use_safe_fusion_for_variant(
            input_variant="G+H_ANCHOR_VI" if variant == "G+H_ANCHOR_AUTO" and int(radius) == 0 else (
                "H_ANCHOR_VI" if variant == "H_ANCHOR_AUTO" and int(radius) == 0 else (
                    "G+H_ANCHOR_LOCAL" if variant == "G+H_ANCHOR_AUTO" else "H_ANCHOR_LOCAL"
                )
            ),
            fusion_cfg=fusion_cfg or {},
        )
        use_gblup_for_g = bool((fusion_cfg or {}).get("g_use_gblup", False))
        val_metrics_list: list[dict[str, Any]] = []

        params = _default_params_for_predictor(
            predictor,
            n_features=max(1, int(pruned_payload["x_test"].shape[1])),
            model_cfg=model_cfg,
        )

        for step_i, inner in enumerate(inner_payload):
            if use_safe_fusion:
                safe_inner = _fit_predict_safe_fusion(
                    inner_payload=[inner],
                    x_test=inner["x_val"],
                    y_test=inner["y_val"],
                    meta_test=inner["meta_val"],
                    predictor=predictor,
                    params=params,
                    preprocess_cfg=preprocess_cfg,
                    model_cfg=model_cfg,
                    feature_spec_df=feature_spec_df,
                    seed=int(seed) + int(radius) * 1000 + step_i,
                    fusion_cfg=fusion_cfg or {},
                    repo_root=repo_root or Path.cwd(),
                    use_gblup_for_g=use_gblup_for_g,
                )
                val_metrics = safe_inner["mean_val_metrics"]
            else:
                pipe = build_pipeline(
                    predictor=predictor,
                    params=params,
                    n_features=inner["x_train"].shape[1],
                    preprocess_cfg=preprocess_cfg,
                    seed=int(seed) + int(radius) * 1000 + step_i,
                    model_cfg=model_cfg,
                )
                pipe.fit(inner["x_train"], inner["y_train"])
                pred_val = pipe.predict(inner["x_val"])
                val_metrics = regression_metrics(inner["y_val"].to_numpy(dtype=float), pred_val)
            val_metrics_list.append(val_metrics)

        if not val_metrics_list:
            continue
        mean_val_metrics = mean_regression_metrics(val_metrics_list)
        candidate_rows.append(
            {
                "radius": int(radius),
                "score": float(_extract_selection_score(mean_val_metrics, metric_name)),
                "mean_val_pearson": float(mean_val_metrics.get("pearson_r", float("nan"))),
                "mean_val_rmse": float(mean_val_metrics.get("rmse", float("nan"))),
                "mean_val_r2": float(mean_val_metrics.get("r2", float("nan"))),
                "payload": pruned_payload,
                "pruning_stats": pruning_stats,
            }
        )

    if not candidate_rows:
        raise RuntimeError("AUTO 窗口选择失败：没有可用候选半径。")

    candidate_rows = sorted(candidate_rows, key=lambda x: (-x["score"], x["radius"]))
    best = candidate_rows[0]
    for cand in candidate_rows[1:]:
        if abs(float(cand["score"]) - float(best["score"])) <= tie_tolerance and int(cand["radius"]) < int(best["radius"]):
            best = cand

    summary_rows = [
        {
            "radius": int(row["radius"]),
            "score": float(row["score"]),
            "mean_val_pearson": float(row["mean_val_pearson"]),
            "mean_val_rmse": float(row["mean_val_rmse"]),
            "mean_val_r2": float(row["mean_val_r2"]),
        }
        for row in candidate_rows
    ]
    pruning_stats = {
        **dict(best["pruning_stats"]),
        "selected_window_radius": int(best["radius"]),
        "selected_window_mode": int(best["radius"]),
        "window_candidate_scores_json": json.dumps(summary_rows, ensure_ascii=False),
    }
    return dict(best["payload"]), pruning_stats, int(best["radius"])


def _apply_cached_feature_selection_to_payload(
    *,
    payload: dict[str, Any],
    cached_stats: dict[str, Any],
    current_target_year: int,
    current_outer_fold: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    keep_cols = [str(col) for col in cached_stats.get("kept_columns", [])]
    if not keep_cols:
        raise RuntimeError("无法复用 H-reduce 选择：缓存中缺少 kept_columns。")

    available_cols = set(str(col) for col in payload["x_test"].columns)
    missing_cols = [col for col in keep_cols if col not in available_cols]
    if missing_cols:
        raise RuntimeError(
            "无法复用 H-reduce 选择：当前 outer-fold 缺少缓存列，"
            f"例如 {missing_cols[:5]}。"
        )

    new_inner_payload = []
    for inner in payload["inner_payload"]:
        new_inner = dict(inner)
        new_inner["x_train"] = inner["x_train"].loc[:, keep_cols].copy()
        new_inner["x_val"] = inner["x_val"].loc[:, keep_cols].copy()
        new_inner_payload.append(new_inner)

    new_payload = dict(payload)
    new_payload["x_test"] = payload["x_test"].loc[:, keep_cols].copy()
    new_payload["inner_payload"] = new_inner_payload

    stats = dict(cached_stats)
    stats["selection_reused_from_first_outer_fold"] = True
    stats["selection_search_performed"] = False
    stats["selection_current_target_year"] = int(current_target_year)
    stats["selection_current_outer_fold"] = int(current_outer_fold)
    stats["n_total_features_after"] = int(len(keep_cols))
    return new_payload, stats


def _prune_outer_payload_for_anchor_local(
    *,
    payload: dict[str, Any],
    input_variant: str,
    feature_spec_df: pd.DataFrame,
    pruning_cfg: dict[str, Any],
    predictor: str,
    preprocess_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    seed: int,
    repo_root: Path | None = None,
    fusion_cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    variant = str(input_variant).strip().upper()
    if variant not in REDUCED_H_VARIANTS:
        return payload, {}

    x_test = payload["x_test"]
    y_test = payload["y_test"]
    inner_payload = payload["inner_payload"]
    if not inner_payload:
        return payload, {"pruning_enabled": False, "reason": "empty_inner_payload"}

    first_inner = inner_payload[0]
    group_scores, scores_by_k, groups_by_k, _, h_spec = _evaluate_group_count_candidates(
        x_train=first_inner["x_train"],
        y_train=first_inner["y_train"],
        x_val=first_inner["x_val"],
        y_val=first_inner["y_val"],
        input_variant=variant,
        feature_spec_df=feature_spec_df,
        pruning_cfg=pruning_cfg,
        predictor=predictor,
        preprocess_cfg=preprocess_cfg,
        model_cfg=model_cfg,
        seed=seed,
        inner_ctx=first_inner,
        repo_root=repo_root,
        fusion_cfg=fusion_cfg,
    )
    if not scores_by_k:
        return payload, {"pruning_enabled": False, "reason": "empty_group_k_scores"}
    ranked_k = sorted(scores_by_k.items(), key=lambda x: (-x[1], x[0]))
    best_k, best_score = ranked_k[0]
    tie_tolerance = float(pruning_cfg.get("group_selection_tie_tolerance", 1.0e-6))
    for cand_k, cand_score in ranked_k[1:]:
        if abs(cand_score - best_score) <= tie_tolerance and cand_k < best_k:
            best_k = cand_k

    keep_groups = groups_by_k[int(best_k)]
    if not h_spec.empty:
        h_spec["anchor_vi_group"] = (
            h_spec["anchor_token"].astype(str).fillna("na_anchor") + "||" + h_spec["vi_name"].astype(str).fillna("na_vi")
        )
    ranked_groups = group_scores.sort_values(["score", "anchor_vi_group"], ascending=[False, True])[
        "anchor_vi_group"
    ].astype(str).tolist()
    keep_groups, keep_h_cols = _extend_groups_to_feature_floor(
        keep_groups=keep_groups,
        h_spec=h_spec,
        ranked_groups=ranked_groups,
        min_h_features=max(0, int(pruning_cfg.get("min_h_features", 0))),
    )
    h_col_set = set(h_spec["feature"].astype(str).tolist())
    g_cols = [col for col in x_test.columns if col not in h_col_set]
    keep_set = set(g_cols) | set(keep_h_cols)
    keep_cols = [col for col in x_test.columns if col in keep_set]
    apply_frames = [x_test.loc[:, keep_cols].copy()]

    new_inner_payload = []
    for inner in inner_payload:
        new_inner = dict(inner)
        new_inner["x_train"] = inner["x_train"].loc[:, keep_cols].copy()
        new_inner["x_val"] = inner["x_val"].loc[:, keep_cols].copy()
        new_inner_payload.append(new_inner)

    new_payload = dict(payload)
    new_payload["x_test"] = apply_frames[0]
    new_payload["inner_payload"] = new_inner_payload
    kept_h_spec = h_spec[h_spec["anchor_vi_group"].astype(str).isin(keep_groups)].copy()
    kept_groups_sorted = sorted(str(x) for x in keep_groups)
    kept_anchor_tokens_sorted = (
        sorted(kept_h_spec["anchor_token"].dropna().astype(str).unique().tolist()) if not kept_h_spec.empty else []
    )
    kept_vi_names_sorted = (
        sorted(kept_h_spec["vi_name"].dropna().astype(str).unique().tolist()) if not kept_h_spec.empty else []
    )

    pruning_stats = {
        "pruning_enabled": True,
        "score_target": "y_direct_g_aware_blend_1.00",
        "selection_unit": "anchor_x_vi_group",
        "stability_tag": "single_inner_fold_selection",
        "n_anchor_candidates": int(h_spec["anchor_token"].astype(str).nunique()),
        "n_vi_candidates": int(h_spec["vi_name"].astype(str).nunique()),
        "n_group_candidates": int(h_spec["anchor_vi_group"].astype(str).nunique()),
        "n_group_kept": int(len(keep_groups)),
        "n_anchor_kept": int(kept_h_spec["anchor_token"].astype(str).nunique()) if not kept_h_spec.empty else 0,
        "n_vi_kept": int(kept_h_spec["vi_name"].astype(str).nunique()) if not kept_h_spec.empty else 0,
        "n_h_features_before": int(len(h_col_set)),
        "n_h_features_after": int(len(keep_h_cols)),
        "n_total_features_after": int(len(keep_cols)),
        "selected_k_mode": int(best_k),
        "selected_k_counter_json": json.dumps({int(best_k): 1}, ensure_ascii=False, sort_keys=True),
        "candidate_scores_json": json.dumps(scores_by_k, ensure_ascii=False, sort_keys=True),
        "kept_group_ids_json": json.dumps(kept_groups_sorted, ensure_ascii=False),
        "kept_anchor_tokens_json": json.dumps(kept_anchor_tokens_sorted, ensure_ascii=False),
        "kept_vi_names_json": json.dumps(kept_vi_names_sorted, ensure_ascii=False),
        "kept_columns": keep_cols,
    }
    return new_payload, pruning_stats


def _use_safe_fusion_for_variant(
    *,
    input_variant: str,
    fusion_cfg: dict[str, Any],
) -> bool:
    variant = str(input_variant).strip().upper()
    if variant not in G_H_FUSION_VARIANTS:
        return False
    return bool(fusion_cfg.get("enabled", False))


def _aggregate_summary(metrics_fold_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_fold_df.empty:
        return pd.DataFrame()
    group_cols = ["scenario", "timeline", "target", "input_variant", "predictor"]
    for col in ["anchor_order", "anchor_idx", "anchor_tb", "anchor_phase", "anchor_token"]:
        if col in metrics_fold_df.columns and pd.Series(metrics_fold_df[col]).notna().any():
            group_cols.append(col)
    return (
        metrics_fold_df.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            mean_r2=("test_r2", "mean"),
            std_r2=("test_r2", _safe_std),
            mean_rmse=("test_rmse", "mean"),
            std_rmse=("test_rmse", _safe_std),
            mean_mae=("test_mae", "mean"),
            std_mae=("test_mae", _safe_std),
            mean_pearson=("test_pearson", "mean"),
            std_pearson=("test_pearson", _safe_std),
            mean_train_loss=("best_mean_train_loss", "mean"),
            mean_val_loss=("best_mean_val_loss", "mean"),
            mean_gap=("best_gap", "mean"),
            n_outer_folds=("outer_fold", "nunique"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def _validate_result34_outer_fold_coverage(
    *,
    metrics_fold_df: pd.DataFrame,
    split_usage_df: pd.DataFrame,
) -> None:
    if metrics_fold_df.empty or split_usage_df.empty:
        return

    required_split_cols = {"scenario", "timeline", "target", "status", "n_groups_used"}
    if not required_split_cols.issubset(split_usage_df.columns):
        return
    required_metric_cols = {
        "scenario",
        "timeline",
        "target",
        "input_variant",
        "predictor",
        "target_year",
        "outer_fold",
    }
    if not required_metric_cols.issubset(metrics_fold_df.columns):
        return

    split_ok = split_usage_df.loc[split_usage_df["status"].astype(str) == "ok", :].copy()
    if split_ok.empty:
        return

    expected = (
        split_ok.groupby(["scenario", "timeline", "target"], as_index=False)["n_groups_used"]
        .max()
        .rename(columns={"n_groups_used": "expected_outer_folds"})
    )
    actual_group_cols = ["scenario", "timeline", "target", "input_variant", "predictor"]
    if "anchor_order" in metrics_fold_df.columns and pd.to_numeric(metrics_fold_df["anchor_order"], errors="coerce").notna().any():
        actual_group_cols.append("anchor_order")
    actual = (
        metrics_fold_df.loc[:, actual_group_cols + ["target_year", "outer_fold"]]
        .drop_duplicates()
        .groupby(actual_group_cols, as_index=False)
        .size()
        .rename(columns={"size": "actual_outer_folds"})
    )
    merged = actual.merge(expected, on=["scenario", "timeline", "target"], how="left")
    bad = merged.loc[
        merged["expected_outer_folds"].notna()
        & (pd.to_numeric(merged["actual_outer_folds"], errors="coerce") < pd.to_numeric(merged["expected_outer_folds"], errors="coerce")),
        :,
    ].copy()
    if bad.empty:
        return

    preview_cols = ["scenario", "timeline", "target", "input_variant", "predictor"]
    if "anchor_order" in bad.columns:
        preview_cols.append("anchor_order")
    preview_cols.extend(["actual_outer_folds", "expected_outer_folds"])
    preview = bad.loc[:, preview_cols].head(10).to_dict(orient="records")
    raise RuntimeError(
        "Result34 outer-fold coverage 不完整，疑似只写出了部分 outer folds。"
        f" 示例: {json.dumps(preview, ensure_ascii=False)}"
    )


def _aggregate_by_year(oof_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if oof_df.empty:
        return pd.DataFrame()
    group_cols = ["scenario", "timeline", "target", "input_variant", "predictor", "year"]
    for col in ["anchor_order", "anchor_idx", "anchor_tb", "anchor_phase", "anchor_token"]:
        if col in oof_df.columns and pd.Series(oof_df[col]).notna().any():
            group_cols.append(col)
    for keys, sub in oof_df.groupby(group_cols, dropna=False):
        key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        m = regression_metrics(sub["y_true"].to_numpy(dtype=float), sub["y_pred"].to_numpy(dtype=float))
        row = {col: key_map[col] for col in group_cols}
        row.update(
            {
                "year": int(key_map["year"]),
                "r2": float(m["r2"]),
                "rmse": float(m["rmse"]),
                "mae": float(m["mae"]),
                "pearson": float(m["pearson_r"]),
                "n_samples": int(len(sub)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _aggregate_overview(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    by_scenario_variant = (
        summary_df.groupby(["scenario", "input_variant"], as_index=False)
        .agg(
            avg_r2=("mean_r2", "mean"),
            avg_rmse=("mean_rmse", "mean"),
            avg_mae=("mean_mae", "mean"),
            avg_pearson=("mean_pearson", "mean"),
            avg_std_r2=("std_r2", "mean"),
            avg_std_pearson=("std_pearson", "mean"),
        )
        .sort_values(["scenario", "avg_r2"], ascending=[True, False])
        .reset_index(drop=True)
    )
    by_scenario_target_variant = (
        summary_df.groupby(["scenario", "target", "input_variant"], as_index=False)
        .agg(
            best_r2=("mean_r2", "max"),
            best_pearson=("mean_pearson", "max"),
            avg_r2=("mean_r2", "mean"),
            avg_pearson=("mean_pearson", "mean"),
        )
        .sort_values(["scenario", "target", "input_variant"])
        .reset_index(drop=True)
    )
    return by_scenario_variant, by_scenario_target_variant


def _plot_heatmap(df: pd.DataFrame, *, title: str, index_col: str, column_col: str, value_col: str, out_path: Path) -> None:
    if df.empty:
        return
    pivot = df.pivot_table(index=index_col, columns=column_col, values=value_col, aggfunc="mean")
    rows = list(pivot.index)
    cols = list(pivot.columns)
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 1.2), max(3.5, len(rows) * 0.9)))
    im = ax.imshow(values, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            txt = "nan" if np.isnan(val) else f"{val:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "(empty)"
    part = df.head(max_rows).copy()
    try:
        return part.to_markdown(index=False)
    except Exception:
        return "```text\n" + part.to_string(index=False) + "\n```"


def _write_summary(
    *,
    output_dir: Path,
    cfg: dict,
    representation_overview_df: pd.DataFrame,
    feature_overview_df: pd.DataFrame,
    split_usage_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    by_year_df: pd.DataFrame,
    scenario_variant_df: pd.DataFrame,
    scenario_target_variant_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Result3.4A Summary")
    lines.append("")
    lines.append("## 1. 实验定义")
    lines.append("- 时间轴：`" + "`, `".join(list(_resolve_timeline_dirs(cfg.get("data", {})).keys())) + "`。")
    lines.append("- 输入比较：`" + "`, `".join(_resolve_input_variants(cfg.get("experiment", {}))) + "`。")
    lines.append("- 性状：`" + ", ".join(cfg.get("experiment", {}).get("targets", [])) + "`。")
    lines.append("- 预测器：`" + ", ".join(cfg.get("experiment", {}).get("predictors_run", [])) + "`。")
    lines.append("")
    lines.append("## 2. reduced-H 表示概览")
    lines.append(_markdown_table(representation_overview_df))
    lines.append("")
    lines.append("## 3. 各输入可用性概览")
    lines.append(_markdown_table(feature_overview_df))
    lines.append("")
    lines.append("## 4. split 使用概览")
    lines.append(_markdown_table(split_usage_df))
    lines.append("")
    lines.append("## 5. 场景 x 输入总体结果")
    lines.append(_markdown_table(scenario_variant_df))
    lines.append("")
    lines.append("## 6. 场景 x 性状 x 输入总体结果")
    lines.append(_markdown_table(scenario_target_variant_df))
    lines.append("")
    lines.append("## 7. 明细结果预览")
    lines.append(_markdown_table(summary_df))
    lines.append("")
    lines.append("## 8. 按年份 OOF 聚合结果预览")
    lines.append(_markdown_table(by_year_df))
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def finalize_result34_outputs(
    *,
    output_dir: Path,
    cfg: dict,
    representation_overview_df: pd.DataFrame,
    representation_feature_spec_df: pd.DataFrame,
    feature_overview_df: pd.DataFrame,
    split_usage_df: pd.DataFrame,
    metrics_fold_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    inner_pred_df: pd.DataFrame,
    trial_df: pd.DataFrame,
) -> None:
    summary_df = _aggregate_summary(metrics_fold_df)
    by_year_df = _aggregate_by_year(oof_df)
    scenario_variant_df, scenario_target_variant_df = _aggregate_overview(summary_df)

    representation_overview_df.to_csv(output_dir / "reduced_h_feature_overview.csv", index=False)
    representation_feature_spec_df.to_csv(output_dir / "reduced_h_feature_spec.csv", index=False)
    feature_overview_df.to_csv(output_dir / "feature_overview.csv", index=False)
    split_usage_df.to_csv(output_dir / "split_usage_summary.csv", index=False)
    metrics_fold_df.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    by_year_df.to_csv(output_dir / "metrics_by_year.csv", index=False)
    scenario_variant_df.to_csv(output_dir / "metrics_by_scenario_input_variant.csv", index=False)
    scenario_target_variant_df.to_csv(output_dir / "metrics_by_scenario_target_input_variant.csv", index=False)
    if not oof_df.empty:
        oof_df.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    if not inner_pred_df.empty:
        inner_pred_df.to_parquet(output_dir / "outer_test_inner_ensemble_predictions.parquet", index=False)
    if not trial_df.empty:
        trial_df.to_csv(output_dir / "optuna_trials.csv", index=False)

    fig_dir = output_dir / "figures"
    ensure_dir(fig_dir)
    if not summary_df.empty:
        for (scenario, target), sub in summary_df.groupby(["scenario", "target"], dropna=False):
            part = sub[["input_variant", "predictor", "mean_r2", "mean_pearson"]].copy()
            _plot_heatmap(
                part,
                title=f"{scenario} | {target} | mean R2",
                index_col="input_variant",
                column_col="predictor",
                value_col="mean_r2",
                out_path=fig_dir / f"figure_r2_heatmap_{scenario}_{target}.png",
            )
            _plot_heatmap(
                part,
                title=f"{scenario} | {target} | mean Pearson",
                index_col="input_variant",
                column_col="predictor",
                value_col="mean_pearson",
                out_path=fig_dir / f"figure_pearson_heatmap_{scenario}_{target}.png",
            )

        for scenario, sub in scenario_variant_df.groupby("scenario", dropna=False):
            part = (
                summary_df[summary_df["scenario"].astype(str) == str(scenario)]
                .groupby(["input_variant", "predictor"], as_index=False)
                .agg(mean_r2=("mean_r2", "mean"), mean_pearson=("mean_pearson", "mean"))
            )
            _plot_heatmap(
                part,
                title=f"{scenario} | overall mean R2",
                index_col="input_variant",
                column_col="predictor",
                value_col="mean_r2",
                out_path=fig_dir / f"figure_r2_heatmap_{scenario}_overall.png",
            )
            _plot_heatmap(
                part,
                title=f"{scenario} | overall mean Pearson",
                index_col="input_variant",
                column_col="predictor",
                value_col="mean_pearson",
                out_path=fig_dir / f"figure_pearson_heatmap_{scenario}_overall.png",
            )

    config_snapshot = {
        "config_path": cfg.get("config_path"),
        "git_commit_hash": get_git_commit_hash(Path.cwd()),
        "generated_at": now_iso(),
        "data": cfg.get("data", {}),
        "experiment": cfg.get("experiment", {}),
        "optuna": cfg.get("optuna", {}),
        "preprocessing": cfg.get("preprocessing", {}),
        "output_dir": output_dir.as_posix(),
    }
    write_yaml(config_snapshot, output_dir / "run_config.yaml")
    write_json(config_snapshot, output_dir / "run_config.json")

    _write_summary(
        output_dir=output_dir,
        cfg=cfg,
        representation_overview_df=representation_overview_df,
        feature_overview_df=feature_overview_df,
        split_usage_df=split_usage_df,
        summary_df=summary_df,
        by_year_df=by_year_df,
        scenario_variant_df=scenario_variant_df,
        scenario_target_variant_df=scenario_target_variant_df,
    )


def run_result34(config_path: Path) -> Path:
    cfg = read_yaml(config_path)
    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})
    optuna_cfg = cfg.get("optuna", {})
    preprocess_cfg = cfg.get("preprocessing", {})
    output_cfg = cfg.get("output", {})

    output_dir = _resolve_output_dir(output_cfg)
    progress_enabled = bool(output_cfg.get("progress_log", True))
    _append_progress(output_dir, enabled=progress_enabled, event="run_start", payload={"config_path": str(config_path)})

    seed = int(exp_cfg.get("random_seed", 42))
    _set_global_seed(seed)

    genotype_representation = str(exp_cfg.get("genotype_representation", "grm_pca"))
    targets = [str(x) for x in exp_cfg.get("targets", ["ActualYD", "CM", "LM", "PHM", "Spike", "TKW"])]
    timeline_dirs = _resolve_timeline_dirs(data_cfg)
    scenarios_cfg = _resolve_scenarios_cfg(data_cfg)
    active_predictors = _resolve_active_predictors(exp_cfg)
    input_variants = _resolve_input_variants(exp_cfg)
    task_filter_set = _resolve_task_filter_set(exp_cfg)
    prefix_cfg = _resolve_growth_prefix_cfg(exp_cfg)
    prefix_orders = prefix_cfg["anchor_orders"]
    planned_tasks = enumerate_result34_tasks_from_config(cfg)
    allowed_task_keys = {
        (
            task["scenario"],
            task["timeline"],
            task["target"],
            task["input_variant"],
            task["predictor"],
            _normalize_anchor_order_value(task.get("anchor_order")),
        )
        for task in planned_tasks
    }
    if not planned_tasks:
        raise RuntimeError("没有匹配到可执行的 Result3.4A 任务。")

    n_trials = int(optuna_cfg.get("n_trials", 3))
    lambda_gap = float(optuna_cfg.get("lambda_gap", 1.0))
    pruner_startup_trials = int(optuna_cfg.get("pruner_startup_trials", 1))
    phase_cfg = PhaseStateConfig(
        sow_segments=int(exp_cfg.get("phase_state", {}).get("sow_segments", 4)),
        head_segments=int(exp_cfg.get("phase_state", {}).get("head_segments", 3)),
        enable_phase_internal_tail_fill=bool(exp_cfg.get("phase_state", {}).get("enable_phase_internal_tail_fill", True)),
    )
    fusion_cfg = dict(exp_cfg.get("fusion", {}))
    anchor_local_cfg = AnchorLocalConfig(
        n_anchor_bins=int(exp_cfg.get("n_anchor_bins", exp_cfg.get("anchor_local", {}).get("n_anchor_bins", 20))),
        enable_phase_internal_tail_fill=bool(
            exp_cfg.get("anchor_local", {}).get("enable_phase_internal_tail_fill", True)
        ),
    )
    radius_candidates = _resolve_anchor_window_radius_candidates(exp_cfg)
    anchor_local_pruning_cfg = dict(exp_cfg.get("anchor_local_pruning", {}))
    timeline_bundles: dict[str, MultiTargetDataBundle] = {}
    timeline_phase_features: dict[str, pd.DataFrame] = {}
    timeline_phase_info: dict[str, dict[str, Any]] = {}
    timeline_anchor_local_features: dict[str, pd.DataFrame] = {}
    timeline_anchor_local_info: dict[str, dict[str, Any]] = {}
    timeline_anchor_vi_features: dict[str, pd.DataFrame] = {}
    timeline_anchor_vi_info: dict[str, dict[str, Any]] = {}
    timeline_anchor_auto_features_by_radius: dict[str, dict[int, pd.DataFrame]] = {}
    timeline_anchor_auto_info_by_radius: dict[str, dict[int, dict[str, Any]]] = {}
    timeline_variant_cache: dict[str, dict[str, dict[str, Any]]] = {}
    feature_overview_rows: list[dict[str, Any]] = []
    representation_overview_rows: list[dict[str, Any]] = []
    representation_feature_spec_frames: list[pd.DataFrame] = []

    for timeline_name, timeline_path in timeline_dirs.items():
        print(f"[LOAD] timeline={timeline_name} path={timeline_path}", flush=True)
        bundle = load_multitarget_model_inputs(Path(timeline_path), genotype_representation=genotype_representation)
        x_phase, phase_info = build_h_full_features(bundle, config=phase_cfg)
        radius_feature_cache: dict[int, pd.DataFrame] = {}
        radius_info_cache: dict[int, dict[str, Any]] = {}
        for radius in radius_candidates:
            radius_cfg = AnchorLocalConfig(
                n_anchor_bins=anchor_local_cfg.n_anchor_bins,
                enable_phase_internal_tail_fill=anchor_local_cfg.enable_phase_internal_tail_fill,
                radius=int(radius),
            )
            x_radius, radius_info = build_h_anchor_local_features(bundle, config=radius_cfg)
            radius_feature_cache[int(radius)] = x_radius
            radius_info_cache[int(radius)] = radius_info
        base_radius = 0 if 0 in radius_feature_cache else sorted(radius_feature_cache.keys())[0]
        x_anchor_local = radius_feature_cache[base_radius]
        anchor_local_info = radius_info_cache[base_radius]
        x_anchor_vi, anchor_vi_info = build_h_anchor_vi_features(
            bundle,
            config=AnchorLocalConfig(
                n_anchor_bins=anchor_local_cfg.n_anchor_bins,
                enable_phase_internal_tail_fill=anchor_local_cfg.enable_phase_internal_tail_fill,
                radius=0,
            ),
        )
        phase_info["feature_spec_df"] = attach_anchor_bin_prefix_to_full_h_spec(
            bundle,
            phase_info["feature_spec_df"],
            n_anchor_bins=anchor_local_cfg.n_anchor_bins,
        )
        timeline_bundles[timeline_name] = bundle
        timeline_phase_features[timeline_name] = x_phase
        timeline_phase_info[timeline_name] = phase_info
        timeline_anchor_local_features[timeline_name] = x_anchor_local
        timeline_anchor_local_info[timeline_name] = anchor_local_info
        timeline_anchor_vi_features[timeline_name] = x_anchor_vi
        timeline_anchor_vi_info[timeline_name] = anchor_vi_info
        timeline_anchor_auto_features_by_radius[timeline_name] = radius_feature_cache
        timeline_anchor_auto_info_by_radius[timeline_name] = radius_info_cache
        timeline_variant_cache[timeline_name] = {}
        representation_overview_rows.append(
            {
                "timeline": timeline_name,
                "representation": phase_info["representation"],
                "n_samples": int(phase_info["n_samples"]),
                "n_features_total": int(phase_info["n_features_total"]),
                "n_vi": int(phase_info["n_vi"]),
                "tail_fill_mode": str(phase_info.get("tail_fill_mode", "none")),
            }
        )
        representation_overview_rows.append(
            {
                "timeline": timeline_name,
                "representation": anchor_local_info["representation"],
                "n_samples": int(anchor_local_info["n_samples"]),
                "n_features_total": int(anchor_local_info["n_features_total"]),
                "n_vi": int(anchor_local_info["n_vi"]),
                "n_anchor_bins_requested": int(anchor_local_info.get("n_anchor_bins_requested", 0)),
                "n_anchor_bins_effective": int(anchor_local_info.get("n_anchor_bins_effective", 0)),
                "n_group_candidates": int(anchor_local_info.get("n_group_candidates", 0)),
            }
        )
        representation_overview_rows.append(
            {
                "timeline": timeline_name,
                "representation": anchor_vi_info["representation"],
                "n_samples": int(anchor_vi_info["n_samples"]),
                "n_features_total": int(anchor_vi_info["n_features_total"]),
                "n_vi": int(anchor_vi_info["n_vi"]),
                "n_anchor_bins_requested": int(anchor_vi_info.get("n_anchor_bins_requested", 0)),
                "n_anchor_bins_effective": int(anchor_vi_info.get("n_anchor_bins_effective", 0)),
                "n_group_candidates": int(anchor_vi_info.get("n_group_candidates", 0)),
            }
        )
        spec_df = phase_info["feature_spec_df"].copy()
        if not spec_df.empty:
            spec_df.insert(0, "representation", phase_info["representation"])
            spec_df.insert(0, "timeline", timeline_name)
            representation_feature_spec_frames.append(spec_df)
        anchor_spec_df = anchor_local_info["feature_spec_df"].copy()
        if not anchor_spec_df.empty:
            anchor_spec_df.insert(0, "representation", anchor_local_info["representation"])
            anchor_spec_df.insert(0, "timeline", timeline_name)
            representation_feature_spec_frames.append(anchor_spec_df)
        anchor_vi_spec_df = anchor_vi_info["feature_spec_df"].copy()
        if not anchor_vi_spec_df.empty:
            anchor_vi_spec_df.insert(0, "representation", anchor_vi_info["representation"])
            anchor_vi_spec_df.insert(0, "timeline", timeline_name)
            representation_feature_spec_frames.append(anchor_vi_spec_df)

        for input_variant in input_variants:
            x_variant, info = _build_input_variant_matrix(
                bundle,
                x_phase=x_phase,
                x_anchor_local=x_anchor_local,
                x_anchor_vi=x_anchor_vi,
                input_variant=input_variant,
            )
            timeline_variant_cache[timeline_name][input_variant] = {"x": x_variant, "info": info}
            feature_overview_rows.append(_availability_row(timeline_name, input_variant, x_variant, info))

    total_tasks = len(planned_tasks)
    task_idx = 0
    metrics_fold_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    inner_pred_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    split_usage_rows: list[dict[str, Any]] = []

    task_filters_cfg = exp_cfg.get("task_filters", {}) if isinstance(exp_cfg.get("task_filters"), dict) else {}
    scenario_seed_order = [str(x) for x in task_filters_cfg.get("scenario_order", list(scenarios_cfg.keys()))]
    timeline_seed_order = [str(x) for x in task_filters_cfg.get("timeline_order", list(timeline_dirs.keys()))]
    target_seed_order = [str(x) for x in task_filters_cfg.get("target_order", targets)]
    variant_seed_order = [str(x).upper() for x in task_filters_cfg.get("input_variant_order", input_variants)]
    predictor_seed_order = [str(x).lower() for x in task_filters_cfg.get("predictor_order", active_predictors)]
    prefix_seed_order = [_normalize_anchor_order_value(x) for x in task_filters_cfg.get("anchor_order_order", prefix_orders)]

    for scenario_name, scenario_cfg in scenarios_cfg.items():
        for timeline_name in timeline_dirs.keys():
            bundle = timeline_bundles[timeline_name]
            meta_full = bundle.meta
            raw_split_groups = _resolve_split_groups_for_scenario(
                scenario_name=scenario_name,
                scenario_cfg=scenario_cfg,
                meta_df=meta_full,
                random_seed=seed,
            )

            for target_col in targets:
                if target_col not in bundle.y_df.columns:
                    split_usage_rows.append(
                        {"scenario": scenario_name, "timeline": timeline_name, "target": target_col, "status": "missing_target_column"}
                    )
                    continue
                y_full = bundle.y_df[target_col].dropna().copy()
                available_ids = set(meta_full.index.intersection(y_full.index))
                prepared_groups, split_stats = _prepare_split_groups(raw_split_groups, available_ids)
                split_usage_rows.append(
                    {
                        "scenario": scenario_name,
                        "timeline": timeline_name,
                        "target": target_col,
                        "status": "ok" if prepared_groups else "no_valid_split_group",
                        **split_stats,
                        "n_target_samples": int(len(available_ids)),
                    }
                )
                if not prepared_groups:
                    continue

                for input_variant in input_variants:
                    is_auto_variant = str(input_variant).strip().upper() in AUTO_WINDOW_VARIANTS
                    if is_auto_variant:
                        x_base = pd.DataFrame(index=meta_full.index)
                        x_info_base = {
                            "input_variant": input_variant,
                            "n_features_total": 0,
                            "n_features_h_full": 0,
                            "n_features_h_anchor_local": 0,
                            "n_features_h_anchor_vi": 0,
                            "n_features_h_anchor_auto": 0,
                            "n_features_h_phase": 0,
                            "n_features_g": 0,
                        }
                        feature_spec_base = pd.DataFrame()
                    else:
                        variant_cache = timeline_variant_cache[timeline_name][input_variant]
                        x_base = variant_cache["x"]
                        x_info_base = variant_cache["info"]
                        feature_spec_base = _resolve_feature_spec_df_for_variant(
                            timeline_phase_info=timeline_phase_info,
                            timeline_anchor_local_info=timeline_anchor_local_info,
                            timeline_anchor_vi_info=timeline_anchor_vi_info,
                            timeline_name=timeline_name,
                            input_variant=input_variant,
                        )
                    anchor_prefix_source_spec = timeline_anchor_local_info[timeline_name]["feature_spec_df"]
                    for predictor in active_predictors:
                        for anchor_order in prefix_orders:
                            task_key = (scenario_name, timeline_name, target_col, input_variant, predictor, anchor_order)
                            if task_filter_set and task_key not in allowed_task_keys:
                                continue

                            if is_auto_variant:
                                auto_candidates: dict[int, dict[str, Any]] = {}
                                auto_feature_specs: dict[int, pd.DataFrame] = {}
                                for radius in radius_candidates:
                                    x_h_radius = timeline_anchor_auto_features_by_radius[timeline_name][int(radius)]
                                    radius_info = timeline_anchor_auto_info_by_radius[timeline_name][int(radius)]
                                    x_candidate_base, info_candidate_base = _build_auto_input_variant_matrix(
                                        bundle,
                                        x_h=x_h_radius,
                                        input_variant=input_variant,
                                        radius=int(radius),
                                    )
                                    feature_spec_candidate = radius_info["feature_spec_df"]
                                    x_candidate, info_candidate, spec_candidate = _limit_variant_to_anchor_prefix(
                                        x=x_candidate_base,
                                        info=info_candidate_base,
                                        feature_spec_df=feature_spec_candidate,
                                        input_variant=input_variant,
                                        anchor_order=anchor_order,
                                    )
                                    spec_candidate = _filter_feature_spec_to_columns(spec_candidate, list(x_candidate.columns))
                                    auto_candidates[int(radius)] = {
                                        "x": x_candidate,
                                        "info": info_candidate,
                                        "feature_spec": spec_candidate,
                                    }
                                    auto_feature_specs[int(radius)] = spec_candidate
                                first_radius = sorted(auto_candidates.keys())[0]
                                x_full = auto_candidates[first_radius]["x"]
                                x_info = auto_candidates[first_radius]["info"]
                                feature_spec_df = auto_candidates[first_radius]["feature_spec"]
                            else:
                                x_full, x_info, feature_spec_df = _limit_variant_to_anchor_prefix(
                                    x=x_base,
                                    info=x_info_base,
                                    feature_spec_df=feature_spec_base,
                                    input_variant=input_variant,
                                    anchor_order=anchor_order,
                                )
                                feature_spec_df = _filter_feature_spec_to_columns(feature_spec_df, list(x_full.columns))
                            anchor_prefix_meta = _anchor_prefix_metadata(anchor_order, anchor_prefix_source_spec)
                            task_idx += 1
                            feature_count_log = str(x_full.shape[1])
                            prefix_log = "full" if anchor_order is None else str(anchor_order)
                            print(
                                f"[TASK {task_idx}/{total_tasks}] scenario={scenario_name} tl={timeline_name} target={target_col} variant={input_variant} predictor={predictor} anchor_order={prefix_log} features={feature_count_log}",
                                flush=True,
                            )
                            _append_progress(
                                output_dir,
                                enabled=progress_enabled,
                                event="task_start",
                                payload={
                                    "task_idx": task_idx,
                                    "total_tasks": total_tasks,
                                    "scenario": scenario_name,
                                    "timeline": timeline_name,
                                    "target": target_col,
                                    "input_variant": input_variant,
                                    "predictor": predictor,
                                    "anchor_order": anchor_prefix_meta["anchor_order"],
                                    "anchor_idx": anchor_prefix_meta["anchor_idx"],
                                    "anchor_tb": anchor_prefix_meta["anchor_tb"],
                                    "anchor_phase": anchor_prefix_meta["anchor_phase"],
                                    "n_features": feature_count_log,
                                },
                            )

                            ordered_outer_items = normalize_outer_groups(prepared_groups)
                            ordered_outer_keys = [key for key, _ in ordered_outer_items]
                            reuse_policy = resolve_optuna_reuse_policy(optuna_cfg)
                            cached_best_params = None
                            cached_source = None
                            cached_feature_selection: dict[str, Any] | None = None
                            cached_feature_selection_radius: int | None = None
                            cached_feature_selection_info: dict[str, Any] | None = None
                            cached_feature_selection_spec: pd.DataFrame | None = None

                            for (target_year, outer_fold), group_splits in ordered_outer_items:
                                is_feature_reduction_variant = str(input_variant).strip().upper() in REDUCED_H_VARIANTS
                                feature_selection_search_performed = (
                                    is_feature_reduction_variant and cached_feature_selection is None
                                )
                                if is_auto_variant:
                                    if feature_selection_search_performed:
                                        payloads_by_radius = {
                                            int(radius): _build_outer_payload(
                                                auto_candidates[int(radius)]["x"],
                                                y_full,
                                                meta_full,
                                                group_splits,
                                            )
                                            for radius in auto_candidates.keys()
                                        }
                                        payload, pruning_stats, selected_radius = _select_best_auto_window_payload(
                                            payloads_by_radius=payloads_by_radius,
                                            feature_specs_by_radius=auto_feature_specs,
                                            input_variant=input_variant,
                                            pruning_cfg=anchor_local_pruning_cfg,
                                            predictor=predictor,
                                            preprocess_cfg=preprocess_cfg,
                                            model_cfg=exp_cfg.get("model_backends", {}),
                                            seed=int(seed) + int(target_year % 100) * 10 + int(outer_fold),
                                            repo_root=Path.cwd(),
                                            fusion_cfg=fusion_cfg,
                                        )
                                        x_info = dict(auto_candidates[int(selected_radius)]["info"])
                                        feature_spec_df = auto_candidates[int(selected_radius)]["feature_spec"]
                                        pruning_stats = {
                                            **pruning_stats,
                                            "selection_reused_from_first_outer_fold": False,
                                            "selection_search_performed": True,
                                            "selection_source_target_year": int(target_year),
                                            "selection_source_outer_fold": int(outer_fold),
                                        }
                                        cached_feature_selection = dict(pruning_stats)
                                        cached_feature_selection_radius = int(selected_radius)
                                        cached_feature_selection_info = dict(x_info)
                                        cached_feature_selection_spec = feature_spec_df.copy()
                                    else:
                                        if (
                                            cached_feature_selection is None
                                            or cached_feature_selection_radius is None
                                            or cached_feature_selection_info is None
                                            or cached_feature_selection_spec is None
                                        ):
                                            raise RuntimeError("AUTO H-reduce 复用失败：缺少首个 outer-fold 的选择缓存。")
                                        selected_radius = int(cached_feature_selection_radius)
                                        payload_raw = _build_outer_payload(
                                            auto_candidates[selected_radius]["x"],
                                            y_full,
                                            meta_full,
                                            group_splits,
                                        )
                                        payload, pruning_stats = _apply_cached_feature_selection_to_payload(
                                            payload=payload_raw,
                                            cached_stats=cached_feature_selection,
                                            current_target_year=int(target_year),
                                            current_outer_fold=int(outer_fold),
                                        )
                                        x_info = dict(cached_feature_selection_info)
                                        feature_spec_df = cached_feature_selection_spec.copy()
                                else:
                                    payload_raw = _build_outer_payload(x_full, y_full, meta_full, group_splits)
                                    if feature_selection_search_performed:
                                        payload, pruning_stats = _stable_anchor_local_pruning(
                                            payload=payload_raw,
                                            input_variant=input_variant,
                                            feature_spec_df=feature_spec_df,
                                            pruning_cfg=anchor_local_pruning_cfg,
                                            predictor=predictor,
                                            preprocess_cfg=preprocess_cfg,
                                            model_cfg=exp_cfg.get("model_backends", {}),
                                            seed=int(seed) + int(target_year % 100) * 10 + int(outer_fold),
                                            repo_root=Path.cwd(),
                                            fusion_cfg=fusion_cfg,
                                        )
                                        pruning_stats = {
                                            **pruning_stats,
                                            "selection_reused_from_first_outer_fold": False,
                                            "selection_search_performed": True,
                                            "selection_source_target_year": int(target_year),
                                            "selection_source_outer_fold": int(outer_fold),
                                        }
                                        cached_feature_selection = dict(pruning_stats)
                                        cached_feature_selection_info = dict(x_info)
                                        cached_feature_selection_spec = feature_spec_df.copy()
                                    elif is_feature_reduction_variant:
                                        if cached_feature_selection is None:
                                            raise RuntimeError("H-reduce 复用失败：缺少首个 outer-fold 的选择缓存。")
                                        payload, pruning_stats = _apply_cached_feature_selection_to_payload(
                                            payload=payload_raw,
                                            cached_stats=cached_feature_selection,
                                            current_target_year=int(target_year),
                                            current_outer_fold=int(outer_fold),
                                        )
                                        if cached_feature_selection_info is not None:
                                            x_info = dict(cached_feature_selection_info)
                                        if cached_feature_selection_spec is not None:
                                            feature_spec_df = cached_feature_selection_spec.copy()
                                    else:
                                        payload = payload_raw
                                        pruning_stats = {}
                                inner_payload = payload["inner_payload"]
                                x_test = payload["x_test"]
                                y_test = payload["y_test"]
                                meta_test = payload["meta_test"]
                                if len(x_test) == 0:
                                    continue

                                s_idx = scenario_seed_order.index(scenario_name)
                                t_idx = timeline_seed_order.index(timeline_name)
                                g_idx = target_seed_order.index(target_col)
                                v_idx = variant_seed_order.index(input_variant)
                                p_idx = predictor_seed_order.index(predictor)
                                a_idx = prefix_seed_order.index(anchor_order) if anchor_order in prefix_seed_order else 0
                                # Full-prefix 3.4A must be directly comparable with unprefixed 3.3B.
                                # Earlier prefixes keep distinct sampler streams; the final prefix does not.
                                prefix_seed_offset = 0
                                if anchor_order is not None and int(anchor_order) < int(prefix_cfg["n_anchor_bins"]):
                                    prefix_seed_offset = int(a_idx) * 10_000_000
                                sampler_seed = (
                                    seed
                                    + s_idx * 1_000_000
                                    + t_idx * 100_000
                                    + g_idx * 10_000
                                    + v_idx * 1_000
                                    + p_idx * 100
                                    + prefix_seed_offset
                                    + int(target_year % 100) * 10
                                    + int(outer_fold)
                                )

                                use_safe_fusion = _use_safe_fusion_for_variant(
                                    input_variant=input_variant,
                                    fusion_cfg=fusion_cfg,
                                )
                                use_gblup_for_g = bool(fusion_cfg.get("g_use_gblup", False))
                                g_only_gblup = str(input_variant).strip().upper() in G_ONLY_VARIANTS and use_gblup_for_g
                                ridge_grid_cfg = (
                                    _resolve_ridge_alpha_grid_cfg(optuna_cfg)
                                    if predictor == "ridge" and not g_only_gblup
                                    else None
                                )
                                n_trials_effective = (
                                    len(ridge_grid_cfg["values"])
                                    if ridge_grid_cfg is not None
                                    else (1 if g_only_gblup else n_trials)
                                )

                                optuna_search_performed = should_search_on_outer_fold(
                                    ordered_outer_keys=ordered_outer_keys,
                                    current_key=(target_year, outer_fold),
                                    reuse_policy=reuse_policy,
                                )
                                if reuse_policy["enabled"] and cached_best_params is None:
                                    optuna_search_performed = True

                                if optuna_search_performed:
                                    if ridge_grid_cfg is not None:
                                        sampler = optuna.samplers.GridSampler(
                                            {"alpha": [float(x) for x in ridge_grid_cfg["values"]]}
                                        )
                                        pruner = optuna.pruners.NopPruner()
                                    else:
                                        sampler = optuna.samplers.TPESampler(seed=int(sampler_seed))
                                        pruner = optuna.pruners.MedianPruner(
                                            n_startup_trials=pruner_startup_trials,
                                            n_warmup_steps=1,
                                        )
                                    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

                                    def objective(trial: optuna.trial.Trial) -> float:
                                        if ridge_grid_cfg is not None:
                                            params = {"alpha": trial.suggest_categorical("alpha", ridge_grid_cfg["values"])}
                                        else:
                                            params = suggest_params(trial, predictor, exp_cfg.get("model_backends", {}))
                                        train_losses = []
                                        val_losses = []
                                        train_metrics_list = []
                                        val_metrics_list = []

                                        for step_i, inner in enumerate(inner_payload):
                                            if use_safe_fusion:
                                                safe_inner = _fit_predict_safe_fusion(
                                                    inner_payload=[inner],
                                                    x_test=inner["x_val"],
                                                    y_test=inner["y_val"],
                                                    meta_test=inner["meta_val"],
                                                    predictor=predictor,
                                                    params=params,
                                                    preprocess_cfg=preprocess_cfg,
                                                    model_cfg=exp_cfg.get("model_backends", {}),
                                                    feature_spec_df=feature_spec_df,
                                                    seed=int(sampler_seed) + trial.number + step_i,
                                                    fusion_cfg=fusion_cfg,
                                                    repo_root=Path.cwd(),
                                                    use_gblup_for_g=use_gblup_for_g,
                                                )
                                                train_metrics = safe_inner["mean_train_metrics"]
                                                pred_val = safe_inner["pred_ens"]
                                            elif str(input_variant).strip().upper() in G_ONLY_VARIANTS and use_gblup_for_g:
                                                g_only = _get_or_fit_cached_g_branch(
                                                    inner_ctx=inner,
                                                    test_meta=inner["meta_val"],
                                                    y_train=inner["y_train"],
                                                    x_train_g=inner["x_train"].iloc[:, :0].copy(),
                                                    x_val_g=inner["x_val"].iloc[:, :0].copy(),
                                                    x_test_g=inner["meta_val"].iloc[:, :0].copy(),
                                                    predictor=predictor,
                                                    params=params,
                                                    preprocess_cfg=preprocess_cfg,
                                                    model_cfg=exp_cfg.get("model_backends", {}),
                                                    seed=int(sampler_seed) + trial.number + step_i,
                                                    repo_root=Path.cwd(),
                                                    use_gblup=use_gblup_for_g,
                                                )
                                                pred_train = g_only["pred_train"]
                                                pred_val = g_only["pred_val"]
                                                train_metrics = regression_metrics(inner["y_train"].to_numpy(), pred_train)
                                            else:
                                                pipe = build_pipeline(
                                                    predictor=predictor,
                                                    params=params,
                                                    n_features=inner["x_train"].shape[1],
                                                    preprocess_cfg=preprocess_cfg,
                                                    seed=int(sampler_seed) + trial.number + step_i,
                                                    model_cfg=exp_cfg.get("model_backends", {}),
                                                )
                                                pipe.fit(inner["x_train"], inner["y_train"])
                                                pred_train = pipe.predict(inner["x_train"])
                                                pred_val = pipe.predict(inner["x_val"])
                                                train_metrics = regression_metrics(inner["y_train"].to_numpy(), pred_train)
                                            val_metrics = regression_metrics(inner["y_val"].to_numpy(), pred_val)
                                            tr_loss = train_metrics["rmse"]
                                            va_loss = val_metrics["rmse"]
                                            train_losses.append(tr_loss)
                                            val_losses.append(va_loss)
                                            train_metrics_list.append(train_metrics)
                                            val_metrics_list.append(val_metrics)
                                            partial = float(np.mean(train_losses) + np.mean(val_losses))
                                            trial.report(partial, step=step_i)
                                            if trial.should_prune():
                                                raise optuna.exceptions.TrialPruned()

                                        mean_train = float(np.mean(train_losses))
                                        mean_val = float(np.mean(val_losses))
                                        gap = float(max(0.0, mean_val - mean_train))
                                        obj = float(mean_train + mean_val + lambda_gap * gap)
                                        mean_train_metrics = mean_regression_metrics(train_metrics_list)
                                        mean_val_metrics = mean_regression_metrics(val_metrics_list)
                                        val_pearsons = [
                                            float(metrics.get("pearson_r", float("nan"))) for metrics in val_metrics_list
                                        ]
                                        finite_val_pearsons = [x for x in val_pearsons if np.isfinite(x)]
                                        std_val_pearson = (
                                            float(np.std(finite_val_pearsons, ddof=0))
                                            if finite_val_pearsons
                                            else float("nan")
                                        )
                                        trial.set_user_attr("mean_train_loss", mean_train)
                                        trial.set_user_attr("mean_val_loss", mean_val)
                                        trial.set_user_attr("gap", gap)
                                        trial.set_user_attr("mean_train_r2", mean_train_metrics["r2"])
                                        trial.set_user_attr("mean_train_rmse", mean_train_metrics["rmse"])
                                        trial.set_user_attr("mean_train_mae", mean_train_metrics["mae"])
                                        trial.set_user_attr("mean_train_pearson", mean_train_metrics["pearson_r"])
                                        trial.set_user_attr("mean_val_r2", mean_val_metrics["r2"])
                                        trial.set_user_attr("mean_val_rmse", mean_val_metrics["rmse"])
                                        trial.set_user_attr("mean_val_mae", mean_val_metrics["mae"])
                                        trial.set_user_attr("mean_val_pearson", mean_val_metrics["pearson_r"])
                                        trial.set_user_attr("std_val_pearson", std_val_pearson)
                                        trial.set_user_attr(
                                            "selection_objective_mode",
                                            "max_mean_val_pearson_grid"
                                            if ridge_grid_cfg is not None
                                            else "min_train_val_rmse_gap",
                                        )
                                        if ridge_grid_cfg is not None:
                                            return -float(mean_val_metrics["pearson_r"])
                                        return obj

                                    study.optimize(objective, n_trials=n_trials_effective, show_progress_bar=False)
                                    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
                                    if not complete_trials:
                                        continue

                                    if ridge_grid_cfg is not None:
                                        best, sorted_trials = _select_best_ridge_grid_trial(
                                            complete_trials,
                                            grid_cfg=ridge_grid_cfg,
                                            input_variant=input_variant,
                                        )
                                    else:
                                        sorted_trials = sorted(complete_trials, key=lambda t: t.value)
                                        best = study.best_trial
                                    rank_map = {t.number: i + 1 for i, t in enumerate(sorted_trials)}
                                    cached_best_params = dict(best.params)
                                    cached_source = {
                                        "target_year": int(target_year),
                                        "outer_fold": int(outer_fold),
                                        "trial_number": int(best.number),
                                        "objective": float(best.value),
                                        "attrs": dict(best.user_attrs),
                                        "n_trials": int(len(complete_trials)),
                                    }
                                    for t in sorted_trials:
                                        trial_rows.append(
                                            {
                                                "scenario": scenario_name,
                                                "timeline": timeline_name,
                                                "target": target_col,
                                                "input_variant": input_variant,
                                                "predictor": predictor,
                                                "anchor_order": anchor_prefix_meta["anchor_order"],
                                                "anchor_idx": anchor_prefix_meta["anchor_idx"],
                                                "anchor_tb": anchor_prefix_meta["anchor_tb"],
                                                "anchor_phase": anchor_prefix_meta["anchor_phase"],
                                                "anchor_token": anchor_prefix_meta["anchor_token"],
                                                "target_year": int(target_year),
                                                "outer_fold": int(outer_fold),
                                                "trial_number": int(t.number),
                                                "params_json": json.dumps(t.params, ensure_ascii=False, sort_keys=True),
                                                "mean_train_loss": t.user_attrs.get("mean_train_loss"),
                                                "mean_val_loss": t.user_attrs.get("mean_val_loss"),
                                                "mean_train_r2": t.user_attrs.get("mean_train_r2"),
                                                "mean_train_rmse": t.user_attrs.get("mean_train_rmse"),
                                                "mean_train_mae": t.user_attrs.get("mean_train_mae"),
                                                "mean_train_pearson": t.user_attrs.get("mean_train_pearson"),
                                                "mean_val_r2": t.user_attrs.get("mean_val_r2"),
                                                "mean_val_rmse": t.user_attrs.get("mean_val_rmse"),
                                                "mean_val_mae": t.user_attrs.get("mean_val_mae"),
                                                "mean_val_pearson": t.user_attrs.get("mean_val_pearson"),
                                                "std_val_pearson": t.user_attrs.get("std_val_pearson"),
                                                "selection_objective_mode": t.user_attrs.get("selection_objective_mode"),
                                                "gap": t.user_attrs.get("gap"),
                                                "objective": t.value,
                                                "rank": rank_map[t.number],
                                                "optuna_search_performed": True,
                                                "optuna_reuse_scope": reuse_policy["mode"],
                                                "optuna_source_target_year": int(target_year),
                                                "optuna_source_outer_fold": int(outer_fold),
                                                "optuna_source_trial_number": int(best.number),
                                                "optuna_source_objective": float(best.value),
                                            }
                                        )
                                else:
                                    if cached_best_params is None or cached_source is None:
                                        raise RuntimeError(
                                            "Optuna 参数复用已启用，但当前任务缺少源 outer-fold 的最佳参数缓存。"
                                        )

                                best_params = dict(cached_best_params)
                                best_attrs = dict(cached_source["attrs"]) if optuna_search_performed and cached_source else {}
                                source_target_year = cached_source["target_year"] if cached_source else np.nan
                                source_outer_fold = cached_source["outer_fold"] if cached_source else np.nan
                                source_trial_number = cached_source["trial_number"] if cached_source else np.nan
                                source_objective = cached_source["objective"] if cached_source else np.nan

                                if use_safe_fusion:
                                    safe_result = _fit_predict_safe_fusion(
                                        inner_payload=inner_payload,
                                        x_test=x_test,
                                        y_test=y_test,
                                        meta_test=meta_test,
                                        predictor=predictor,
                                        params=best_params,
                                        preprocess_cfg=preprocess_cfg,
                                        model_cfg=exp_cfg.get("model_backends", {}),
                                        feature_spec_df=feature_spec_df,
                                        seed=int(sampler_seed) + 17,
                                        fusion_cfg=fusion_cfg,
                                        repo_root=Path.cwd(),
                                        use_gblup_for_g=use_gblup_for_g,
                                    )
                                    pred_mat = safe_result["pred_mat"]
                                    pred_ens = safe_result["pred_ens"]
                                    test_metrics = safe_result["test_metrics"]
                                    mean_val_metrics = safe_result["mean_val_metrics"]
                                    mean_val_h_metrics = safe_result["mean_val_h_metrics"]
                                    mean_val_g_metrics = safe_result["mean_val_g_metrics"]
                                    pruning_stats = {
                                        **pruning_stats,
                                        "fusion_mode": "ols_or_branch_h_gblup"
                                        if use_gblup_for_g
                                        else "ols_or_branch_h_g",
                                        "fusion_candidate_mode": safe_result["candidate_selected_mode"],
                                        "fusion_candidate_list_json": json.dumps(
                                            [str(x) for x in safe_result["candidate_selected_list"]],
                                            ensure_ascii=False,
                                        ),
                                        "fusion_intercept_mean": float(safe_result["intercept_mean"]),
                                        "fusion_intercept_mode": float(safe_result["intercept_mode"]),
                                        "fusion_intercept_list_json": json.dumps(
                                            [float(x) for x in safe_result["intercept_list"]],
                                            ensure_ascii=False,
                                        ),
                                        "fusion_weight_h_mean": float(safe_result["weight_h_mean"]),
                                        "fusion_weight_g_mean": float(safe_result["weight_g_mean"]),
                                        "fusion_weight_h_mode": float(safe_result["weight_h_mode"]),
                                        "fusion_weight_g_mode": float(safe_result["weight_g_mode"]),
                                        "fusion_weight_h_list_json": json.dumps(
                                            [float(x) for x in safe_result["weight_h_list"]],
                                            ensure_ascii=False,
                                        ),
                                        "fusion_weight_g_list_json": json.dumps(
                                            [float(x) for x in safe_result["weight_g_list"]],
                                            ensure_ascii=False,
                                        ),
                                    }
                                elif str(input_variant).strip().upper() in G_ONLY_VARIANTS and use_gblup_for_g:
                                    inner_preds = []
                                    val_metrics_list = []
                                    for inner in inner_payload:
                                        g_only = _get_or_fit_cached_g_branch(
                                            inner_ctx=inner,
                                            test_meta=meta_test,
                                            y_train=inner["y_train"],
                                            x_train_g=inner["x_train"].iloc[:, :0].copy(),
                                            x_val_g=inner["x_val"].iloc[:, :0].copy(),
                                            x_test_g=x_test.iloc[:, :0].copy(),
                                            predictor=predictor,
                                            params=best_params,
                                            preprocess_cfg=preprocess_cfg,
                                            model_cfg=exp_cfg.get("model_backends", {}),
                                            seed=int(sampler_seed) + int(inner["inner_fold"]) + 17,
                                            repo_root=Path.cwd(),
                                            use_gblup=use_gblup_for_g,
                                        )
                                        pred_val = g_only["pred_val"]
                                        pred_test = g_only["pred_test"]
                                        val_metrics_list.append(regression_metrics(inner["y_val"].to_numpy(), pred_val))
                                        inner_preds.append(pred_test)

                                    pred_mat = np.column_stack(inner_preds)
                                    pred_ens = pred_mat.mean(axis=1)
                                    test_metrics = regression_metrics(y_test.to_numpy(dtype=float), pred_ens)
                                    mean_val_metrics = mean_regression_metrics(val_metrics_list)
                                    mean_val_h_metrics = _nan_metric_dict()
                                    mean_val_g_metrics = mean_val_metrics.copy()
                                    pruning_stats = {**pruning_stats, "fusion_mode": "g_only_gblup"}
                                else:
                                    inner_preds = []
                                    val_metrics_list = []
                                    for inner in inner_payload:
                                        pipe = build_pipeline(
                                            predictor=predictor,
                                            params=best_params,
                                            n_features=inner["x_train"].shape[1],
                                            preprocess_cfg=preprocess_cfg,
                                            seed=int(sampler_seed) + int(inner["inner_fold"]) + 17,
                                            model_cfg=exp_cfg.get("model_backends", {}),
                                        )
                                        pipe.fit(inner["x_train"], inner["y_train"])
                                        pred_val = pipe.predict(inner["x_val"])
                                        val_metrics_list.append(regression_metrics(inner["y_val"].to_numpy(), pred_val))
                                        inner_preds.append(pipe.predict(x_test))

                                    pred_mat = np.column_stack(inner_preds)
                                    pred_ens = pred_mat.mean(axis=1)
                                    test_metrics = regression_metrics(y_test.to_numpy(dtype=float), pred_ens)
                                    mean_val_metrics = mean_regression_metrics(val_metrics_list)
                                    mean_val_h_metrics = _nan_metric_dict()
                                    mean_val_g_metrics = _nan_metric_dict()

                                for i, pid in enumerate(x_test.index.tolist()):
                                    y_true_i = float(y_test.loc[pid])
                                    y_ens_i = float(pred_ens[i])
                                    for j, inner in enumerate(inner_payload):
                                        inner_pred_rows.append(
                                            {
                                                "plot_id": pid,
                                                "year": int(meta_test.loc[pid, "year"]),
                                                "scenario": scenario_name,
                                                "timeline": timeline_name,
                                                "target": target_col,
                                                "input_variant": input_variant,
                                                "predictor": predictor,
                                                "anchor_order": anchor_prefix_meta["anchor_order"],
                                                "anchor_idx": anchor_prefix_meta["anchor_idx"],
                                                "anchor_tb": anchor_prefix_meta["anchor_tb"],
                                                "anchor_phase": anchor_prefix_meta["anchor_phase"],
                                                "anchor_token": anchor_prefix_meta["anchor_token"],
                                                "target_year": int(target_year),
                                                "outer_fold": int(outer_fold),
                                                "inner_fold": int(inner["inner_fold"]),
                                                "y_true": y_true_i,
                                                "y_pred_inner_model": float(pred_mat[i, j]),
                                                "y_pred_ensemble_mean": y_ens_i,
                                            }
                                        )
                                    oof_rows.append(
                                        {
                                            "plot_id": pid,
                                            "year": int(meta_test.loc[pid, "year"]),
                                            "scenario": scenario_name,
                                            "timeline": timeline_name,
                                            "target": target_col,
                                            "input_variant": input_variant,
                                            "predictor": predictor,
                                            "anchor_order": anchor_prefix_meta["anchor_order"],
                                            "anchor_idx": anchor_prefix_meta["anchor_idx"],
                                            "anchor_tb": anchor_prefix_meta["anchor_tb"],
                                            "anchor_phase": anchor_prefix_meta["anchor_phase"],
                                            "anchor_token": anchor_prefix_meta["anchor_token"],
                                            "target_year": int(target_year),
                                            "outer_fold": int(outer_fold),
                                            "y_true": y_true_i,
                                            "y_pred": y_ens_i,
                                        }
                                    )

                                variant_upper = str(input_variant).strip().upper()
                                has_g_branch = "G" in variant_upper
                                has_h_branch = variant_upper != "G"
                                canonical_variant_upper = variant_upper
                                if variant_upper == "H_ANCHOR_AUTO":
                                    canonical_variant_upper = "H_ANCHOR_AUTO"
                                elif variant_upper == "G+H_ANCHOR_AUTO":
                                    canonical_variant_upper = "G+H_ANCHOR_AUTO"
                                g_branch_model = (
                                    "GBLUP_Sommer" if has_g_branch and use_gblup_for_g else (predictor if has_g_branch else "")
                                )
                                h_branch_model = predictor if has_h_branch else ""
                                metrics_fold_rows.append(
                                    {
                                        "scenario": scenario_name,
                                        "timeline": timeline_name,
                                        "target": target_col,
                                        "input_variant": input_variant,
                                        "predictor": predictor,
                                        "anchor_order": anchor_prefix_meta["anchor_order"],
                                        "anchor_idx": anchor_prefix_meta["anchor_idx"],
                                        "anchor_tb": anchor_prefix_meta["anchor_tb"],
                                        "anchor_phase": anchor_prefix_meta["anchor_phase"],
                                        "anchor_token": anchor_prefix_meta["anchor_token"],
                                        "target_year": int(target_year),
                                        "outer_fold": int(outer_fold),
                                        "best_trial_number": int(source_trial_number),
                                        "best_objective": float(source_objective) if optuna_search_performed else float("nan"),
                                        "best_mean_train_loss": float(best_attrs.get("mean_train_loss", np.nan)),
                                        "best_mean_val_loss": float(best_attrs.get("mean_val_loss", np.nan)),
                                        "best_gap": float(best_attrs.get("gap", np.nan)),
                                        "train_r2": float(best_attrs.get("mean_train_r2", np.nan)),
                                        "train_rmse": float(best_attrs.get("mean_train_rmse", np.nan)),
                                        "train_mae": float(best_attrs.get("mean_train_mae", np.nan)),
                                        "train_pearson": float(best_attrs.get("mean_train_pearson", np.nan)),
                                        "val_r2": float(mean_val_metrics["r2"]) if optuna_search_performed else float("nan"),
                                        "val_rmse": float(mean_val_metrics["rmse"]) if optuna_search_performed else float("nan"),
                                        "val_mae": float(mean_val_metrics["mae"]) if optuna_search_performed else float("nan"),
                                        "val_pearson": float(mean_val_metrics["pearson_r"]) if optuna_search_performed else float("nan"),
                                        "val_h_r2": float(mean_val_h_metrics["r2"]) if optuna_search_performed else float("nan"),
                                        "val_h_rmse": float(mean_val_h_metrics["rmse"]) if optuna_search_performed else float("nan"),
                                        "val_h_mae": float(mean_val_h_metrics["mae"]) if optuna_search_performed else float("nan"),
                                        "val_h_pearson": float(mean_val_h_metrics["pearson_r"]) if optuna_search_performed else float("nan"),
                                        "val_g_r2": float(mean_val_g_metrics["r2"]) if optuna_search_performed else float("nan"),
                                        "val_g_rmse": float(mean_val_g_metrics["rmse"]) if optuna_search_performed else float("nan"),
                                        "val_g_mae": float(mean_val_g_metrics["mae"]) if optuna_search_performed else float("nan"),
                                        "val_g_pearson": float(mean_val_g_metrics["pearson_r"]) if optuna_search_performed else float("nan"),
                                        "best_std_val_pearson": float(best_attrs.get("std_val_pearson", np.nan)),
                                        "test_r2": float(test_metrics["r2"]),
                                        "test_rmse": float(test_metrics["rmse"]),
                                        "test_mae": float(test_metrics["mae"]),
                                        "test_pearson": float(test_metrics["pearson_r"]),
                                        "n_train": int(np.mean([len(inner["train_ids"]) for inner in inner_payload])),
                                        "n_val": int(np.mean([len(inner["val_ids"]) for inner in inner_payload])),
                                        "n_test": int(len(payload["test_ids"])),
                                        "n_inner_folds": int(len(inner_payload)),
                                        "n_features_total": int(x_info["n_features_total"]),
                                        "n_features_h_phase": int(x_info["n_features_h_phase"]),
                                        "n_features_h_anchor_local": int(x_info.get("n_features_h_anchor_local", 0)),
                                        "n_features_h_anchor_vi": int(x_info.get("n_features_h_anchor_vi", 0)),
                                        "n_features_h_anchor_auto": int(x_info.get("n_features_h_anchor_auto", 0)),
                                        "n_features_after_pruning": int(
                                            pruning_stats.get("n_total_features_after", x_test.shape[1])
                                        ),
                                        "n_h_features_after_pruning": int(
                                            pruning_stats.get(
                                                "n_h_features_after",
                                                x_info.get(
                                                    "n_features_h_anchor_local",
                                                    x_info.get("n_features_h_anchor_vi", x_info["n_features_h_phase"]),
                                                ),
                                            )
                                        ),
                                        "anchor_local_score_target": str(pruning_stats.get("score_target", "")),
                                        "n_group_kept": int(pruning_stats.get("n_group_kept", 0)),
                                        "n_anchor_kept": int(pruning_stats.get("n_anchor_kept", 0)),
                                        "n_vi_kept": int(pruning_stats.get("n_vi_kept", 0)),
                                        "selected_k_mode": int(pruning_stats.get("selected_k_mode", 0)),
                                        "selected_window_radius": int(pruning_stats.get("selected_window_radius", 0)),
                                        "feature_selection_search_performed": bool(
                                            pruning_stats.get("selection_search_performed", False)
                                        ),
                                        "feature_selection_reused_from_first_outer_fold": bool(
                                            pruning_stats.get("selection_reused_from_first_outer_fold", False)
                                        ),
                                        "feature_selection_source_target_year": pruning_stats.get(
                                            "selection_source_target_year",
                                            np.nan,
                                        ),
                                        "feature_selection_source_outer_fold": pruning_stats.get(
                                            "selection_source_outer_fold",
                                            np.nan,
                                        ),
                                        "growth_aware_min_groups": int(
                                            pruning_stats.get(
                                                "group_min_per_phase_effective_mode",
                                                pruning_stats.get("group_min_per_phase_effective", 0),
                                            )
                                        ),
                                        "kept_group_ids_json": pruning_stats.get("kept_group_ids_json", ""),
                                        "kept_anchor_tokens_json": pruning_stats.get("kept_anchor_tokens_json", ""),
                                        "kept_vi_names_json": pruning_stats.get("kept_vi_names_json", ""),
                                        "n_features_h_full": int(x_info["n_features_h_full"]),
                                        "n_features_g": int(x_info["n_features_g"]),
                                        "fusion_mode": pruning_stats.get("fusion_mode", ""),
                                        "fusion_candidate_mode": pruning_stats.get("fusion_candidate_mode", ""),
                                        "fusion_candidate_list_json": pruning_stats.get("fusion_candidate_list_json", ""),
                                        "fusion_intercept_mean": float(
                                            pruning_stats.get("fusion_intercept_mean", np.nan)
                                        ),
                                        "fusion_intercept_mode": float(
                                            pruning_stats.get("fusion_intercept_mode", np.nan)
                                        ),
                                        "fusion_intercept_list_json": pruning_stats.get(
                                            "fusion_intercept_list_json",
                                            "",
                                        ),
                                        "fusion_alpha_mean": float("nan"),
                                        "fusion_alpha_mode": float("nan"),
                                        "fusion_weight_h_mean": float(
                                            pruning_stats.get("fusion_weight_h_mean", np.nan)
                                        ),
                                        "fusion_weight_g_mean": float(
                                            pruning_stats.get("fusion_weight_g_mean", np.nan)
                                        ),
                                        "fusion_weight_h_mode": float(
                                            pruning_stats.get("fusion_weight_h_mode", np.nan)
                                        ),
                                        "fusion_weight_g_mode": float(
                                            pruning_stats.get("fusion_weight_g_mode", np.nan)
                                        ),
                                        "fusion_weight_h_list_json": pruning_stats.get(
                                            "fusion_weight_h_list_json",
                                            "",
                                        ),
                                        "fusion_weight_g_list_json": pruning_stats.get(
                                            "fusion_weight_g_list_json",
                                            "",
                                        ),
                                        "fusion_alpha_list_json": "",
                                        "g_branch_model": g_branch_model,
                                        "h_branch_model": h_branch_model,
                                        "effective_h_variant": canonical_variant_upper,
                                        "n_optuna_trials": int(cached_source["n_trials"])
                                        if optuna_search_performed and cached_source
                                        else 0,
                                        "optuna_search_performed": bool(optuna_search_performed),
                                        "optuna_reuse_scope": reuse_policy["mode"],
                                        "optuna_source_target_year": source_target_year,
                                        "optuna_source_outer_fold": source_outer_fold,
                                        "optuna_source_trial_number": source_trial_number,
                                        "optuna_source_objective": source_objective,
                                        "optuna_selection_objective_mode": str(
                                            best_attrs.get("selection_objective_mode", "")
                                        ),
                                    }
                                )
                        _append_progress(
                            output_dir,
                            enabled=progress_enabled,
                            event="task_end",
                            payload={
                                "task_idx": task_idx,
                                "total_tasks": total_tasks,
                                "scenario": scenario_name,
                                "timeline": timeline_name,
                                "target": target_col,
                                "input_variant": input_variant,
                                "predictor": predictor,
                                "anchor_order": anchor_prefix_meta["anchor_order"],
                                "anchor_idx": anchor_prefix_meta["anchor_idx"],
                                "anchor_tb": anchor_prefix_meta["anchor_tb"],
                                "anchor_phase": anchor_prefix_meta["anchor_phase"],
                            },
                        )

    representation_overview_df = pd.DataFrame(representation_overview_rows)
    representation_feature_spec_df = (
        pd.concat(representation_feature_spec_frames, ignore_index=True) if representation_feature_spec_frames else pd.DataFrame()
    )
    feature_overview_df = pd.DataFrame(feature_overview_rows)
    split_usage_df = pd.DataFrame(split_usage_rows)
    metrics_fold_df = pd.DataFrame(metrics_fold_rows)
    oof_df = pd.DataFrame(oof_rows)
    inner_pred_df = pd.DataFrame(inner_pred_rows)
    trial_df = pd.DataFrame(trial_rows)
    _validate_result34_outer_fold_coverage(
        metrics_fold_df=metrics_fold_df,
        split_usage_df=split_usage_df,
    )

    finalize_result34_outputs(
        output_dir=output_dir,
        cfg={
            "config_path": str(config_path),
            "data": data_cfg,
            "experiment": {
                **exp_cfg,
                "predictors_run": active_predictors,
                "targets": targets,
                "input_variants": input_variants,
                "genotype_representation": genotype_representation,
            },
            "optuna": optuna_cfg,
            "preprocessing": preprocess_cfg,
        },
        representation_overview_df=representation_overview_df,
        representation_feature_spec_df=representation_feature_spec_df,
        feature_overview_df=feature_overview_df,
        split_usage_df=split_usage_df,
        metrics_fold_df=metrics_fold_df,
        oof_df=oof_df,
        inner_pred_df=inner_pred_df,
        trial_df=trial_df,
    )

    _append_progress(output_dir, enabled=progress_enabled, event="run_end", payload={"output_dir": output_dir.as_posix()})
    print(f"Result3.4A completed: {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Result3.4A reduced-H feature engineering experiments.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run_result34(Path(args.config))


if __name__ == "__main__":
    main()
