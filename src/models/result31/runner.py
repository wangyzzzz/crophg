from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
from matplotlib.lines import Line2D
from optuna.trial import TrialState
from sklearn.exceptions import ConvergenceWarning

from models.anchorwise.data_loader import MultiTargetDataBundle, load_multitarget_model_inputs
from models.anchorwise.modeling import HAS_LIGHTGBM, HAS_XGBOOST, build_pipeline, sanitize_feature_matrix, suggest_params
from models.common.io_utils import ensure_dir, get_git_commit_hash, now_iso, read_yaml, write_json, write_yaml
from models.common.metrics import mean_regression_metrics, regression_metrics, rmse
from models.common.optuna_reuse import normalize_outer_groups, resolve_optuna_reuse_policy, should_search_on_outer_fold
from models.common.split_loader import WithinSeasonSplit, load_split_groups

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Skipping features without any observed values")
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt", category=RuntimeWarning)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _resolve_output_dir(output_cfg: dict) -> Path:
    base = Path(
        output_cfg.get(
            "output_dir_base",
            output_cfg.get("output_dir", "outputs/experiments/result3_1_fullseason_hcg_multitrait"),
        )
    )
    append_timestamp = bool(output_cfg.get("append_timestamp", True))
    allow_overwrite = bool(output_cfg.get("allow_overwrite", False))
    timestamp_format = str(output_cfg.get("timestamp_format", "%Y%m%d_%H%M%S"))

    if append_timestamp:
        stamp = datetime.now().strftime(timestamp_format)
        out = Path(f"{base.as_posix()}_{stamp}")
        idx = 1
        while out.exists():
            out = Path(f"{base.as_posix()}_{stamp}_{idx:02d}")
            idx += 1
    else:
        out = base
        if out.exists() and any(out.iterdir()) and not allow_overwrite:
            raise RuntimeError(
                f"输出目录已存在且非空：{out}。"
                "请开启 output.append_timestamp=true，或设置 output.allow_overwrite=true。"
            )

    ensure_dir(out)

    latest_pointer_file = output_cfg.get("latest_pointer_file")
    if latest_pointer_file:
        pointer_path = Path(latest_pointer_file)
        ensure_dir(pointer_path.parent)
        pointer_path.write_text(out.as_posix(), encoding="utf-8")

    return out


def _append_progress(out_dir: Path, *, enabled: bool, event: str, payload: dict) -> None:
    if not enabled:
        return
    path = out_dir / "progress.jsonl"
    record = {"ts": now_iso(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _default_timeline_dirs() -> dict:
    return {
        "day": "data/processed/model_inputs_engineered/day",
        "day_rel_heading": "data/processed/model_inputs_engineered/day_rel_heading",
        "gdd_abs": "data/processed/model_inputs_engineered/gdd_abs",
        "gdd_rel_heading": "data/processed/model_inputs_engineered/gdd_rel_heading",
    }


def _default_scenarios_cfg() -> dict:
    return {
        "within_season": {"split_dir": "data/processed/splits", "validation_scenario": "within_season"},
        "loso": {"split_dir": "data/processed/splits", "validation_scenario": "loso"},
        "loso_genotype": {
            "split_dir": "data/processed/splits_loso_genotype_nested",
            "validation_scenario": "loso_genotype",
        },
    }


def _resolve_timeline_dirs(data_cfg: dict) -> dict:
    return data_cfg.get("timeline_dirs", _default_timeline_dirs())


def _resolve_scenarios_cfg(data_cfg: dict) -> dict:
    return data_cfg.get("scenarios", _default_scenarios_cfg())


def _load_genotype_fold_map(fold_map_path: Path) -> dict[str, int]:
    df = pd.read_csv(fold_map_path)
    required = {"genotype_id", "genotype_fold"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"genotype fold map 缺少列: {sorted(missing)}")
    out: dict[str, int] = {}
    for _, row in df.iterrows():
        out[str(row["genotype_id"])] = int(row["genotype_fold"])
    if not out:
        raise ValueError(f"genotype fold map 为空: {fold_map_path}")
    return out


def _build_meta_with_genotype_fold(
    meta_df: pd.DataFrame,
    *,
    genotype_fold_map: dict[str, int],
) -> pd.DataFrame:
    work = meta_df.loc[:, ["year", "genotype_id"]].copy()
    work["sample_id"] = meta_df.index.astype(str)
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work["genotype_id"] = work["genotype_id"].astype(str)
    work["genotype_fold"] = work["genotype_id"].map(genotype_fold_map).astype("Int64")
    missing = work["genotype_fold"].isna()
    if missing.any():
        missing_genotypes = sorted(work.loc[missing, "genotype_id"].unique().tolist())[:10]
        raise ValueError(f"存在无法映射到 genotype_fold 的 genotype_id，例如: {missing_genotypes}")
    return work


def _build_reference_split_groups(
    meta_df: pd.DataFrame,
    *,
    genotype_fold_map: dict[str, int],
    random_seed: int,
) -> Dict[Tuple[int, int], List[WithinSeasonSplit]]:
    work = _build_meta_with_genotype_fold(meta_df, genotype_fold_map=genotype_fold_map)
    years = sorted(work["year"].dropna().astype(int).unique().tolist())
    folds = sorted(work["genotype_fold"].dropna().astype(int).unique().tolist())

    groups: Dict[Tuple[int, int], List[WithinSeasonSplit]] = {}
    for year in years:
        year_df = work.loc[work["year"] == int(year)].copy()
        for outer_fold in folds:
            test_ids = year_df.loc[year_df["genotype_fold"] == int(outer_fold), "sample_id"].astype(str).tolist()
            if not test_ids:
                continue
            val_fold = folds[(folds.index(int(outer_fold)) + 1) % len(folds)]
            val_ids = year_df.loc[year_df["genotype_fold"] == int(val_fold), "sample_id"].astype(str).tolist()
            train_mask = ~(
                ((work["year"] == int(year)) & (work["genotype_fold"] == int(outer_fold)))
                | ((work["year"] == int(year)) & (work["genotype_fold"] == int(val_fold)))
            )
            train_ids = work.loc[train_mask, "sample_id"].astype(str).tolist()
            split_name = f"reference_{year}_outer{outer_fold}_inner0"
            groups[(int(year), int(outer_fold))] = [
                WithinSeasonSplit(
                    split_name=split_name,
                    target_year=int(year),
                    outer_fold=int(outer_fold),
                    inner_fold=0,
                    train_ids=train_ids,
                    val_ids=val_ids,
                    test_ids=test_ids,
                    random_seed=int(random_seed),
                )
            ]
    return groups


def _build_known_year_unknown_genotype_split_groups(
    meta_df: pd.DataFrame,
    *,
    genotype_fold_map: dict[str, int],
    random_seed: int,
) -> Dict[Tuple[int, int], List[WithinSeasonSplit]]:
    work = _build_meta_with_genotype_fold(meta_df, genotype_fold_map=genotype_fold_map)
    years = sorted(work["year"].dropna().astype(int).unique().tolist())
    folds = sorted(work["genotype_fold"].dropna().astype(int).unique().tolist())

    groups: Dict[Tuple[int, int], List[WithinSeasonSplit]] = {}
    for year in years:
        year_df = work.loc[work["year"] == int(year)].copy()
        for outer_fold in folds:
            test_ids = year_df.loc[year_df["genotype_fold"] == int(outer_fold), "sample_id"].astype(str).tolist()
            if not test_ids:
                continue
            val_fold = folds[(folds.index(int(outer_fold)) + 1) % len(folds)]
            val_ids = year_df.loc[year_df["genotype_fold"] == int(val_fold), "sample_id"].astype(str).tolist()
            train_mask = (
                (work["genotype_fold"] != int(outer_fold))
                & ~((work["year"] == int(year)) & (work["genotype_fold"] == int(val_fold)))
            )
            train_ids = work.loc[train_mask, "sample_id"].astype(str).tolist()
            split_name = f"within_season_known_year_{year}_outer{outer_fold}_inner0"
            groups[(int(year), int(outer_fold))] = [
                WithinSeasonSplit(
                    split_name=split_name,
                    target_year=int(year),
                    outer_fold=int(outer_fold),
                    inner_fold=0,
                    train_ids=train_ids,
                    val_ids=val_ids,
                    test_ids=test_ids,
                    random_seed=int(random_seed),
                )
            ]
    return groups


def _build_unknown_year_known_genotype_split_groups(
    meta_df: pd.DataFrame,
    *,
    genotype_fold_map: dict[str, int],
    random_seed: int,
) -> Dict[Tuple[int, int], List[WithinSeasonSplit]]:
    work = _build_meta_with_genotype_fold(meta_df, genotype_fold_map=genotype_fold_map)
    years = sorted(work["year"].dropna().astype(int).unique().tolist())
    folds = sorted(work["genotype_fold"].dropna().astype(int).unique().tolist())

    groups: Dict[Tuple[int, int], List[WithinSeasonSplit]] = {}
    for year in years:
        remaining_years = [y for y in years if int(y) != int(year)]
        if len(remaining_years) != 2:
            raise ValueError("当前 loso 自定义划分要求恰有 3 个年份。")
        val_year = int(remaining_years[0])
        for outer_fold in folds:
            test_ids = work.loc[
                (work["year"] == int(year)) & (work["genotype_fold"] == int(outer_fold)),
                "sample_id",
            ].astype(str).tolist()
            if not test_ids:
                continue
            val_ids = work.loc[
                (work["year"] == val_year) & (work["genotype_fold"] == int(outer_fold)),
                "sample_id",
            ].astype(str).tolist()
            train_mask = (work["year"] != int(year)) & ~(
                (work["year"] == val_year) & (work["genotype_fold"] == int(outer_fold))
            )
            train_ids = work.loc[train_mask, "sample_id"].astype(str).tolist()
            split_name = f"loso_known_genotype_{year}_outer{outer_fold}_inner0"
            groups[(int(year), int(outer_fold))] = [
                WithinSeasonSplit(
                    split_name=split_name,
                    target_year=int(year),
                    outer_fold=int(outer_fold),
                    inner_fold=0,
                    train_ids=train_ids,
                    val_ids=val_ids,
                    test_ids=test_ids,
                    random_seed=int(random_seed),
                )
            ]
    return groups


def _build_unknown_year_unknown_genotype_split_groups(
    meta_df: pd.DataFrame,
    *,
    genotype_fold_map: dict[str, int],
    random_seed: int,
) -> Dict[Tuple[int, int], List[WithinSeasonSplit]]:
    work = _build_meta_with_genotype_fold(meta_df, genotype_fold_map=genotype_fold_map)
    years = sorted(work["year"].dropna().astype(int).unique().tolist())
    folds = sorted(work["genotype_fold"].dropna().astype(int).unique().tolist())

    groups: Dict[Tuple[int, int], List[WithinSeasonSplit]] = {}
    for year in years:
        remaining_years = [y for y in years if int(y) != int(year)]
        if len(remaining_years) != 2:
            raise ValueError("当前 loso_genotype 自定义划分要求恰有 3 个年份。")
        val_year = int(remaining_years[0])
        for outer_fold in folds:
            test_ids = work.loc[
                (work["year"] == int(year)) & (work["genotype_fold"] == int(outer_fold)),
                "sample_id",
            ].astype(str).tolist()
            if not test_ids:
                continue
            val_fold = folds[(folds.index(int(outer_fold)) + 1) % len(folds)]
            if int(val_fold) == int(outer_fold):
                raise ValueError("val_fold 不能与 outer_fold 相同。")
            val_ids = work.loc[
                (work["year"] == val_year) & (work["genotype_fold"] == int(val_fold)),
                "sample_id",
            ].astype(str).tolist()
            train_mask = (
                (work["year"] != int(year))
                & (work["genotype_fold"] != int(outer_fold))
                & ~((work["year"] == val_year) & (work["genotype_fold"] == int(val_fold)))
            )
            train_ids = work.loc[train_mask, "sample_id"].astype(str).tolist()
            split_name = f"loso_genotype_unknown_{year}_outer{outer_fold}_inner0"
            groups[(int(year), int(outer_fold))] = [
                WithinSeasonSplit(
                    split_name=split_name,
                    target_year=int(year),
                    outer_fold=int(outer_fold),
                    inner_fold=0,
                    train_ids=train_ids,
                    val_ids=val_ids,
                    test_ids=test_ids,
                    random_seed=int(random_seed),
                )
            ]
    return groups


def _resolve_split_groups_for_scenario(
    *,
    scenario_name: str,
    scenario_cfg: dict,
    meta_df: pd.DataFrame,
    random_seed: int,
) -> Dict[Tuple[int, int], List[WithinSeasonSplit]]:
    custom_strategy = str(scenario_cfg.get("custom_split_strategy", "")).strip().lower()
    if not custom_strategy:
        split_dir = Path(str(scenario_cfg["split_dir"]))
        validation_scenario = str(scenario_cfg.get("validation_scenario", scenario_name))
        return load_split_groups(split_dir, validation_scenario=validation_scenario)

    fold_map_rel = str(
        scenario_cfg.get(
            "genotype_fold_map_path",
            "outputs/reports/loso_genotype_nested_genotype_fold_map.csv",
        )
    )
    fold_map_path = Path(fold_map_rel)
    if not fold_map_path.is_absolute():
        fold_map_path = (Path.cwd() / fold_map_path).resolve()
    genotype_fold_map = _load_genotype_fold_map(fold_map_path)

    if custom_strategy in {"reference", "known_year_known_genotype", "reference_known_year_known_genotype"}:
        return _build_reference_split_groups(
            meta_df,
            genotype_fold_map=genotype_fold_map,
            random_seed=random_seed,
        )
    if custom_strategy in {
        "within_season_known_year_unknown_genotype",
        "known_year_unknown_genotype",
        "within_season_grouped_genotype",
    }:
        return _build_known_year_unknown_genotype_split_groups(
            meta_df,
            genotype_fold_map=genotype_fold_map,
            random_seed=random_seed,
        )
    if custom_strategy in {"loso_known_genotype", "unknown_year_known_genotype", "loso_cellwise"}:
        return _build_unknown_year_known_genotype_split_groups(
            meta_df,
            genotype_fold_map=genotype_fold_map,
            random_seed=random_seed,
        )
    if custom_strategy in {"loso_genotype_unknown", "unknown_year_unknown_genotype", "loso_genotype_cellwise"}:
        return _build_unknown_year_unknown_genotype_split_groups(
            meta_df,
            genotype_fold_map=genotype_fold_map,
            random_seed=random_seed,
        )
    raise ValueError(f"不支持的 custom_split_strategy: {custom_strategy}")


def _resolve_active_predictors(exp_cfg: dict) -> list[str]:
    predictors_supported = [
        str(x).lower()
        for x in exp_cfg.get(
            "predictors_supported",
            ["ridge", "lasso", "elasticnet", "lightgbm", "xgboost", "pls", "random_forest"],
        )
    ]
    predictors_run = [str(x).lower() for x in exp_cfg.get("predictors_run", predictors_supported)]
    skip_lightgbm_runtime = bool(exp_cfg.get("skip_lightgbm_runtime", False))
    backend_cfg = exp_cfg.get("model_backends", {})

    active_predictors: List[str] = []
    for p in predictors_run:
        if p not in predictors_supported:
            continue
        if p == "lightgbm" and skip_lightgbm_runtime:
            continue
        if p == "lightgbm" and str(backend_cfg.get("lightgbm_backend", "native")).lower() == "native" and not HAS_LIGHTGBM:
            continue
        if p == "xgboost" and not HAS_XGBOOST:
            continue
        active_predictors.append(p)
    if not active_predictors:
        raise RuntimeError("没有可运行的预测器。")
    return active_predictors


def _normalize_task_spec(spec: dict) -> tuple[str, str, str, str]:
    required = ["scenario", "timeline", "target", "predictor"]
    missing = [k for k in required if k not in spec]
    if missing:
        raise KeyError(f"task_spec 缺少字段: {missing}")
    return (
        str(spec["scenario"]),
        str(spec["timeline"]),
        str(spec["target"]),
        str(spec["predictor"]).lower(),
    )


def _resolve_task_filter_set(exp_cfg: dict) -> set[tuple[str, str, str, str]]:
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


def enumerate_result31_tasks_from_config(cfg: dict) -> list[dict]:
    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})

    timeline_dirs = _resolve_timeline_dirs(data_cfg)
    scenarios_cfg = _resolve_scenarios_cfg(data_cfg)
    targets = [str(x) for x in exp_cfg.get("targets", ["ActualYD", "CM", "LM", "PHM", "Spike", "TKW"])]
    active_predictors = _resolve_active_predictors(exp_cfg)
    task_filter_set = _resolve_task_filter_set(exp_cfg)

    tasks = []
    for scenario_name in scenarios_cfg.keys():
        for timeline_name in timeline_dirs.keys():
            for target_col in targets:
                for predictor in active_predictors:
                    task_key = (scenario_name, timeline_name, target_col, predictor)
                    if task_filter_set and task_key not in task_filter_set:
                        continue
                    tasks.append(
                        {
                            "scenario": scenario_name,
                            "timeline": timeline_name,
                            "target": target_col,
                            "predictor": predictor,
                        }
                    )
    return tasks


def _filter_ids(ids: List[str], available_set: set[str]) -> List[str]:
    return [str(x) for x in ids if str(x) in available_set]


def _prepare_split_groups(
    split_groups: Dict[Tuple[int, int], List[WithinSeasonSplit]],
    available_ids: set[str],
) -> Tuple[Dict[Tuple[int, int], List[dict]], dict]:
    prepared: Dict[Tuple[int, int], List[dict]] = {}
    stats = {
        "n_groups_total": int(len(split_groups)),
        "n_groups_used": 0,
        "n_ids_dropped_train": 0,
        "n_ids_dropped_val": 0,
        "n_ids_dropped_test": 0,
    }

    for key, splits in split_groups.items():
        rows = []
        for sp in splits:
            tr = _filter_ids(sp.train_ids, available_ids)
            va = _filter_ids(sp.val_ids, available_ids)
            te = _filter_ids(sp.test_ids, available_ids)

            stats["n_ids_dropped_train"] += max(0, len(sp.train_ids) - len(tr))
            stats["n_ids_dropped_val"] += max(0, len(sp.val_ids) - len(va))
            stats["n_ids_dropped_test"] += max(0, len(sp.test_ids) - len(te))

            if not tr or not va or not te:
                continue

            rows.append(
                {
                    "split_name": sp.split_name,
                    "target_year": int(sp.target_year),
                    "outer_fold": int(sp.outer_fold),
                    "inner_fold": int(sp.inner_fold),
                    "train_ids": tr,
                    "val_ids": va,
                    "test_ids": te,
                }
            )

        if rows:
            prepared[key] = sorted(rows, key=lambda x: x["inner_fold"])

    stats["n_groups_used"] = int(len(prepared))
    return prepared, stats


def _build_outer_payload(x: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, group_splits: List[dict]) -> dict:
    inner_payload = []
    for row in group_splits:
        tr_ids = row["train_ids"]
        va_ids = row["val_ids"]
        inner_payload.append(
            {
                "inner_fold": int(row["inner_fold"]),
                "split_name": row["split_name"],
                "train_ids": tr_ids,
                "val_ids": va_ids,
                "x_train": x.loc[tr_ids],
                "y_train": y.loc[tr_ids],
                "meta_train": meta.loc[tr_ids],
                "x_val": x.loc[va_ids],
                "y_val": y.loc[va_ids],
                "meta_val": meta.loc[va_ids],
            }
        )

    test_ids = group_splits[0]["test_ids"]
    return {
        "target_year": int(group_splits[0]["target_year"]),
        "outer_fold": int(group_splits[0]["outer_fold"]),
        "test_ids": test_ids,
        "x_test": x.loc[test_ids],
        "y_test": y.loc[test_ids],
        "meta_test": meta.loc[test_ids],
        "inner_payload": inner_payload,
    }


def _safe_std(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    return float(series.std(ddof=0))


def _normalize_modality_combo(combo: str) -> str:
    token = str(combo).strip().upper().replace(" ", "")
    if not token:
        raise ValueError("modality_combo 不能为空。")
    parts = [x for x in token.split("+") if x]
    legal = {"H", "C", "G"}
    unknown = [x for x in parts if x not in legal]
    if unknown:
        raise ValueError(f"非法 modality_combo: {combo}")
    ordered = [key for key in ["H", "C", "G"] if key in parts]
    if not ordered:
        raise ValueError(f"modality_combo={combo} 没有可用模态。")
    return "+".join(ordered)


def _resolve_modality_combo(exp_cfg: dict) -> str:
    raw_single = exp_cfg.get("modality_combo")
    raw_multi = exp_cfg.get("modality_combos")
    if raw_single not in (None, "") and raw_multi not in (None, [], ""):
        raise ValueError("Result3.1 只能配置 experiment.modality_combo 或 experiment.modality_combos 之一。")
    if raw_multi not in (None, [], ""):
        if not isinstance(raw_multi, list) or len(raw_multi) != 1:
            raise ValueError("Result3.1 仅支持单个模态组合，请使用一个元素的 experiment.modality_combos。")
        raw_single = raw_multi[0]
    if raw_single in (None, ""):
        raw_single = "H+C+G"
    return _normalize_modality_combo(str(raw_single))


def _build_full_feature_matrix(bundle: MultiTargetDataBundle, *, modality_combo: str = "H+C+G") -> tuple[pd.DataFrame, dict]:
    combo = _normalize_modality_combo(modality_combo)
    tokens = set(combo.split("+"))

    h_cols = list(bundle.x_hyperspectral.columns) if "H" in tokens else []
    c_cols = list(bundle.x_climate.columns) if "C" in tokens else []
    g_cols = list(bundle.x_genotype.columns) if "G" in tokens else []
    if not (h_cols or c_cols or g_cols):
        raise ValueError(f"modality_combo={modality_combo} 没有可用列。")

    frames = []
    if h_cols:
        frames.append(bundle.x_hyperspectral[h_cols])
    if c_cols:
        frames.append(bundle.x_climate[c_cols])
    if g_cols:
        frames.append(bundle.x_genotype[g_cols])

    x = pd.concat(frames, axis=1)
    x = x.loc[bundle.meta.index].copy()
    x = sanitize_feature_matrix(x)
    info = {
        "modality_combo": combo,
        "n_samples": int(x.shape[0]),
        "n_features_total": int(x.shape[1]),
        "n_features_h": int(len(h_cols)),
        "n_features_c": int(len(c_cols)),
        "n_features_g": int(len(g_cols)),
        "n_time_keys_h": int(len(bundle.hyperspectral_tb_map) if h_cols else 0),
        "n_time_keys_c": int(len(bundle.climate_tb_map) if c_cols else 0),
        "first_time_key": bundle.common_tbs[0] if bundle.common_tbs else None,
        "last_time_key": bundle.common_tbs[-1] if bundle.common_tbs else None,
        "h_cols": h_cols,
        "c_cols": c_cols,
        "g_cols": g_cols,
    }
    return x, info


def _availability_row(timeline: str, x: pd.DataFrame, info: dict) -> dict:
    miss = x.isna()
    row_miss = miss.mean(axis=1)
    overall_missing_ratio = float(miss.values.mean()) if x.size > 0 else float("nan")
    return {
        "timeline": timeline,
        "n_samples": int(info["n_samples"]),
        "n_features_total": int(info["n_features_total"]),
        "n_features_h": int(info["n_features_h"]),
        "n_features_c": int(info["n_features_c"]),
        "n_features_g": int(info["n_features_g"]),
        "n_time_keys_h": int(info["n_time_keys_h"]),
        "n_time_keys_c": int(info["n_time_keys_c"]),
        "overall_missing_ratio": overall_missing_ratio,
        "row_missing_ratio_mean": float(row_miss.mean()),
        "row_missing_ratio_std": float(row_miss.std(ddof=0)),
        "fully_observed_sample_count": int((row_miss == 0).sum()),
        "partially_missing_sample_count": int(((row_miss > 0) & (row_miss <= 0.5)).sum()),
        "severely_missing_sample_count": int((row_miss > 0.5).sum()),
        "first_time_key": str(info.get("first_time_key")),
        "last_time_key": str(info.get("last_time_key")),
    }


def _aggregate_summary(metrics_fold_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_fold_df.empty:
        return pd.DataFrame()
    return (
        metrics_fold_df.groupby(["scenario", "timeline", "target", "predictor"], as_index=False)
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
            n_inner_folds=("n_inner_folds", "mean"),
        )
        .sort_values(["scenario", "timeline", "target", "predictor"])
        .reset_index(drop=True)
    )


def _aggregate_by_year(oof_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if oof_df.empty:
        return pd.DataFrame()
    for keys, sub in oof_df.groupby(["scenario", "timeline", "target", "predictor", "year"], dropna=False):
        m = regression_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        rows.append(
            {
                "scenario": keys[0],
                "timeline": keys[1],
                "target": keys[2],
                "predictor": keys[3],
                "year": int(keys[4]),
                "r2": float(m["r2"]),
                "rmse": float(m["rmse"]),
                "mae": float(m["mae"]),
                "pearson": float(m["pearson_r"]),
                "n_samples": int(len(sub)),
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario", "timeline", "target", "predictor", "year"]).reset_index(drop=True)


def _aggregate_overall(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    by_scenario_timeline = (
        summary_df.groupby(["scenario", "timeline"], as_index=False)
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

    by_scenario_target = (
        summary_df.groupby(["scenario", "target"], as_index=False)
        .agg(
            best_r2=("mean_r2", "max"),
            best_pearson=("mean_pearson", "max"),
            avg_r2=("mean_r2", "mean"),
            avg_pearson=("mean_pearson", "mean"),
        )
        .sort_values(["scenario", "target"])
        .reset_index(drop=True)
    )
    return by_scenario_timeline, by_scenario_target


def _plot_heatmap(df: pd.DataFrame, *, title: str, value_col: str, out_path: Path) -> None:
    if df.empty:
        return
    pivot = df.pivot(index="timeline", columns="predictor", values=value_col)
    timelines = list(pivot.index)
    predictors = list(pivot.columns)
    values = pivot.to_numpy(dtype=float)

    fig_w = max(6, len(predictors) * 1.25)
    fig_h = max(3.5, len(timelines) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(np.arange(len(predictors)))
    ax.set_xticklabels(predictors, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(timelines)))
    ax.set_yticklabels(timelines)
    ax.set_title(title)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            txt = "nan" if np.isnan(val) else f"{val:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _ordered_unique(values: list[str], preferred: list[str] | None = None) -> list[str]:
    preferred = preferred or []
    seen = set()
    ordered: list[str] = []
    for item in preferred + [str(x) for x in values]:
        token = str(item)
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _safe_filename_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_").lower()
    return token or "item"


def _plot_metric_target_faceted(
    df: pd.DataFrame,
    *,
    value_col: str,
    y_label: str,
    out_path: Path,
    scenario_label: str,
    target_order: list[str],
    timeline_order: list[str],
    title_suffix: str = "",
) -> None:
    if df.empty or value_col not in df.columns:
        return

    predictors = _ordered_unique(sorted(df["predictor"].astype(str).unique()))
    targets_present = [x for x in target_order if x in set(df["target"].astype(str).unique())]
    if not targets_present:
        targets_present = _ordered_unique(sorted(df["target"].astype(str).unique()), preferred=target_order)
    timelines_present = [x for x in timeline_order if x in set(df["timeline"].astype(str).unique())]
    if not timelines_present:
        timelines_present = _ordered_unique(sorted(df["timeline"].astype(str).unique()), preferred=timeline_order)
    if not predictors or not targets_present or not timelines_present:
        return

    n = len(predictors)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 4.2 * nrows), sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)

    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(i % 10) for i, name in enumerate(timelines_present)}
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    marker_map = {name: markers[i % len(markers)] for i, name in enumerate(timelines_present)}
    x_pos = np.arange(len(targets_present))

    for i, predictor in enumerate(predictors):
        ax = axes[i]
        sub = df[df["predictor"].astype(str) == predictor].copy()
        for timeline in timelines_present:
            part = sub[sub["timeline"].astype(str) == timeline].copy()
            if part.empty:
                continue
            values = []
            for target in targets_present:
                row = part[part["target"].astype(str) == target]
                values.append(float(row.iloc[0][value_col]) if not row.empty else np.nan)
            ax.plot(
                x_pos,
                values,
                color=colors[timeline],
                marker=marker_map[timeline],
                linewidth=1.6,
                markersize=4,
                alpha=0.95,
                label=timeline,
            )

        ax.set_title(f"{predictor} | {scenario_label}{title_suffix}")
        ax.set_xlabel("target")
        ax.set_ylabel(y_label)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(targets_present, rotation=30, ha="right")
        ax.grid(alpha=0.25)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles = [Line2D([0], [0], color=colors[name], marker=marker_map[name], lw=2, label=name) for name in timelines_present]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(4, len(handles)),
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_global_timeline_vs_target(
    df: pd.DataFrame,
    *,
    value_col: str,
    y_label: str,
    out_path: Path,
    scenario_label: str,
    target_order: list[str],
    timeline_order: list[str],
) -> None:
    if df.empty or value_col not in df.columns:
        return

    agg = (
        df.groupby(["timeline", "target"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "metric_value"})
    )
    targets_present = [x for x in target_order if x in set(agg["target"].astype(str).unique())]
    if not targets_present:
        targets_present = _ordered_unique(sorted(agg["target"].astype(str).unique()), preferred=target_order)
    timelines_present = [x for x in timeline_order if x in set(agg["timeline"].astype(str).unique())]
    if not timelines_present:
        timelines_present = _ordered_unique(sorted(agg["timeline"].astype(str).unique()), preferred=timeline_order)
    if not targets_present or not timelines_present:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(targets_present) * 1.7), 5.0))
    cmap = plt.get_cmap("tab10")
    width = 0.78 / max(1, len(timelines_present))
    x_pos = np.arange(len(targets_present))

    for idx, timeline in enumerate(timelines_present):
        part = agg[agg["timeline"].astype(str) == timeline]
        values = []
        for target in targets_present:
            row = part[part["target"].astype(str) == target]
            values.append(float(row.iloc[0]["metric_value"]) if not row.empty else np.nan)
        offsets = x_pos - 0.39 + width / 2 + idx * width
        ax.bar(offsets, values, width=width, color=cmap(idx % 10), alpha=0.9, label=timeline)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(targets_present, rotation=30, ha="right")
    ax.set_xlabel("target")
    ax.set_ylabel(y_label)
    ax.set_title(f"{scenario_label} | global {y_label} by target/timeline")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(4, len(timelines_present)))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_rank_and_stability(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rank_rows = []
    for keys, sub in summary_df.groupby(["scenario", "target", "predictor"], dropna=False):
        work = sub.copy()
        work["rank_r2"] = work["mean_r2"].rank(method="average", ascending=False)
        work["rank_rmse"] = work["mean_rmse"].rank(method="average", ascending=True)
        work["rank_mae"] = work["mean_mae"].rank(method="average", ascending=True)
        work["rank_pearson"] = work["mean_pearson"].rank(method="average", ascending=False)
        work["win_r2"] = (work["rank_r2"] == work["rank_r2"].min()).astype(int)
        work["win_pearson"] = (work["rank_pearson"] == work["rank_pearson"].min()).astype(int)
        for _, row in work.iterrows():
            rank_rows.append(
                {
                    "scenario": keys[0],
                    "target": keys[1],
                    "predictor": keys[2],
                    "timeline": str(row["timeline"]),
                    "rank_r2": float(row["rank_r2"]),
                    "rank_rmse": float(row["rank_rmse"]),
                    "rank_mae": float(row["rank_mae"]),
                    "rank_pearson": float(row["rank_pearson"]),
                    "win_r2": int(row["win_r2"]),
                    "win_pearson": int(row["win_pearson"]),
                }
            )

    rank_df = pd.DataFrame(rank_rows)
    if rank_df.empty:
        return rank_df, pd.DataFrame()

    rank_summary = (
        rank_df.groupby(["scenario", "timeline"], as_index=False)
        .agg(
            avg_rank_r2=("rank_r2", "mean"),
            avg_rank_rmse=("rank_rmse", "mean"),
            avg_rank_mae=("rank_mae", "mean"),
            avg_rank_pearson=("rank_pearson", "mean"),
            win_count_r2=("win_r2", "sum"),
            win_count_pearson=("win_pearson", "sum"),
        )
        .sort_values(["scenario", "avg_rank_r2", "avg_rank_pearson"])
        .reset_index(drop=True)
    )

    stability_df = (
        summary_df.groupby(["scenario", "timeline"], as_index=False)
        .agg(
            avg_std_r2=("std_r2", "mean"),
            avg_std_rmse=("std_rmse", "mean"),
            avg_std_mae=("std_mae", "mean"),
            avg_std_pearson=("std_pearson", "mean"),
        )
        .sort_values(["scenario", "timeline"])
        .reset_index(drop=True)
    )
    return rank_summary, stability_df


def _plot_rank_summary(rank_df: pd.DataFrame, *, out_path: Path, scenario_label: str, timeline_order: list[str]) -> None:
    if rank_df.empty:
        return

    order = [x for x in timeline_order if x in set(rank_df["timeline"].astype(str).unique())]
    if not order:
        order = _ordered_unique(sorted(rank_df["timeline"].astype(str).unique()), preferred=timeline_order)

    metric_specs = [
        ("avg_rank_r2", "avg rank (R2)"),
        ("avg_rank_rmse", "avg rank (RMSE)"),
        ("avg_rank_mae", "avg rank (MAE)"),
        ("avg_rank_pearson", "avg rank (Pearson)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = np.array(axes).reshape(-1)
    cmap = plt.get_cmap("tab10")

    for ax, (col, title) in zip(axes, metric_specs):
        values = []
        for name in order:
            row = rank_df[rank_df["timeline"].astype(str) == name]
            values.append(float(row.iloc[0][col]) if not row.empty else np.nan)
        ax.bar(order, values, color=[cmap(i % 10) for i, _ in enumerate(order)], alpha=0.9)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"{scenario_label} | rank summary", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_stability_summary(
    stability_df: pd.DataFrame,
    *,
    out_path: Path,
    scenario_label: str,
    timeline_order: list[str],
) -> None:
    if stability_df.empty:
        return

    order = [x for x in timeline_order if x in set(stability_df["timeline"].astype(str).unique())]
    if not order:
        order = _ordered_unique(sorted(stability_df["timeline"].astype(str).unique()), preferred=timeline_order)

    metric_specs = [
        ("avg_std_r2", "mean std(R2)"),
        ("avg_std_rmse", "mean std(RMSE)"),
        ("avg_std_mae", "mean std(MAE)"),
        ("avg_std_pearson", "mean std(Pearson)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = np.array(axes).reshape(-1)
    cmap = plt.get_cmap("tab10")

    for ax, (col, title) in zip(axes, metric_specs):
        values = []
        for name in order:
            row = stability_df[stability_df["timeline"].astype(str) == name]
            values.append(float(row.iloc[0][col]) if not row.empty else np.nan)
        ax.bar(order, values, color=[cmap(i % 10) for i, _ in enumerate(order)], alpha=0.9)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"{scenario_label} | stability summary", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _generate_figures(
    summary_df: pd.DataFrame,
    by_year_df: pd.DataFrame,
    fig_dir: Path,
    *,
    scenario_order: list[str] | None = None,
    target_order: list[str] | None = None,
    timeline_order: list[str] | None = None,
) -> None:
    ensure_dir(fig_dir)
    if summary_df.empty:
        return

    scenario_order = _ordered_unique(
        sorted(summary_df["scenario"].astype(str).unique()),
        preferred=scenario_order or ["within_season", "loso", "loso_genotype"],
    )
    target_order = _ordered_unique(
        sorted(summary_df["target"].astype(str).unique()),
        preferred=target_order or ["ActualYD", "CM", "LM", "PHM", "Spike", "TKW"],
    )
    timeline_order = _ordered_unique(
        sorted(summary_df["timeline"].astype(str).unique()),
        preferred=timeline_order or ["day", "day_rel_heading", "gdd_abs", "gdd_rel_heading"],
    )

    metric_specs = [
        ("mean_r2", "mean R2", "figure_r2_by_target_faceted"),
        ("mean_rmse", "mean RMSE", "figure_rmse_by_target_faceted"),
        ("mean_mae", "mean MAE", "figure_mae_by_target_faceted"),
        ("mean_pearson", "mean Pearson", "figure_pearson_by_target_faceted"),
    ]
    year_metric_specs = [
        ("r2", "R2", "figure_r2_by_target_faceted"),
        ("rmse", "RMSE", "figure_rmse_by_target_faceted"),
        ("mae", "MAE", "figure_mae_by_target_faceted"),
        ("pearson", "Pearson", "figure_pearson_by_target_faceted"),
    ]

    for (scenario, target), sub in summary_df.groupby(["scenario", "target"], dropna=False):
        part = sub[["timeline", "predictor", "mean_r2", "mean_pearson"]].copy()
        _plot_heatmap(
            part,
            title=f"{scenario} | {target} | mean R2",
            value_col="mean_r2",
            out_path=fig_dir / f"figure_r2_heatmap_{scenario}_{target}.png",
        )
        _plot_heatmap(
            part,
            title=f"{scenario} | {target} | mean Pearson",
            value_col="mean_pearson",
            out_path=fig_dir / f"figure_pearson_heatmap_{scenario}_{target}.png",
        )

    overall = (
        summary_df.groupby(["scenario", "timeline", "predictor"], as_index=False)
        .agg(mean_r2=("mean_r2", "mean"), mean_pearson=("mean_pearson", "mean"))
        .sort_values(["scenario", "timeline", "predictor"])
    )
    for scenario, sub in overall.groupby("scenario", dropna=False):
        part = sub[["timeline", "predictor", "mean_r2", "mean_pearson"]].copy()
        _plot_heatmap(
            part,
            title=f"{scenario} | overall mean R2",
            value_col="mean_r2",
            out_path=fig_dir / f"figure_r2_heatmap_{scenario}_overall.png",
        )
        _plot_heatmap(
            part,
            title=f"{scenario} | overall mean Pearson",
            value_col="mean_pearson",
            out_path=fig_dir / f"figure_pearson_heatmap_{scenario}_overall.png",
        )

    for scenario in scenario_order:
        sub = summary_df[summary_df["scenario"].astype(str) == scenario].copy()
        if sub.empty:
            continue
        for value_col, y_label, file_prefix in metric_specs:
            _plot_metric_target_faceted(
                sub,
                value_col=value_col,
                y_label=y_label,
                out_path=fig_dir / f"{file_prefix}_{_safe_filename_token(scenario)}.png",
                scenario_label=scenario,
                target_order=target_order,
                timeline_order=timeline_order,
            )

        by_scenario_year = by_year_df[by_year_df["scenario"].astype(str) == scenario].copy() if not by_year_df.empty else pd.DataFrame()
        if not by_scenario_year.empty:
            for year in sorted(by_scenario_year["year"].dropna().astype(int).unique()):
                year_sub = by_scenario_year[by_scenario_year["year"].astype(int) == int(year)].copy()
                for value_col, y_label, file_prefix in year_metric_specs:
                    _plot_metric_target_faceted(
                        year_sub,
                        value_col=value_col,
                        y_label=y_label,
                        out_path=fig_dir / f"{file_prefix}_{_safe_filename_token(scenario)}_year{int(year)}.png",
                        scenario_label=scenario,
                        target_order=target_order,
                        timeline_order=timeline_order,
                        title_suffix=f" (year={int(year)})",
                    )

        _plot_global_timeline_vs_target(
            sub,
            value_col="mean_r2",
            y_label="global mean R2",
            out_path=fig_dir / f"figure_global_r2_by_target_timeline_{_safe_filename_token(scenario)}.png",
            scenario_label=scenario,
            target_order=target_order,
            timeline_order=timeline_order,
        )
        _plot_global_timeline_vs_target(
            sub,
            value_col="mean_pearson",
            y_label="global mean Pearson",
            out_path=fig_dir / f"figure_global_pearson_by_target_timeline_{_safe_filename_token(scenario)}.png",
            scenario_label=scenario,
            target_order=target_order,
            timeline_order=timeline_order,
        )

    rank_df, stability_df = _build_rank_and_stability(summary_df)
    for scenario in scenario_order:
        rank_sub = rank_df[rank_df["scenario"].astype(str) == scenario].copy() if not rank_df.empty else pd.DataFrame()
        stability_sub = (
            stability_df[stability_df["scenario"].astype(str) == scenario].copy() if not stability_df.empty else pd.DataFrame()
        )
        _plot_rank_summary(
            rank_sub,
            out_path=fig_dir / f"figure_rank_summary_{_safe_filename_token(scenario)}.png",
            scenario_label=scenario,
            timeline_order=timeline_order,
        )
        _plot_stability_summary(
            stability_sub,
            out_path=fig_dir / f"figure_stability_summary_{_safe_filename_token(scenario)}.png",
            scenario_label=scenario,
            timeline_order=timeline_order,
        )


def regenerate_result3_1_figures(output_dir: Path) -> None:
    summary_path = output_dir / "metrics_summary.csv"
    by_year_path = output_dir / "metrics_by_year.csv"
    run_config_path_yaml = output_dir / "run_config.yaml"
    run_config_path_json = output_dir / "run_config.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"缺少文件: {summary_path}")
    if not by_year_path.exists():
        raise FileNotFoundError(f"缺少文件: {by_year_path}")

    summary_df = pd.read_csv(summary_path)
    by_year_df = pd.read_csv(by_year_path)

    cfg = {}
    if run_config_path_yaml.exists():
        cfg = read_yaml(run_config_path_yaml)
    elif run_config_path_json.exists():
        cfg = json.loads(run_config_path_json.read_text(encoding="utf-8"))

    exp_cfg = cfg.get("experiment", {}) if isinstance(cfg, dict) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}

    target_order = [str(x) for x in exp_cfg.get("targets", [])] or None
    scenario_order = list((data_cfg.get("scenarios") or {}).keys()) or None
    timeline_order = list((data_cfg.get("timeline_dirs") or {}).keys()) or None

    _generate_figures(
        summary_df,
        by_year_df,
        output_dir / "figures",
        scenario_order=scenario_order,
        target_order=target_order,
        timeline_order=timeline_order,
    )


def _markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "(empty)"
    part = df.head(max_rows).copy()
    try:
        return part.to_markdown(index=False)
    except Exception:
        # Fallback when optional dependency `tabulate` is unavailable.
        return "```text\n" + part.to_string(index=False) + "\n```"


def _write_summary(
    *,
    output_dir: Path,
    cfg: dict,
    feature_overview_df: pd.DataFrame,
    split_usage_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    by_year_df: pd.DataFrame,
    scenario_timeline_df: pd.DataFrame,
    scenario_target_df: pd.DataFrame,
) -> None:
    modality_combo = _resolve_modality_combo(cfg.get("experiment", {}))
    timeline_names = list(_resolve_timeline_dirs(cfg.get("data", {})).keys())
    scenario_names = list(_resolve_scenarios_cfg(cfg.get("data", {})).keys())
    lines = []
    lines.append("# Result3.1 Summary")
    lines.append("")
    lines.append("## 1. 实验定义")
    lines.append(f"- 输入：完整时间点 `{modality_combo}` 全量特征，不做 anchor 截断。")
    lines.append("- 时间轴：`" + "`, `".join(timeline_names) + "`。")
    lines.append("- 场景：`" + "`, `".join(scenario_names) + "`。")
    lines.append(f"- 性状：`{', '.join(cfg['experiment']['targets'])}`。")
    lines.append(f"- 预测器：`{', '.join(cfg['experiment']['predictors_run'])}`。")
    lines.append(f"- 基因型表示：`{cfg['experiment'].get('genotype_representation', 'grm_pca')}`。")
    lines.append("")
    lines.append("## 2. 特征概览")
    lines.append(_markdown_table(feature_overview_df))
    lines.append("")
    lines.append("## 3. split 使用概览")
    lines.append(_markdown_table(split_usage_df))
    lines.append("")
    lines.append("## 4. 场景 x 时间轴总体结果")
    lines.append(_markdown_table(scenario_timeline_df))
    lines.append("")
    lines.append("## 5. 场景 x 性状总体结果")
    lines.append(_markdown_table(scenario_target_df))
    lines.append("")
    lines.append("## 6. 明细结果预览")
    lines.append(_markdown_table(summary_df))
    lines.append("")
    lines.append("## 7. 按年份 OOF 聚合结果预览")
    lines.append(_markdown_table(by_year_df))
    lines.append("")
    lines.append("## 8. 结果文件")
    lines.append("- `feature_overview.csv`")
    lines.append("- `split_usage_summary.csv`")
    lines.append("- `metrics_by_fold.csv`")
    lines.append("- `metrics_summary.csv`")
    lines.append("- `metrics_by_year.csv`")
    lines.append("- `metrics_by_scenario_timeline.csv`")
    lines.append("- `metrics_by_scenario_target.csv`")
    lines.append("- `oof_predictions.parquet`")
    lines.append("- `outer_test_inner_ensemble_predictions.parquet`")
    lines.append("- `optuna_trials.csv`")
    lines.append("- `run_config.yaml`")
    lines.append("- `figures/*.png`")
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def finalize_result3_1_outputs(
    *,
    output_dir: Path,
    cfg: dict,
    feature_overview_df: pd.DataFrame,
    split_usage_df: pd.DataFrame,
    metrics_fold_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    inner_pred_df: pd.DataFrame,
    trial_df: pd.DataFrame,
) -> None:
    feature_overview_df = (
        feature_overview_df.sort_values(["timeline"]).reset_index(drop=True)
        if not feature_overview_df.empty and "timeline" in feature_overview_df.columns
        else feature_overview_df
    )
    split_usage_df = (
        split_usage_df.sort_values(["scenario", "timeline", "target"]).reset_index(drop=True)
        if not split_usage_df.empty and {"scenario", "timeline", "target"}.issubset(split_usage_df.columns)
        else split_usage_df
    )
    metrics_fold_df = (
        metrics_fold_df.sort_values(["scenario", "timeline", "target", "predictor", "target_year", "outer_fold"]).reset_index(drop=True)
        if not metrics_fold_df.empty
        else metrics_fold_df
    )
    oof_df = (
        oof_df.sort_values(["scenario", "timeline", "target", "predictor", "year", "plot_id"]).reset_index(drop=True)
        if not oof_df.empty
        else oof_df
    )
    inner_pred_df = (
        inner_pred_df.sort_values(
            ["scenario", "timeline", "target", "predictor", "target_year", "outer_fold", "inner_fold", "plot_id"]
        ).reset_index(drop=True)
        if not inner_pred_df.empty
        else inner_pred_df
    )
    trial_df = (
        trial_df.sort_values(["scenario", "timeline", "target", "predictor", "target_year", "outer_fold", "rank"]).reset_index(drop=True)
        if not trial_df.empty
        else trial_df
    )

    summary_df = _aggregate_summary(metrics_fold_df)
    by_year_df = _aggregate_by_year(oof_df)
    scenario_timeline_df, scenario_target_df = _aggregate_overall(summary_df)

    feature_overview_df.to_csv(output_dir / "feature_overview.csv", index=False)
    split_usage_df.to_csv(output_dir / "split_usage_summary.csv", index=False)
    metrics_fold_df.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    by_year_df.to_csv(output_dir / "metrics_by_year.csv", index=False)
    scenario_timeline_df.to_csv(output_dir / "metrics_by_scenario_timeline.csv", index=False)
    scenario_target_df.to_csv(output_dir / "metrics_by_scenario_target.csv", index=False)
    if not oof_df.empty:
        oof_df.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    if not inner_pred_df.empty:
        inner_pred_df.to_parquet(output_dir / "outer_test_inner_ensemble_predictions.parquet", index=False)
    trial_df.to_csv(output_dir / "optuna_trials.csv", index=False)

    figures_dir = output_dir / "figures"
    _generate_figures(
        summary_df,
        by_year_df,
        figures_dir,
        scenario_order=list(scenarios_cfg.keys()) if isinstance((scenarios_cfg := cfg.get("data", {}).get("scenarios")), dict) else None,
        target_order=[str(x) for x in cfg.get("experiment", {}).get("targets", [])] or None,
        timeline_order=list(timeline_dirs_cfg.keys()) if isinstance((timeline_dirs_cfg := cfg.get("data", {}).get("timeline_dirs")), dict) else None,
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
        feature_overview_df=feature_overview_df,
        split_usage_df=split_usage_df,
        summary_df=summary_df,
        by_year_df=by_year_df,
        scenario_timeline_df=scenario_timeline_df,
        scenario_target_df=scenario_target_df,
    )


def run_result3_1(config_path: Path) -> Path:
    cfg = read_yaml(config_path)
    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("modeling", {})
    optuna_cfg = cfg.get("optuna", {})
    preprocess_cfg = cfg.get("preprocessing", {})
    output_cfg = cfg.get("output", {})

    output_dir = _resolve_output_dir(output_cfg)
    progress_enabled = bool(output_cfg.get("progress_log", True))
    _append_progress(output_dir, enabled=progress_enabled, event="run_start", payload={"config_path": str(config_path)})

    seed = int(exp_cfg.get("random_seed", 42))
    _set_global_seed(seed)

    genotype_representation = str(exp_cfg.get("genotype_representation", "grm_pca"))
    modality_combo = _resolve_modality_combo(exp_cfg)
    targets = [str(x) for x in exp_cfg.get("targets", ["ActualYD", "CM", "LM", "PHM", "Spike", "TKW"])]
    timeline_dirs = _resolve_timeline_dirs(data_cfg)
    scenarios_cfg = _resolve_scenarios_cfg(data_cfg)
    backend_cfg = exp_cfg.get("model_backends", {})
    active_predictors = _resolve_active_predictors(exp_cfg)
    task_filter_set = _resolve_task_filter_set(exp_cfg)
    planned_tasks = enumerate_result31_tasks_from_config(cfg)
    allowed_task_keys = {
        (task["scenario"], task["timeline"], task["target"], task["predictor"]) for task in planned_tasks
    }
    if not planned_tasks:
        raise RuntimeError("没有匹配到可执行的 Result3.1 任务。")

    n_trials = int(optuna_cfg.get("n_trials", 3))
    lambda_gap = float(optuna_cfg.get("lambda_gap", 1.0))
    pruner_startup_trials = int(optuna_cfg.get("pruner_startup_trials", 1))

    timeline_bundles: Dict[str, MultiTargetDataBundle] = {}
    timeline_features: Dict[str, pd.DataFrame] = {}
    timeline_feature_info: Dict[str, dict] = {}
    feature_overview_rows = []

    for timeline_name, timeline_path in timeline_dirs.items():
        print(f"[LOAD] timeline={timeline_name} path={timeline_path}", flush=True)
        bundle = load_multitarget_model_inputs(Path(timeline_path), genotype_representation=genotype_representation)
        x_full, info = _build_full_feature_matrix(bundle, modality_combo=modality_combo)
        timeline_bundles[timeline_name] = bundle
        timeline_features[timeline_name] = x_full
        timeline_feature_info[timeline_name] = info
        feature_overview_rows.append(_availability_row(timeline_name, x_full, info))

    total_tasks = len(planned_tasks)
    task_idx = 0

    metrics_fold_rows = []
    oof_rows = []
    inner_pred_rows = []
    trial_rows = []
    split_usage_rows = []

    task_filters_cfg = exp_cfg.get("task_filters", {}) if isinstance(exp_cfg.get("task_filters"), dict) else {}
    scenario_exec_order = list(scenarios_cfg.keys())
    timeline_exec_order = list(timeline_dirs.keys())
    scenario_seed_order = [str(x) for x in task_filters_cfg.get("scenario_order", scenario_exec_order)]
    timeline_seed_order = [str(x) for x in task_filters_cfg.get("timeline_order", timeline_exec_order)]
    target_seed_order = [str(x) for x in task_filters_cfg.get("target_order", targets)]
    predictor_seed_order = [str(x).lower() for x in task_filters_cfg.get("predictor_order", active_predictors)]

    for scenario_name, scenario_cfg in scenarios_cfg.items():
        for timeline_name in timeline_exec_order:
            bundle = timeline_bundles[timeline_name]
            x_full = timeline_features[timeline_name]
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
                        {
                            "scenario": scenario_name,
                            "timeline": timeline_name,
                            "target": target_col,
                            "status": "missing_target_column",
                        }
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

                for predictor in active_predictors:
                    task_key = (scenario_name, timeline_name, target_col, predictor)
                    if task_filter_set and task_key not in allowed_task_keys:
                        continue
                    task_idx += 1
                    print(
                        f"[TASK {task_idx}/{total_tasks}] scenario={scenario_name} tl={timeline_name} target={target_col} predictor={predictor} features={x_full.shape[1]}",
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
                            "predictor": predictor,
                            "n_features": int(x_full.shape[1]),
                        },
                    )

                    ordered_outer_items = normalize_outer_groups(prepared_groups)
                    ordered_outer_keys = [key for key, _ in ordered_outer_items]
                    reuse_policy = resolve_optuna_reuse_policy(optuna_cfg)
                    cached_best_params = None
                    cached_source = None

                    for (target_year, outer_fold), group_splits in ordered_outer_items:
                        payload = _build_outer_payload(x_full, y_full, meta_full, group_splits)
                        inner_payload = payload["inner_payload"]
                        x_test = payload["x_test"]
                        y_test = payload["y_test"]
                        meta_test = payload["meta_test"]
                        if len(x_test) == 0:
                            continue

                        s_idx = scenario_seed_order.index(scenario_name)
                        t_idx = timeline_seed_order.index(timeline_name)
                        p_idx = predictor_seed_order.index(predictor)
                        g_idx = target_seed_order.index(target_col)
                        sampler_seed = (
                            seed
                            + s_idx * 1_000_000
                            + t_idx * 100_000
                            + g_idx * 10_000
                            + p_idx * 1_000
                            + int(target_year % 100) * 10
                            + int(outer_fold)
                        )

                        optuna_search_performed = should_search_on_outer_fold(
                            ordered_outer_keys=ordered_outer_keys,
                            current_key=(target_year, outer_fold),
                            reuse_policy=reuse_policy,
                        )
                        if reuse_policy["enabled"] and cached_best_params is None:
                            optuna_search_performed = True

                        if optuna_search_performed:
                            sampler = optuna.samplers.TPESampler(seed=int(sampler_seed))
                            pruner = optuna.pruners.MedianPruner(n_startup_trials=pruner_startup_trials, n_warmup_steps=1)
                            study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

                            def objective(trial):
                                params = suggest_params(trial, predictor, model_cfg=backend_cfg)
                                train_losses = []
                                val_losses = []
                                train_metrics_list = []
                                val_metrics_list = []

                                for step_i, inner in enumerate(inner_payload):
                                    pipe = build_pipeline(
                                        predictor=predictor,
                                        params=params,
                                        n_features=inner["x_train"].shape[1],
                                        preprocess_cfg=preprocess_cfg,
                                        seed=int(sampler_seed) + trial.number + step_i,
                                        model_cfg=backend_cfg,
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

                                trial.set_user_attr("mean_train_loss", mean_train)
                                trial.set_user_attr("mean_val_loss", mean_val)
                                trial.set_user_attr("gap", gap)
                                trial.set_user_attr("objective", obj)
                                trial.set_user_attr("mean_train_r2", mean_train_metrics["r2"])
                                trial.set_user_attr("mean_train_rmse", mean_train_metrics["rmse"])
                                trial.set_user_attr("mean_train_mae", mean_train_metrics["mae"])
                                trial.set_user_attr("mean_train_pearson", mean_train_metrics["pearson_r"])
                                trial.set_user_attr("mean_val_r2", mean_val_metrics["r2"])
                                trial.set_user_attr("mean_val_rmse", mean_val_metrics["rmse"])
                                trial.set_user_attr("mean_val_mae", mean_val_metrics["mae"])
                                trial.set_user_attr("mean_val_pearson", mean_val_metrics["pearson_r"])
                                return obj

                            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

                            complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
                            if not complete_trials:
                                continue

                            sorted_trials = sorted(complete_trials, key=lambda t: t.value)
                            rank_map = {t.number: i + 1 for i, t in enumerate(sorted_trials)}
                            best = study.best_trial
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
                                        "predictor": predictor,
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

                        best_params = dict(cached_best_params) if cached_best_params is not None else dict(best.params)
                        best_attrs = dict(cached_source["attrs"]) if optuna_search_performed and cached_source else {}
                        source_target_year = cached_source["target_year"] if cached_source else np.nan
                        source_outer_fold = cached_source["outer_fold"] if cached_source else np.nan
                        source_trial_number = cached_source["trial_number"] if cached_source else np.nan
                        source_objective = cached_source["objective"] if cached_source else np.nan
                        inner_preds = []
                        for inner in inner_payload:
                            pipe = build_pipeline(
                                predictor=predictor,
                                params=best_params,
                                n_features=inner["x_train"].shape[1],
                                preprocess_cfg=preprocess_cfg,
                                seed=int(sampler_seed) + int(inner["inner_fold"]) + 17,
                                model_cfg=backend_cfg,
                            )
                            pipe.fit(inner["x_train"], inner["y_train"])
                            inner_preds.append(pipe.predict(x_test))

                        pred_mat = np.column_stack(inner_preds)
                        pred_ens = pred_mat.mean(axis=1)
                        y_true = y_test.to_numpy()
                        fold_metrics = regression_metrics(y_true, pred_ens)

                        for i, pid in enumerate(x_test.index.tolist()):
                            y_true_i = float(y_true[i])
                            y_ens_i = float(pred_ens[i])
                            for j, inner in enumerate(inner_payload):
                                inner_pred_rows.append(
                                    {
                                        "plot_id": pid,
                                        "year": int(meta_test.loc[pid, "year"]),
                                        "scenario": scenario_name,
                                        "timeline": timeline_name,
                                        "target": target_col,
                                        "predictor": predictor,
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
                                    "predictor": predictor,
                                    "target_year": int(target_year),
                                    "outer_fold": int(outer_fold),
                                    "y_true": y_true_i,
                                    "y_pred": y_ens_i,
                                }
                            )

                        metrics_fold_rows.append(
                            {
                                "scenario": scenario_name,
                                "timeline": timeline_name,
                                "target": target_col,
                                "predictor": predictor,
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
                                "val_r2": float(best_attrs.get("mean_val_r2", np.nan)),
                                "val_rmse": float(best_attrs.get("mean_val_rmse", np.nan)),
                                "val_mae": float(best_attrs.get("mean_val_mae", np.nan)),
                                "val_pearson": float(best_attrs.get("mean_val_pearson", np.nan)),
                                "test_r2": float(fold_metrics["r2"]),
                                "test_rmse": float(fold_metrics["rmse"]),
                                "test_mae": float(fold_metrics["mae"]),
                                "test_pearson": float(fold_metrics["pearson_r"]),
                                "n_train": int(np.mean([len(inner["train_ids"]) for inner in inner_payload])),
                                "n_val": int(np.mean([len(inner["val_ids"]) for inner in inner_payload])),
                                "n_test": int(len(payload["test_ids"])),
                                "n_inner_folds": int(len(inner_payload)),
                                "n_features_total": int(timeline_feature_info[timeline_name]["n_features_total"]),
                                "n_features_h": int(timeline_feature_info[timeline_name]["n_features_h"]),
                                "n_features_c": int(timeline_feature_info[timeline_name]["n_features_c"]),
                                "n_features_g": int(timeline_feature_info[timeline_name]["n_features_g"]),
                                "n_optuna_trials": int(cached_source["n_trials"]) if optuna_search_performed and cached_source else 0,
                                "optuna_search_performed": bool(optuna_search_performed),
                                "optuna_reuse_scope": reuse_policy["mode"],
                                "optuna_source_target_year": source_target_year,
                                "optuna_source_outer_fold": source_outer_fold,
                                "optuna_source_trial_number": source_trial_number,
                                "optuna_source_objective": source_objective,
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
                            "predictor": predictor,
                        },
                    )

    feature_overview_df = pd.DataFrame(feature_overview_rows)
    split_usage_df = pd.DataFrame(split_usage_rows)
    metrics_fold_df = pd.DataFrame(metrics_fold_rows)
    oof_df = pd.DataFrame(oof_rows)
    inner_pred_df = pd.DataFrame(inner_pred_rows)
    trial_df = pd.DataFrame(trial_rows)
    finalize_result3_1_outputs(
        output_dir=output_dir,
        cfg={
            "config_path": str(config_path),
            "data": data_cfg,
            "experiment": {
                **exp_cfg,
                "active_predictors": active_predictors,
                "predictors_run": active_predictors,
                "targets": targets,
                "genotype_representation": genotype_representation,
            },
            "optuna": optuna_cfg,
            "preprocessing": preprocess_cfg,
        },
        feature_overview_df=feature_overview_df,
        split_usage_df=split_usage_df,
        metrics_fold_df=metrics_fold_df,
        oof_df=oof_df,
        inner_pred_df=inner_pred_df,
        trial_df=trial_df,
    )

    _append_progress(output_dir, enabled=progress_enabled, event="run_end", payload={"output_dir": output_dir.as_posix()})
    print(f"Result3.1 completed: {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Result3.1 full-season H+C+G multitrait experiments.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run_result3_1(Path(args.config))


if __name__ == "__main__":
    main()
