from __future__ import annotations

import argparse
import json
import os
import random
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".tmp_mpl").resolve()))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState
from sklearn.exceptions import ConvergenceWarning

from models.anchorwise.data_loader import MultiTargetDataBundle, load_multitarget_model_inputs
from models.anchorwise.feature_builder import AnchorDefinition, build_anchor_definitions, build_feature_matrix
from models.anchorwise.modeling import HAS_LIGHTGBM, build_pipeline, sanitize_feature_matrix, suggest_params
from models.common.io_utils import ensure_dir, get_git_commit_hash, now_iso, read_yaml, write_json, write_yaml
from models.common.metrics import mean_regression_metrics, regression_metrics
from models.common.optuna_reuse import normalize_outer_groups, resolve_optuna_reuse_policy, should_search_on_outer_fold
from models.common.split_loader import WithinSeasonSplit, load_split_groups
from models.multimodal_ablation.runner import (
    DEFAULT_MODALITY_COMBOS,
    _availability_row,
    _build_metrics_by_year as _ablation_build_metrics_by_year,
    _build_metrics_summary as _ablation_build_metrics_summary,
    _build_modality_feature_matrix,
    _build_rank_tables as _ablation_build_rank_tables,
    _parse_modality_combos,
    _plot_global_metric_by_modality,
    _plot_modality_group_views,
    _plot_overall_metrics,
    _plot_rank_summary as _ablation_plot_rank_summary,
    _plot_stability_summary as _ablation_plot_stability_summary,
    _plot_timeline_group_views,
    _plot_year_metrics,
)
from models.result31.runner import _resolve_split_groups_for_scenario

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
            output_cfg.get("output_dir", "outputs/experiments/result3_2_multitimeline_multimodal_multitrait"),
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


def _default_data_cfg() -> dict:
    return {
        "timeline_dirs": {
            "gdd_abs": "data/processed/model_inputs_engineered/gdd_abs",
            "gdd_rel_heading": "data/processed/model_inputs_engineered/gdd_rel_heading",
        },
        "scenarios": {
            "within_season": {
                "split_dir": "data/processed/splits",
                "validation_scenario": "within_season",
            },
            "loso": {
                "split_dir": "data/processed/splits",
                "validation_scenario": "loso",
            },
            "loso_genotype": {
                "split_dir": "data/processed/splits_loso_genotype_nested",
                "validation_scenario": "loso_genotype",
            },
        },
    }


def _resolve_scenarios_cfg(data_cfg: dict) -> dict:
    return data_cfg.get("scenarios", _default_data_cfg()["scenarios"])


def _resolve_timeline_dirs(data_cfg: dict, exp_cfg: dict | None = None) -> dict:
    exp_cfg = exp_cfg or {}
    if isinstance(data_cfg.get("timeline_dirs"), dict) and data_cfg.get("timeline_dirs"):
        return {str(k): str(v) for k, v in data_cfg["timeline_dirs"].items()}
    if data_cfg.get("input_dir"):
        axis_name = str(exp_cfg.get("axis_type", exp_cfg.get("timeline", "day_rel_heading")))
        return {axis_name: str(data_cfg["input_dir"])}
    return _default_data_cfg()["timeline_dirs"]


def _resolve_modality_combos(exp_cfg: dict) -> list[str]:
    raw = exp_cfg.get("modality_combos")
    if raw in (None, [], ""):
        raw = ["H+C+G"]
    return _parse_modality_combos(raw)


def _resolve_active_predictors(exp_cfg: dict) -> list[str]:
    predictors_supported = [
        str(x).lower()
        for x in exp_cfg.get(
            "predictors_supported",
            ["ridge", "lasso", "elasticnet", "lightgbm", "random_forest"],
        )
    ]
    predictors_run = [str(x).lower() for x in exp_cfg.get("predictors_run", predictors_supported)]
    skip_lightgbm_runtime = bool(exp_cfg.get("skip_lightgbm_runtime", False))
    backend_cfg = exp_cfg.get("model_backends", {})

    active_predictors: List[str] = []
    for predictor in predictors_run:
        if predictor not in predictors_supported:
            continue
        if predictor == "lightgbm" and skip_lightgbm_runtime:
            continue
        if predictor == "lightgbm" and str(backend_cfg.get("lightgbm_backend", "native")).lower() == "native" and not HAS_LIGHTGBM:
            continue
        active_predictors.append(predictor)
    if not active_predictors:
        raise RuntimeError("没有可运行的预测器。")
    return active_predictors


def _normalize_task_spec(spec: dict) -> tuple[str, str, str, str, str]:
    required = ["scenario", "timeline", "target", "modality_combo", "predictor"]
    missing = [k for k in required if k not in spec]
    if missing:
        raise KeyError(f"task_spec 缺少字段: {missing}")
    return (
        str(spec["scenario"]),
        str(spec["timeline"]),
        str(spec["target"]),
        str(spec["modality_combo"]),
        str(spec["predictor"]).lower(),
    )


def _resolve_task_filter_set(exp_cfg: dict) -> set[tuple[str, str, str, str, str]]:
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


def enumerate_result32_tasks_from_config(cfg: dict) -> list[dict]:
    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})

    timeline_dirs = _resolve_timeline_dirs(data_cfg, exp_cfg)
    scenarios_cfg = _resolve_scenarios_cfg(data_cfg)
    targets = [str(x) for x in exp_cfg.get("targets", ["ActualYD", "CM", "LM", "PHM", "Spike", "TKW"])]
    active_predictors = _resolve_active_predictors(exp_cfg)
    modality_combos = _resolve_modality_combos(exp_cfg)
    task_filter_set = _resolve_task_filter_set(exp_cfg)

    tasks: list[dict] = []
    for scenario_name in scenarios_cfg.keys():
        for timeline_name in timeline_dirs.keys():
            for target_col in targets:
                for modality_combo in modality_combos:
                    for predictor in active_predictors:
                        task_key = (scenario_name, timeline_name, target_col, modality_combo, predictor)
                        if task_filter_set and task_key not in task_filter_set:
                            continue
                        tasks.append(
                            {
                                "scenario": scenario_name,
                                "timeline": timeline_name,
                                "target": target_col,
                                "modality_combo": modality_combo,
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
                "x_val": x.loc[va_ids],
                "y_val": y.loc[va_ids],
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


def _markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "(empty)"
    part = df.head(max_rows).copy()
    try:
        return part.to_markdown(index=False)
    except Exception:
        return "```text\n" + part.to_string(index=False) + "\n```"


def _build_anchor_cache(
    timeline_name: str,
    bundle: MultiTargetDataBundle,
    anchors: list[AnchorDefinition],
    modality_combos: list[str],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    anchor_cache: dict[int, dict[str, Any]] = {}
    feature_rows: list[dict[str, Any]] = []
    anchor_json: list[dict[str, Any]] = []

    for anchor in anchors:
        x_all, x_info = build_feature_matrix(bundle, anchor.anchor_key, "temporal")
        x_all = sanitize_feature_matrix(x_all)
        anchor_cache[int(anchor.anchor_idx)] = {
            "anchor": anchor,
            "x_all": x_all,
            "x_info": x_info,
        }

        anchor_record = {
            **anchor.__dict__,
            "timeline": timeline_name,
            "input_type": "temporal",
            "n_features_h_full": int(x_info["n_features_h"]),
            "n_features_c_full": int(x_info["n_features_c"]),
            "n_features_g_full": int(x_info["n_features_g"]),
            "climate_prefix_columns": list(x_info.get("c_cols", [])),
            "hyperspectral_prefix_columns": list(x_info.get("h_cols", [])),
            "genotype_columns": list(x_info.get("g_cols", [])),
        }
        anchor_json.append(anchor_record)

        for modality_combo in modality_combos:
            x_mod, mod_info = _build_modality_feature_matrix(
                x_all=x_all,
                x_info=x_info,
                modality_combo=modality_combo,
                sample_index=bundle.meta.index,
            )
            x_mod = sanitize_feature_matrix(x_mod)
            avail = _availability_row(
                timeline=timeline_name,
                anchor_idx=int(anchor.anchor_idx),
                anchor_tb=float(anchor.anchor_tb),
                modality_combo=modality_combo,
                x=x_mod,
                n_h=int(mod_info["n_features_h"]),
                n_c=int(mod_info["n_features_c"]),
                n_g=int(mod_info["n_features_g"]),
            )
            feature_rows.append(
                {
                    **avail,
                    "anchor_phase": anchor.anchor_phase,
                    "anchor_tb_token": anchor.anchor_tb_token,
                    "hyperspectral_prefix_col_count": int(anchor.hyperspectral_prefix_col_count),
                    "climate_prefix_col_count": int(anchor.climate_prefix_col_count),
                    "n_features_total": int(mod_info["n_features_total"]),
                    "n_features_h": int(mod_info["n_features_h"]),
                    "n_features_c": int(mod_info["n_features_c"]),
                    "n_features_g": int(mod_info["n_features_g"]),
                }
            )

    return anchor_cache, feature_rows, anchor_json


def _build_metrics_summary_all(metrics_fold_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_fold_df.empty:
        return pd.DataFrame()
    rows = []
    for (scenario, target), sub in metrics_fold_df.groupby(["scenario", "target"], dropna=False):
        tmp = _ablation_build_metrics_summary(sub)
        if tmp.empty:
            continue
        tmp.insert(0, "target", str(target))
        tmp.insert(0, "scenario", str(scenario))
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["scenario", "target", "timeline", "anchor_idx", "modality_combo", "predictor"]
    ).reset_index(drop=True)


def _build_metrics_by_year_all(oof_df: pd.DataFrame) -> pd.DataFrame:
    if oof_df.empty:
        return pd.DataFrame()
    rows = []
    for (scenario, target), sub in oof_df.groupby(["scenario", "target"], dropna=False):
        tmp = _ablation_build_metrics_by_year(sub)
        if tmp.empty:
            continue
        tmp.insert(0, "target", str(target))
        tmp.insert(0, "scenario", str(scenario))
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["scenario", "target", "timeline", "year", "anchor_idx", "modality_combo", "predictor"]
    ).reset_index(drop=True)


def _build_rank_tables_all(metrics_summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if metrics_summary_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    pred_rows = []
    global_rows = []
    rank_rows = []
    for (scenario, target), sub in metrics_summary_df.groupby(["scenario", "target"], dropna=False):
        pred_table, global_table, rank_df = _ablation_build_rank_tables(sub)
        if not pred_table.empty:
            pred_table.insert(0, "target", str(target))
            pred_table.insert(0, "scenario", str(scenario))
            pred_rows.append(pred_table)
        if not global_table.empty:
            global_table.insert(0, "target", str(target))
            global_table.insert(0, "scenario", str(scenario))
            global_rows.append(global_table)
        if not rank_df.empty:
            rank_df.insert(0, "target", str(target))
            rank_df.insert(0, "scenario", str(scenario))
            rank_rows.append(rank_df)

    pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    global_df = pd.concat(global_rows, ignore_index=True) if global_rows else pd.DataFrame()
    rank_df = pd.concat(rank_rows, ignore_index=True) if rank_rows else pd.DataFrame()
    if not pred_df.empty:
        pred_df = pred_df.sort_values(["scenario", "target", "timeline", "modality_combo", "predictor"]).reset_index(drop=True)
    if not global_df.empty:
        global_df = global_df.sort_values(["scenario", "target", "timeline", "modality_combo"]).reset_index(drop=True)
    if not rank_df.empty:
        rank_df = rank_df.sort_values(["scenario", "target", "timeline", "modality_combo", "predictor"]).reset_index(drop=True)
    return pred_df, global_df, rank_df


def _build_overview_tables(pred_table_df: pd.DataFrame, global_table_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_timeline_df = pd.DataFrame()
    scenario_target_df = pd.DataFrame()
    scenario_target_predictor_df = pd.DataFrame()

    if not global_table_df.empty:
        scenario_timeline_df = (
            global_table_df.groupby(["scenario", "timeline"], as_index=False)
            .agg(
                global_avg_r2=("global_avg_r2", "mean"),
                global_avg_rmse=("global_avg_rmse", "mean"),
                global_avg_mae=("global_avg_mae", "mean"),
                global_avg_pearson=("global_avg_pearson", "mean"),
                global_avg_rank_r2=("global_avg_rank_r2", "mean"),
                global_avg_rank_pearson=("global_avg_rank_pearson", "mean"),
            )
            .sort_values(["scenario", "global_avg_r2"], ascending=[True, False])
            .reset_index(drop=True)
        )
        scenario_target_df = (
            global_table_df.groupby(["scenario", "target"], as_index=False)
            .agg(
                global_avg_r2=("global_avg_r2", "mean"),
                global_avg_rmse=("global_avg_rmse", "mean"),
                global_avg_mae=("global_avg_mae", "mean"),
                global_avg_pearson=("global_avg_pearson", "mean"),
            )
            .sort_values(["scenario", "target"])
            .reset_index(drop=True)
        )

    if not pred_table_df.empty:
        scenario_target_predictor_df = (
            pred_table_df.groupby(["scenario", "target", "predictor"], as_index=False)
            .agg(
                avg_r2_across_anchors=("avg_r2_across_anchors", "mean"),
                avg_rmse_across_anchors=("avg_rmse_across_anchors", "mean"),
                avg_mae_across_anchors=("avg_mae_across_anchors", "mean"),
                avg_pearson_across_anchors=("avg_pearson_across_anchors", "mean"),
                avg_rank_by_r2=("avg_rank_by_r2", "mean"),
                avg_rank_by_pearson=("avg_rank_by_pearson", "mean"),
            )
            .sort_values(["scenario", "target", "avg_r2_across_anchors"], ascending=[True, True, False])
            .reset_index(drop=True)
        )

    return scenario_timeline_df, scenario_target_df, scenario_target_predictor_df


def _generate_figures_all(
    *,
    metrics_summary_df: pd.DataFrame,
    metrics_by_year_df: pd.DataFrame,
    pred_table_df: pd.DataFrame,
    global_table_df: pd.DataFrame,
    fig_dir: Path,
    scenario_order: list[str],
    target_order: list[str],
    timeline_order: list[str],
    modality_order: list[str],
) -> None:
    ensure_dir(fig_dir)
    if metrics_summary_df.empty:
        return

    for scenario in _ordered_unique(metrics_summary_df["scenario"].astype(str).tolist(), preferred=scenario_order):
        for target in _ordered_unique(
            metrics_summary_df[metrics_summary_df["scenario"].astype(str) == scenario]["target"].astype(str).tolist(),
            preferred=target_order,
        ):
            sub_summary = metrics_summary_df[
                (metrics_summary_df["scenario"].astype(str) == scenario)
                & (metrics_summary_df["target"].astype(str) == target)
            ].copy()
            if sub_summary.empty:
                continue
            sub_year = metrics_by_year_df[
                (metrics_by_year_df["scenario"].astype(str) == scenario)
                & (metrics_by_year_df["target"].astype(str) == target)
            ].copy() if not metrics_by_year_df.empty else pd.DataFrame()
            sub_pred = pred_table_df[
                (pred_table_df["scenario"].astype(str) == scenario)
                & (pred_table_df["target"].astype(str) == target)
            ].copy() if not pred_table_df.empty else pd.DataFrame()
            sub_global = global_table_df[
                (global_table_df["scenario"].astype(str) == scenario)
                & (global_table_df["target"].astype(str) == target)
            ].copy() if not global_table_df.empty else pd.DataFrame()

            sub_dir = fig_dir / _safe_filename_token(scenario) / _safe_filename_token(target)
            ensure_dir(sub_dir)
            _plot_overall_metrics(sub_summary, sub_dir, modality_order=modality_order, timeline_order=timeline_order)
            _plot_year_metrics(sub_year, sub_dir, modality_order=modality_order, timeline_order=timeline_order)
            _plot_timeline_group_views(sub_summary, sub_year, sub_dir, modality_order=modality_order, timeline_order=timeline_order)
            _plot_modality_group_views(sub_summary, sub_year, sub_dir, modality_order=modality_order, timeline_order=timeline_order)
            _plot_global_metric_by_modality(
                sub_pred,
                value_col="avg_r2_across_anchors",
                y_label="global mean R2 across anchors",
                out_path=sub_dir / "figure_global_r2_by_modality_timeline.png",
                modality_order=modality_order,
                timeline_order=timeline_order,
            )
            _plot_global_metric_by_modality(
                sub_pred,
                value_col="avg_pearson_across_anchors",
                y_label="global mean Pearson across anchors",
                out_path=sub_dir / "figure_global_pearson_by_modality_timeline.png",
                modality_order=modality_order,
                timeline_order=timeline_order,
            )
            _ablation_plot_rank_summary(
                sub_global,
                out_path=sub_dir / "figure_rank_summary.png",
                modality_order=modality_order,
                timeline_order=timeline_order,
            )
            _ablation_plot_stability_summary(
                sub_pred,
                out_path=sub_dir / "figure_stability_summary.png",
                modality_order=modality_order,
                timeline_order=timeline_order,
            )


def _write_summary(
    *,
    output_dir: Path,
    cfg: dict,
    anchor_df: pd.DataFrame,
    feature_overview_df: pd.DataFrame,
    split_usage_df: pd.DataFrame,
    metrics_summary_df: pd.DataFrame,
    metrics_by_year_df: pd.DataFrame,
    pred_table_df: pd.DataFrame,
    global_table_df: pd.DataFrame,
    scenario_timeline_df: pd.DataFrame,
    scenario_target_df: pd.DataFrame,
    scenario_target_predictor_df: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Result3.2 Summary")
    lines.append("")
    lines.append("## 1. 实验定义")
    lines.append(f"- 时间轴: `{', '.join((_resolve_timeline_dirs(cfg.get('data', {}), cfg.get('experiment', {}))).keys())}`")
    lines.append("- 输入方式: `temporal prefix-expanded tabular modeling`")
    lines.append(f"- Anchor 数: `{cfg.get('experiment', {}).get('n_anchor_bins', 12)}`")
    lines.append(f"- 组学组合: `{', '.join(_resolve_modality_combos(cfg.get('experiment', {})))}`")
    lines.append(f"- 性状: `{', '.join(cfg.get('experiment', {}).get('targets', []))}`")
    lines.append(f"- 预测器: `{', '.join(cfg.get('experiment', {}).get('predictors_run', []))}`")
    lines.append(f"- 场景: `{', '.join((_resolve_scenarios_cfg(cfg.get('data', {}))).keys())}`")
    lines.append(f"- 基因型表示: `{cfg.get('experiment', {}).get('genotype_representation', 'grm_pca')}`")
    lines.append("")
    lines.append("## 2. Anchor 定义预览")
    lines.append(_markdown_table(anchor_df, max_rows=40))
    lines.append("")
    lines.append("## 3. 时间轴 x Anchor x 组学可用性预览")
    lines.append(_markdown_table(feature_overview_df, max_rows=40))
    lines.append("")
    lines.append("## 4. split 使用概览")
    lines.append(_markdown_table(split_usage_df, max_rows=40))
    lines.append("")
    lines.append("## 5. 场景 x 时间轴总体概览")
    lines.append(_markdown_table(scenario_timeline_df, max_rows=40))
    lines.append("")
    lines.append("## 6. 场景 x 性状总体概览")
    lines.append(_markdown_table(scenario_target_df, max_rows=40))
    lines.append("")
    lines.append("## 7. 场景 x 性状 x 模型总体概览")
    lines.append(_markdown_table(scenario_target_predictor_df, max_rows=40))
    lines.append("")
    lines.append("## 8. Anchor 汇总预览")
    lines.append(_markdown_table(metrics_summary_df, max_rows=40))
    lines.append("")
    lines.append("## 9. 年份聚合预览")
    lines.append(_markdown_table(metrics_by_year_df, max_rows=40))
    lines.append("")
    lines.append("## 10. 时间轴 x 组学 x 模型汇总预览")
    lines.append(_markdown_table(pred_table_df, max_rows=40))
    lines.append("")
    lines.append("## 11. 时间轴 x 组学全局汇总预览")
    lines.append(_markdown_table(global_table_df, max_rows=40))
    lines.append("")
    lines.append("## 12. 图形文件")
    lines.append("- 图形目录结构: `figures/{scenario}/{target}/...`")
    lines.append("- 每个 `scenario × target` 下均会生成 Result2.7 风格的总体图、分年图、按 timeline 分组图、按 modality 分组图、global 图、rank 图、stability 图。")
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def finalize_result3_2_outputs(
    *,
    output_dir: Path,
    cfg: dict,
    anchor_bins_obj: list[dict] | None,
    feature_overview_df: pd.DataFrame,
    split_usage_df: pd.DataFrame,
    metrics_fold_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    inner_pred_df: pd.DataFrame,
    trial_df: pd.DataFrame,
) -> None:
    anchor_df = pd.DataFrame(anchor_bins_obj or [])
    if not anchor_df.empty:
        anchor_df = anchor_df.sort_values(["timeline", "anchor_idx"]).reset_index(drop=True)
    if not feature_overview_df.empty:
        feature_overview_df = feature_overview_df.sort_values(["timeline", "anchor_idx", "modality_combo"]).reset_index(drop=True)
    if not split_usage_df.empty:
        split_usage_df = split_usage_df.sort_values(["scenario", "timeline", "target"]).reset_index(drop=True)
    if not metrics_fold_df.empty:
        metrics_fold_df = metrics_fold_df.sort_values(
            ["scenario", "target", "timeline", "anchor_idx", "modality_combo", "predictor", "target_year", "outer_fold"]
        ).reset_index(drop=True)
    if not oof_df.empty:
        oof_df = oof_df.sort_values(
            ["scenario", "target", "timeline", "anchor_idx", "modality_combo", "predictor", "year", "plot_id"]
        ).reset_index(drop=True)
    if not inner_pred_df.empty:
        inner_pred_df = inner_pred_df.sort_values(
            [
                "scenario",
                "target",
                "timeline",
                "anchor_idx",
                "modality_combo",
                "predictor",
                "target_year",
                "outer_fold",
                "inner_fold",
                "plot_id",
            ]
        ).reset_index(drop=True)
    if not trial_df.empty:
        trial_df = trial_df.sort_values(
            [
                "scenario",
                "target",
                "timeline",
                "anchor_idx",
                "modality_combo",
                "predictor",
                "target_year",
                "outer_fold",
                "rank",
            ]
        ).reset_index(drop=True)

    metrics_summary_df = _build_metrics_summary_all(metrics_fold_df)
    metrics_by_year_df = _build_metrics_by_year_all(oof_df)
    pred_table_df, global_table_df, rank_df = _build_rank_tables_all(metrics_summary_df)
    scenario_timeline_df, scenario_target_df, scenario_target_predictor_df = _build_overview_tables(pred_table_df, global_table_df)

    if anchor_bins_obj is not None:
        write_json(anchor_bins_obj, output_dir / "anchor_bins.json")
    anchor_df.to_csv(output_dir / "anchor_feature_overview.csv", index=False)
    feature_overview_df.to_csv(output_dir / "feature_overview.csv", index=False)
    feature_overview_df.to_csv(output_dir / "timeline_anchor_modality_availability.csv", index=False)
    split_usage_df.to_csv(output_dir / "split_usage_summary.csv", index=False)
    metrics_fold_df.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    metrics_summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    metrics_by_year_df.to_csv(output_dir / "metrics_by_year.csv", index=False)
    pred_table_df.to_csv(output_dir / "metrics_by_scenario_target_timeline_modality_predictor.csv", index=False)
    global_table_df.to_csv(output_dir / "metrics_by_scenario_target_timeline_modality_global.csv", index=False)
    rank_df.to_csv(output_dir / "rank_details.csv", index=False)
    scenario_timeline_df.to_csv(output_dir / "metrics_by_scenario_timeline.csv", index=False)
    scenario_target_df.to_csv(output_dir / "metrics_by_scenario_target.csv", index=False)
    scenario_target_predictor_df.to_csv(output_dir / "metrics_by_scenario_target_predictor.csv", index=False)
    if not oof_df.empty:
        oof_df.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    if not inner_pred_df.empty:
        inner_pred_df.to_parquet(output_dir / "outer_test_inner_ensemble_predictions.parquet", index=False)
    trial_df.to_csv(output_dir / "optuna_trials.csv", index=False)

    timeline_order = list(_resolve_timeline_dirs(cfg.get("data", {}), cfg.get("experiment", {})).keys())
    scenario_order = list(_resolve_scenarios_cfg(cfg.get("data", {})).keys())
    target_order = [str(x) for x in cfg.get("experiment", {}).get("targets", [])]
    modality_order = _resolve_modality_combos(cfg.get("experiment", {}))
    fig_dir = output_dir / "figures"
    _generate_figures_all(
        metrics_summary_df=metrics_summary_df,
        metrics_by_year_df=metrics_by_year_df,
        pred_table_df=pred_table_df,
        global_table_df=global_table_df,
        fig_dir=fig_dir,
        scenario_order=scenario_order,
        target_order=target_order,
        timeline_order=timeline_order,
        modality_order=modality_order,
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
        anchor_df=anchor_df,
        feature_overview_df=feature_overview_df,
        split_usage_df=split_usage_df,
        metrics_summary_df=metrics_summary_df,
        metrics_by_year_df=metrics_by_year_df,
        pred_table_df=pred_table_df,
        global_table_df=global_table_df,
        scenario_timeline_df=scenario_timeline_df,
        scenario_target_df=scenario_target_df,
        scenario_target_predictor_df=scenario_target_predictor_df,
    )


def run_result3_2(config_path: Path) -> Path:
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
    timeline_dirs = _resolve_timeline_dirs(data_cfg, exp_cfg)
    scenarios_cfg = _resolve_scenarios_cfg(data_cfg)
    model_cfg = exp_cfg.get("model_backends", {})
    active_predictors = _resolve_active_predictors(exp_cfg)
    modality_combos = _resolve_modality_combos(exp_cfg)
    task_filter_set = _resolve_task_filter_set(exp_cfg)
    planned_tasks = enumerate_result32_tasks_from_config(cfg)
    allowed_task_keys = {
        (task["scenario"], task["timeline"], task["target"], task["modality_combo"], task["predictor"])
        for task in planned_tasks
    }
    if not planned_tasks:
        raise RuntimeError("没有匹配到可执行的 Result3.2 任务。")

    n_anchor_bins = int(exp_cfg.get("n_anchor_bins", 12))
    n_trials = int(optuna_cfg.get("n_trials", 3))
    lambda_gap = float(optuna_cfg.get("lambda_gap", 1.0))
    pruner_startup_trials = int(optuna_cfg.get("pruner_startup_trials", 1))

    timeline_bundles: Dict[str, MultiTargetDataBundle] = {}
    timeline_anchors: Dict[str, list[AnchorDefinition]] = {}
    timeline_anchor_cache: Dict[str, dict[int, dict[str, Any]]] = {}
    feature_overview_rows: list[dict[str, Any]] = []
    anchor_json: list[dict[str, Any]] = []
    timeline_stats: dict[str, Any] = {}

    for timeline_name, timeline_path in timeline_dirs.items():
        print(f"[LOAD] timeline={timeline_name} path={timeline_path}", flush=True)
        bundle = load_multitarget_model_inputs(Path(timeline_path), genotype_representation=genotype_representation)
        anchors = build_anchor_definitions(bundle, n_anchor_bins=n_anchor_bins)
        anchor_cache, feature_rows, anchor_rows = _build_anchor_cache(timeline_name, bundle, anchors, modality_combos)
        timeline_bundles[timeline_name] = bundle
        timeline_anchors[timeline_name] = anchors
        timeline_anchor_cache[timeline_name] = anchor_cache
        feature_overview_rows.extend(feature_rows)
        anchor_json.extend(anchor_rows)
        timeline_stats[timeline_name] = {
            "input_dir": str(timeline_path),
            "n_samples": int(bundle.meta.shape[0]),
            "n_common_tbs": int(len(bundle.common_tbs)),
            "n_anchors": int(len(anchors)),
        }

    split_usage_rows = []
    prepared_split_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    for scenario_name, scenario_cfg in scenarios_cfg.items():
        for timeline_name, bundle in timeline_bundles.items():
            raw_groups = _resolve_split_groups_for_scenario(
                scenario_name=scenario_name,
                scenario_cfg=scenario_cfg,
                meta_df=bundle.meta,
                random_seed=seed,
            )
            for target_col in targets:
                if target_col not in bundle.y_df.columns:
                    info = {
                        "scenario": scenario_name,
                        "timeline": timeline_name,
                        "target": target_col,
                        "status": "missing_target_column",
                    }
                    split_usage_rows.append(info)
                    prepared_split_cache[(scenario_name, timeline_name, target_col)] = info
                    continue

                y_full = bundle.y_df[target_col].dropna().copy()
                available_ids = set(bundle.meta.index.intersection(y_full.index))
                prepared_groups, split_stats = _prepare_split_groups(raw_groups, available_ids)
                info = {
                    "scenario": scenario_name,
                    "timeline": timeline_name,
                    "target": target_col,
                    "status": "ok" if prepared_groups else "no_valid_split_group",
                    **split_stats,
                    "n_target_samples": int(len(available_ids)),
                    "prepared_groups": prepared_groups,
                    "y_full": y_full,
                }
                split_usage_rows.append({k: v for k, v in info.items() if k not in {"prepared_groups", "y_full"}})
                prepared_split_cache[(scenario_name, timeline_name, target_col)] = info
                timeline_stats[timeline_name].setdefault("split_filter_stats", {})[f"{scenario_name}:{target_col}"] = {
                    **split_stats,
                    "n_target_samples": int(len(available_ids)),
                }

    total_tasks = 0
    for task in planned_tasks:
        split_info = prepared_split_cache.get((task["scenario"], task["timeline"], task["target"]), {})
        if split_info.get("status") != "ok":
            continue
        total_tasks += len(timeline_anchors[task["timeline"]])
    if total_tasks <= 0:
        raise RuntimeError("没有可执行的有效任务，请检查 split / target / timeline 配置。")

    metrics_fold_rows = []
    oof_rows = []
    inner_pred_rows = []
    trial_rows = []
    task_idx = 0

    task_filters_cfg = exp_cfg.get("task_filters", {}) if isinstance(exp_cfg.get("task_filters"), dict) else {}
    scenario_order = list(scenarios_cfg.keys())
    timeline_order = list(timeline_dirs.keys())
    combo_order = list(modality_combos)
    scenario_seed_order = [str(x) for x in task_filters_cfg.get("scenario_order", scenario_order)]
    timeline_seed_order = [str(x) for x in task_filters_cfg.get("timeline_order", timeline_order)]
    target_seed_order = [str(x) for x in task_filters_cfg.get("target_order", targets)]
    combo_seed_order = [str(x) for x in task_filters_cfg.get("modality_combo_order", combo_order)]
    predictor_seed_order = [str(x).lower() for x in task_filters_cfg.get("predictor_order", active_predictors)]

    for timeline_name in timeline_order:
        bundle = timeline_bundles[timeline_name]
        meta_full = bundle.meta
        anchors = timeline_anchors[timeline_name]
        for anchor in anchors:
            cache_row = timeline_anchor_cache[timeline_name][int(anchor.anchor_idx)]
            x_all = cache_row["x_all"]
            x_info = cache_row["x_info"]

            for modality_combo in combo_order:
                x_mod, mod_info = _build_modality_feature_matrix(
                    x_all=x_all,
                    x_info=x_info,
                    modality_combo=modality_combo,
                    sample_index=bundle.meta.index,
                )
                x_mod = sanitize_feature_matrix(x_mod)

                for scenario_name in scenario_order:
                    for target_col in targets:
                        task_base_key = (scenario_name, timeline_name, target_col)
                        split_info = prepared_split_cache.get(task_base_key, {})
                        if split_info.get("status") != "ok":
                            continue
                        y_full = split_info["y_full"]
                        prepared_groups = split_info["prepared_groups"]

                        for predictor in active_predictors:
                            task_key = (scenario_name, timeline_name, target_col, modality_combo, predictor)
                            if task_filter_set and task_key not in allowed_task_keys:
                                continue

                            task_idx += 1
                            print(
                                f"[TASK {task_idx}/{total_tasks}] scenario={scenario_name} tl={timeline_name} target={target_col} combo={modality_combo} predictor={predictor} anchor={anchor.anchor_tb} features={x_mod.shape[1]}",
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
                                    "modality_combo": modality_combo,
                                    "predictor": predictor,
                                    "anchor_idx": int(anchor.anchor_idx),
                                    "anchor_tb": float(anchor.anchor_tb),
                                    "n_features": int(x_mod.shape[1]),
                                },
                            )

                            fold_rows_for_combo = []
                            ordered_outer_items = normalize_outer_groups(prepared_groups)
                            ordered_outer_keys = [key for key, _ in ordered_outer_items]
                            reuse_policy = resolve_optuna_reuse_policy(optuna_cfg)
                            cached_best_params = None
                            cached_source = None

                            for (target_year, outer_fold), group_splits in ordered_outer_items:
                                payload = _build_outer_payload(x_mod, y_full, meta_full, group_splits)
                                inner_payload = payload["inner_payload"]
                                x_test = payload["x_test"]
                                y_test = payload["y_test"]
                                meta_test = payload["meta_test"]
                                if len(x_test) == 0 or not inner_payload:
                                    continue

                                s_idx = scenario_seed_order.index(scenario_name) if scenario_name in scenario_seed_order else 0
                                t_idx = timeline_seed_order.index(timeline_name) if timeline_name in timeline_seed_order else 0
                                g_idx = target_seed_order.index(target_col) if target_col in target_seed_order else 0
                                c_idx = combo_seed_order.index(modality_combo) if modality_combo in combo_seed_order else 0
                                p_idx = predictor_seed_order.index(predictor) if predictor in predictor_seed_order else 0
                                sampler_seed = (
                                    seed
                                    + s_idx * 1_000_000
                                    + t_idx * 100_000
                                    + g_idx * 10_000
                                    + c_idx * 1_000
                                    + p_idx * 100
                                    + int(anchor.anchor_idx) * 10
                                    + int(target_year % 100)
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

                                    def objective(trial: optuna.Trial) -> float:
                                        params = suggest_params(trial, predictor, model_cfg=model_cfg)
                                        train_losses = []
                                        val_losses = []
                                        train_metrics_list = []
                                        val_metrics_list = []

                                        for fold_i, inner in enumerate(inner_payload):
                                            pipe = build_pipeline(
                                                predictor=predictor,
                                                params=params,
                                                n_features=inner["x_train"].shape[1],
                                                preprocess_cfg=preprocess_cfg,
                                                seed=int(sampler_seed) + trial.number + fold_i,
                                                model_cfg=model_cfg,
                                            )
                                            pipe.fit(inner["x_train"], inner["y_train"])
                                            pred_train = pipe.predict(inner["x_train"])
                                            pred_val = pipe.predict(inner["x_val"])

                                            train_metrics = regression_metrics(inner["y_train"].to_numpy(), pred_train)
                                            val_metrics = regression_metrics(inner["y_val"].to_numpy(), pred_val)
                                            train_losses.append(float(train_metrics["rmse"]))
                                            val_losses.append(float(val_metrics["rmse"]))
                                            train_metrics_list.append(train_metrics)
                                            val_metrics_list.append(val_metrics)

                                            partial = float(np.mean(train_losses) + np.mean(val_losses))
                                            trial.report(partial, step=fold_i)
                                            if trial.should_prune():
                                                raise optuna.exceptions.TrialPruned()

                                        mean_train = float(np.mean(train_losses))
                                        mean_val = float(np.mean(val_losses))
                                        gap = float(max(0.0, mean_val - mean_train))
                                        objective_value = float(mean_train + mean_val + lambda_gap * gap)
                                        mean_train_metrics = mean_regression_metrics(train_metrics_list)
                                        mean_val_metrics = mean_regression_metrics(val_metrics_list)

                                        trial.set_user_attr("mean_train_loss", mean_train)
                                        trial.set_user_attr("mean_val_loss", mean_val)
                                        trial.set_user_attr("gap", gap)
                                        trial.set_user_attr("objective", objective_value)
                                        trial.set_user_attr("mean_train_r2", mean_train_metrics["r2"])
                                        trial.set_user_attr("mean_train_rmse", mean_train_metrics["rmse"])
                                        trial.set_user_attr("mean_train_mae", mean_train_metrics["mae"])
                                        trial.set_user_attr("mean_train_pearson", mean_train_metrics["pearson_r"])
                                        trial.set_user_attr("mean_val_r2", mean_val_metrics["r2"])
                                        trial.set_user_attr("mean_val_rmse", mean_val_metrics["rmse"])
                                        trial.set_user_attr("mean_val_mae", mean_val_metrics["mae"])
                                        trial.set_user_attr("mean_val_pearson", mean_val_metrics["pearson_r"])
                                        return objective_value

                                    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
                                    complete_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
                                    if not complete_trials:
                                        continue

                                    sorted_trials = sorted(complete_trials, key=lambda trial: trial.value)
                                    rank_map = {trial.number: rank + 1 for rank, trial in enumerate(sorted_trials)}
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
                                    for trial in sorted_trials:
                                        trial_rows.append(
                                            {
                                                "scenario": scenario_name,
                                                "timeline": timeline_name,
                                                "target": target_col,
                                                "anchor_idx": int(anchor.anchor_idx),
                                                "anchor_tb": float(anchor.anchor_tb),
                                                "modality_combo": modality_combo,
                                                "predictor": predictor,
                                                "target_year": int(target_year),
                                                "outer_fold": int(outer_fold),
                                                "trial_number": int(trial.number),
                                                "params_json": json.dumps(trial.params, ensure_ascii=False, sort_keys=True),
                                                "mean_train_loss": trial.user_attrs.get("mean_train_loss"),
                                                "mean_val_loss": trial.user_attrs.get("mean_val_loss"),
                                                "mean_train_r2": trial.user_attrs.get("mean_train_r2"),
                                                "mean_train_rmse": trial.user_attrs.get("mean_train_rmse"),
                                                "mean_train_mae": trial.user_attrs.get("mean_train_mae"),
                                                "mean_train_pearson": trial.user_attrs.get("mean_train_pearson"),
                                                "mean_val_r2": trial.user_attrs.get("mean_val_r2"),
                                                "mean_val_rmse": trial.user_attrs.get("mean_val_rmse"),
                                                "mean_val_mae": trial.user_attrs.get("mean_val_mae"),
                                                "mean_val_pearson": trial.user_attrs.get("mean_val_pearson"),
                                                "gap": trial.user_attrs.get("gap"),
                                                "objective": trial.value,
                                                "rank": rank_map[trial.number],
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
                                inner_model_preds = []
                                for inner in inner_payload:
                                    pipe = build_pipeline(
                                        predictor=predictor,
                                        params=best_params,
                                        n_features=inner["x_train"].shape[1],
                                        preprocess_cfg=preprocess_cfg,
                                        seed=int(sampler_seed) + int(inner["inner_fold"]) + 991,
                                        model_cfg=model_cfg,
                                    )
                                    pipe.fit(inner["x_train"], inner["y_train"])
                                    inner_model_preds.append(pipe.predict(x_test))

                                pred_mat = np.column_stack(inner_model_preds)
                                ensemble_pred = pred_mat.mean(axis=1)
                                y_true = y_test.to_numpy()
                                fold_metrics = regression_metrics(y_true, ensemble_pred)

                                for i, plot_id in enumerate(x_test.index.tolist()):
                                    y_true_i = float(y_true[i])
                                    y_pred_i = float(ensemble_pred[i])
                                    for j, inner in enumerate(inner_payload):
                                        inner_pred_rows.append(
                                            {
                                                "plot_id": plot_id,
                                                "year": int(meta_test.loc[plot_id, "year"]),
                                                "scenario": scenario_name,
                                                "timeline": timeline_name,
                                                "target": target_col,
                                                "target_year": int(target_year),
                                                "outer_fold": int(outer_fold),
                                                "inner_fold": int(inner["inner_fold"]),
                                                "anchor_idx": int(anchor.anchor_idx),
                                                "anchor_tb": float(anchor.anchor_tb),
                                                "modality_combo": modality_combo,
                                                "predictor": predictor,
                                                "y_true": y_true_i,
                                                "y_pred_inner_model": float(pred_mat[i, j]),
                                                "y_pred_ensemble_mean": y_pred_i,
                                            }
                                        )
                                    oof_rows.append(
                                        {
                                            "plot_id": plot_id,
                                            "year": int(meta_test.loc[plot_id, "year"]),
                                            "scenario": scenario_name,
                                            "timeline": timeline_name,
                                            "target": target_col,
                                            "target_year": int(target_year),
                                            "outer_fold": int(outer_fold),
                                            "anchor_idx": int(anchor.anchor_idx),
                                            "anchor_tb": float(anchor.anchor_tb),
                                            "modality_combo": modality_combo,
                                            "predictor": predictor,
                                            "y_true": y_true_i,
                                            "y_pred": y_pred_i,
                                        }
                                    )

                                n_train = int(np.mean([len(inner["train_ids"]) for inner in inner_payload]))
                                n_val = int(np.mean([len(inner["val_ids"]) for inner in inner_payload]))
                                n_test = int(len(payload["test_ids"]))
                                fold_row = {
                                    "scenario": scenario_name,
                                    "timeline": timeline_name,
                                    "target": target_col,
                                    "anchor_idx": int(anchor.anchor_idx),
                                    "anchor_tb": float(anchor.anchor_tb),
                                    "modality_combo": modality_combo,
                                    "predictor": predictor,
                                    "target_year": int(target_year),
                                    "outer_fold": int(outer_fold),
                                    "n_inner_folds": int(len(inner_payload)),
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
                                    "n_train": n_train,
                                    "n_val": n_val,
                                    "n_test": n_test,
                                    "n_features_total": int(mod_info["n_features_total"]),
                                    "n_features_h": int(mod_info["n_features_h"]),
                                    "n_features_c": int(mod_info["n_features_c"]),
                                    "n_features_g": int(mod_info["n_features_g"]),
                                    "n_optuna_trials": int(cached_source["n_trials"]) if optuna_search_performed and cached_source else 0,
                                    "optuna_search_performed": bool(optuna_search_performed),
                                    "optuna_reuse_scope": reuse_policy["mode"],
                                    "optuna_source_target_year": source_target_year,
                                    "optuna_source_outer_fold": source_outer_fold,
                                    "optuna_source_trial_number": source_trial_number,
                                    "optuna_source_objective": source_objective,
                                }
                                metrics_fold_rows.append(fold_row)
                                fold_rows_for_combo.append(fold_row)

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
                                    "modality_combo": modality_combo,
                                    "predictor": predictor,
                                    "anchor_idx": int(anchor.anchor_idx),
                                    "anchor_tb": float(anchor.anchor_tb),
                                    "mean_r2": float(pd.DataFrame(fold_rows_for_combo)["test_r2"].mean()) if fold_rows_for_combo else None,
                                },
                            )

    feature_overview_df = pd.DataFrame(feature_overview_rows)
    split_usage_df = pd.DataFrame(split_usage_rows)
    metrics_fold_df = pd.DataFrame(metrics_fold_rows)
    oof_df = pd.DataFrame(oof_rows)
    inner_pred_df = pd.DataFrame(inner_pred_rows)
    trial_df = pd.DataFrame(trial_rows)

    run_snapshot = {
        **cfg,
        "config_path": str(config_path),
        "runtime": {
            "timestamp": now_iso(),
            "git_commit_hash": get_git_commit_hash(Path.cwd()),
            "resolved": {
                "objective_definition": "mean_train_loss + mean_val_loss + lambda_gap * max(0, mean_val_loss - mean_train_loss)",
                "input_type": "temporal",
                "timeline_dirs": timeline_dirs,
                "modality_combos": modality_combos,
                "n_anchor_bins_requested": int(n_anchor_bins),
                "n_anchor_bins_effective": {k: int(len(v)) for k, v in timeline_anchors.items()},
                "n_samples_by_timeline": {k: int(v.meta.shape[0]) for k, v in timeline_bundles.items()},
            },
        },
        "data": {
            **data_cfg,
            "timeline_dirs": timeline_dirs,
            "scenarios": scenarios_cfg,
        },
        "experiment": {
            **exp_cfg,
            "targets": targets,
            "predictors_run": active_predictors,
            "active_predictors": active_predictors,
            "modality_combos": modality_combos,
        },
    }

    finalize_result3_2_outputs(
        output_dir=output_dir,
        cfg=run_snapshot,
        anchor_bins_obj=anchor_json,
        feature_overview_df=feature_overview_df,
        split_usage_df=split_usage_df,
        metrics_fold_df=metrics_fold_df,
        oof_df=oof_df,
        inner_pred_df=inner_pred_df,
        trial_df=trial_df,
    )

    _append_progress(output_dir, enabled=progress_enabled, event="run_end", payload={"output_dir": output_dir.as_posix()})
    print(f"Result3.2 completed: {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Result3.2 multitrait multi-timeline multimodal ablation experiments.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run_result3_2(Path(args.config))


if __name__ == "__main__":
    main()
