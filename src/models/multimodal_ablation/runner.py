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

from models.anchorwise.data_loader import load_day_model_inputs
from models.anchorwise.feature_builder import build_anchor_definitions, build_feature_matrix
from models.anchorwise.modeling import HAS_LIGHTGBM, build_pipeline, sanitize_feature_matrix, suggest_params
from models.common.io_utils import ensure_dir, get_git_commit_hash, now_iso, read_yaml, write_json, write_yaml
from models.common.metrics import mean_regression_metrics, regression_metrics, rmse
from models.common.split_loader import WithinSeasonSplit, load_split_groups

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Skipping features without any observed values")
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt", category=RuntimeWarning)

DEFAULT_MODALITY_COMBOS = ["H", "C", "G", "H+C", "H+G", "C+G", "H+C+G"]


def _normalize_validation_scenario(validation_scenario: str) -> str:
    return str(validation_scenario).strip().lower().replace("-", "_")


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _resolve_output_dir(output_cfg: dict) -> Path:
    base = Path(
        output_cfg.get(
            "output_dir_base",
            output_cfg.get("output_dir", "outputs/experiments/within_season_temporal_gdd_vs_relheading_multimodal_ablation"),
        )
    )
    append_timestamp = bool(output_cfg.get("append_timestamp", False))
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


def _filter_ids(ids: List[str], available_set: set) -> List[str]:
    return [x for x in ids if x in available_set]


def _append_progress(
    out_dir: Path,
    *,
    enabled: bool,
    event: str,
    payload: dict,
) -> None:
    if not enabled:
        return
    record = {"ts": now_iso(), "event": event, **payload}
    path = out_dir / "progress.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _prepare_split_groups(
    split_groups: Dict[Tuple[int, int], List[WithinSeasonSplit]],
    available_ids: set,
) -> Tuple[Dict[Tuple[int, int], List[dict]], dict]:
    prepared: Dict[Tuple[int, int], List[dict]] = {}
    dropped_stats = {
        "n_groups_total": len(split_groups),
        "n_groups_used": 0,
        "n_ids_dropped_train": 0,
        "n_ids_dropped_val": 0,
        "n_ids_dropped_test": 0,
    }

    for key, splits in split_groups.items():
        rows = []
        for sp in splits:
            tr = _filter_ids(sp.train_ids, available_set=available_ids)
            va = _filter_ids(sp.val_ids, available_set=available_ids)
            te = _filter_ids(sp.test_ids, available_set=available_ids)

            dropped_stats["n_ids_dropped_train"] += max(0, len(sp.train_ids) - len(tr))
            dropped_stats["n_ids_dropped_val"] += max(0, len(sp.val_ids) - len(va))
            dropped_stats["n_ids_dropped_test"] += max(0, len(sp.test_ids) - len(te))

            if not tr or not va or not te:
                continue

            rows.append(
                {
                    "split_name": sp.split_name,
                    "target_year": sp.target_year,
                    "outer_fold": sp.outer_fold,
                    "inner_fold": sp.inner_fold,
                    "train_ids": tr,
                    "val_ids": va,
                    "test_ids": te,
                }
            )

        if rows:
            prepared[key] = sorted(rows, key=lambda x: x["inner_fold"])

    dropped_stats["n_groups_used"] = len(prepared)
    return prepared, dropped_stats


def _build_outer_payload(x: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, group_splits: List[dict]) -> dict:
    inner_payload = []
    for row in group_splits:
        tr_ids = row["train_ids"]
        va_ids = row["val_ids"]
        inner_payload.append(
            {
                "inner_fold": row["inner_fold"],
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
    payload = {
        "target_year": group_splits[0]["target_year"],
        "outer_fold": group_splits[0]["outer_fold"],
        "test_ids": test_ids,
        "x_test": x.loc[test_ids],
        "y_test": y.loc[test_ids],
        "meta_test": meta.loc[test_ids],
        "inner_payload": inner_payload,
    }
    return payload


def _normalize_modality_combo(combo: str) -> str:
    parts = [p.strip().upper() for p in str(combo).split("+") if p.strip()]
    valid = [p for p in parts if p in {"H", "C", "G"}]
    if not valid:
        raise ValueError(f"非法 modality_combo: {combo}")
    order = ["H", "C", "G"]
    out = [x for x in order if x in valid]
    return "+".join(out)


def _parse_modality_combos(raw_combos: List[str] | None) -> List[str]:
    combos = raw_combos or DEFAULT_MODALITY_COMBOS
    norm = [_normalize_modality_combo(c) for c in combos]
    uniq = []
    seen = set()
    for c in norm:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _build_modality_feature_matrix(
    x_all: pd.DataFrame,
    x_info: dict,
    modality_combo: str,
    sample_index: pd.Index,
) -> tuple[pd.DataFrame, dict]:
    tokens = set(modality_combo.split("+"))

    h_cols = list(x_info.get("h_cols", [])) if "H" in tokens else []
    c_cols = list(x_info.get("c_cols", [])) if "C" in tokens else []
    g_cols = list(x_info.get("g_cols", [])) if "G" in tokens else []

    selected_cols = h_cols + c_cols + g_cols
    if not selected_cols:
        raise ValueError(f"modality_combo={modality_combo} 没有可用列。")

    x = x_all[selected_cols].copy()
    x = x.loc[sample_index]

    info = {
        "modality_combo": modality_combo,
        "n_features_total": int(x.shape[1]),
        "n_features_h": int(len(h_cols)),
        "n_features_c": int(len(c_cols)),
        "n_features_g": int(len(g_cols)),
        "h_cols": h_cols,
        "c_cols": c_cols,
        "g_cols": g_cols,
    }
    return x, info


def _availability_row(
    timeline: str,
    anchor_idx: int,
    anchor_tb: int,
    modality_combo: str,
    x: pd.DataFrame,
    n_h: int,
    n_c: int,
    n_g: int,
) -> dict:
    miss = x.isna()
    row_miss = miss.mean(axis=1)
    overall_missing_ratio = float(miss.values.mean()) if x.size > 0 else float("nan")
    fully = int((row_miss == 0).sum())
    partially = int(((row_miss > 0) & (row_miss <= 0.5)).sum())
    severe = int((row_miss > 0.5).sum())

    return {
        "timeline": timeline,
        "anchor_idx": int(anchor_idx),
        "anchor_tb": int(anchor_tb),
        "modality_combo": modality_combo,
        "n_samples": int(x.shape[0]),
        "n_h_features": int(n_h),
        "n_c_features": int(n_c),
        "n_g_features": int(n_g),
        "total_feature_dim": int(x.shape[1]),
        "overall_missing_ratio": overall_missing_ratio,
        "row_missing_ratio_mean": float(row_miss.mean()),
        "row_missing_ratio_std": float(row_miss.std(ddof=0)),
        "fully_observed_sample_count": fully,
        "partially_missing_sample_count": partially,
        "severely_missing_sample_count": severe,
    }


def _safe_std(s: pd.Series) -> float:
    return float(np.std(s, ddof=0))


def _build_metrics_summary(metrics_fold: pd.DataFrame) -> pd.DataFrame:
    if metrics_fold.empty:
        return pd.DataFrame()

    out = (
        metrics_fold.groupby(["timeline", "anchor_idx", "anchor_tb", "modality_combo", "predictor"], as_index=False)
        .agg(
            mean_r2=("test_r2", "mean"),
            std_r2=("test_r2", _safe_std),
            mean_rmse=("test_rmse", "mean"),
            std_rmse=("test_rmse", _safe_std),
            mean_mae=("test_mae", "mean"),
            std_mae=("test_mae", _safe_std),
            mean_pearson=("test_pearson", "mean"),
            std_pearson=("test_pearson", _safe_std),
            n_outer_folds=("outer_fold", "nunique"),
            n_inner_folds=("n_inner_folds", "mean"),
            n_optuna_trials=("n_optuna_trials", "mean"),
            n_features_total=("n_features_total", "mean"),
            n_features_h=("n_features_h", "mean"),
            n_features_c=("n_features_c", "mean"),
            n_features_g=("n_features_g", "mean"),
        )
        .sort_values(["timeline", "anchor_idx", "modality_combo", "predictor"])
    )
    for col in ["n_inner_folds", "n_optuna_trials", "n_features_total", "n_features_h", "n_features_c", "n_features_g"]:
        out[col] = out[col].astype(int)
    return out


def _build_metrics_by_year(oof_df: pd.DataFrame) -> pd.DataFrame:
    if oof_df.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["timeline", "anchor_idx", "anchor_tb", "modality_combo", "predictor", "year"]
    for keys, sub in oof_df.groupby(group_cols):
        y_true = sub["y_true"].to_numpy(dtype=float)
        y_pred = sub["y_pred"].to_numpy(dtype=float)
        m = regression_metrics(y_true, y_pred)
        rows.append(
            {
                "timeline": keys[0],
                "anchor_idx": int(keys[1]),
                "anchor_tb": int(keys[2]),
                "modality_combo": keys[3],
                "predictor": keys[4],
                "year": int(keys[5]),
                "r2": float(m["r2"]),
                "rmse": float(m["rmse"]),
                "mae": float(m["mae"]),
                "pearson": float(m["pearson_r"]),
                "n_samples": int(len(sub)),
            }
        )

    return pd.DataFrame(rows).sort_values(group_cols)


def _build_rank_tables(metrics_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if metrics_summary.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rank_rows = []
    for (timeline, predictor, anchor_idx), sub in metrics_summary.groupby(["timeline", "predictor", "anchor_idx"]):
        part = sub.copy()
        part["rank_r2"] = part["mean_r2"].rank(method="average", ascending=False)
        part["rank_rmse"] = part["mean_rmse"].rank(method="average", ascending=True)
        part["rank_mae"] = part["mean_mae"].rank(method="average", ascending=True)
        part["rank_pearson"] = part["mean_pearson"].rank(method="average", ascending=False)

        for _, r in part.iterrows():
            rank_rows.append(
                {
                    "timeline": timeline,
                    "predictor": predictor,
                    "anchor_idx": int(anchor_idx),
                    "modality_combo": r["modality_combo"],
                    "rank_r2": float(r["rank_r2"]),
                    "rank_rmse": float(r["rank_rmse"]),
                    "rank_mae": float(r["rank_mae"]),
                    "rank_pearson": float(r["rank_pearson"]),
                }
            )

    rank_df = pd.DataFrame(rank_rows)

    pred_table = (
        metrics_summary.groupby(["timeline", "modality_combo", "predictor"], as_index=False)
        .agg(
            avg_r2_across_anchors=("mean_r2", "mean"),
            avg_rmse_across_anchors=("mean_rmse", "mean"),
            avg_mae_across_anchors=("mean_mae", "mean"),
            avg_pearson_across_anchors=("mean_pearson", "mean"),
            avg_std_r2_across_anchors=("std_r2", "mean"),
            avg_std_pearson_across_anchors=("std_pearson", "mean"),
        )
        .sort_values(["timeline", "modality_combo", "predictor"])
    )

    rank_aggr = (
        rank_df.groupby(["timeline", "modality_combo", "predictor"], as_index=False)
        .agg(
            avg_rank_by_r2=("rank_r2", "mean"),
            avg_rank_by_rmse=("rank_rmse", "mean"),
            avg_rank_by_mae=("rank_mae", "mean"),
            avg_rank_by_pearson=("rank_pearson", "mean"),
        )
        .sort_values(["timeline", "modality_combo", "predictor"])
    )
    pred_table = pred_table.merge(rank_aggr, on=["timeline", "modality_combo", "predictor"], how="left")

    global_table = (
        metrics_summary.groupby(["timeline", "modality_combo"], as_index=False)
        .agg(
            global_avg_r2=("mean_r2", "mean"),
            global_avg_rmse=("mean_rmse", "mean"),
            global_avg_mae=("mean_mae", "mean"),
            global_avg_pearson=("mean_pearson", "mean"),
        )
        .sort_values(["timeline", "modality_combo"])
    )

    global_rank = (
        rank_df.groupby(["timeline", "modality_combo"], as_index=False)
        .agg(
            global_avg_rank_r2=("rank_r2", "mean"),
            global_avg_rank_rmse=("rank_rmse", "mean"),
            global_avg_rank_mae=("rank_mae", "mean"),
            global_avg_rank_pearson=("rank_pearson", "mean"),
        )
        .sort_values(["timeline", "modality_combo"])
    )
    global_table = global_table.merge(global_rank, on=["timeline", "modality_combo"], how="left")

    return pred_table, global_table, rank_df


def _plot_anchor_metric_faceted(
    df: pd.DataFrame,
    value_col: str,
    y_label: str,
    out_path: Path,
    modality_order: List[str],
    timeline_order: List[str],
    title_suffix: str = "",
) -> None:
    if df.empty or value_col not in df.columns:
        return

    predictors = sorted(df["predictor"].unique())
    n = len(predictors)
    ncols = 2
    nrows = int(math.ceil(n / ncols))

    cmap = plt.get_cmap("tab10")
    combo_colors = {c: cmap(i % 10) for i, c in enumerate(modality_order)}
    line_styles = ["-", "--", "-.", ":"]
    timeline_styles = {t: line_styles[i % len(line_styles)] for i, t in enumerate(timeline_order)}

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 4.2 * nrows), sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)

    for i, pred in enumerate(predictors):
        ax = axes[i]
        sub = df[df["predictor"] == pred].copy().sort_values("anchor_idx")
        for combo in modality_order:
            for tl in timeline_order:
                part = sub[(sub["modality_combo"] == combo) & (sub["timeline"] == tl)].sort_values("anchor_idx")
                if part.empty:
                    continue
                ax.plot(
                    part["anchor_idx"],
                    part[value_col],
                    color=combo_colors[combo],
                    linestyle=timeline_styles[tl],
                    marker="o",
                    markersize=2.8,
                    linewidth=1.2,
                    alpha=0.92,
                )

        ax.set_title(f"{pred}{title_suffix}")
        ax.set_xlabel("anchor_idx")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    combo_handles = [Line2D([0], [0], color=combo_colors[c], lw=2, label=f"combo:{c}") for c in modality_order]
    tl_handles = [
        Line2D([0], [0], color="black", linestyle=timeline_styles[t], lw=2, label=f"timeline:{t}")
        for t in timeline_order
    ]

    fig.legend(handles=combo_handles, loc="lower center", ncol=min(7, len(combo_handles)), bbox_to_anchor=(0.5, 0.01), frameon=False)
    fig.legend(handles=tl_handles, loc="lower right", bbox_to_anchor=(0.99, 0.01), frameon=False)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_overall_metrics(
    metrics_summary: pd.DataFrame,
    fig_dir: Path,
    modality_order: List[str],
    timeline_order: List[str],
) -> None:
    _plot_anchor_metric_faceted(
        df=metrics_summary,
        value_col="mean_r2",
        y_label="mean R2",
        out_path=fig_dir / "figure_r2_by_anchor_faceted.png",
        modality_order=modality_order,
        timeline_order=timeline_order,
    )
    _plot_anchor_metric_faceted(
        df=metrics_summary,
        value_col="mean_rmse",
        y_label="mean RMSE",
        out_path=fig_dir / "figure_rmse_by_anchor_faceted.png",
        modality_order=modality_order,
        timeline_order=timeline_order,
    )
    _plot_anchor_metric_faceted(
        df=metrics_summary,
        value_col="mean_mae",
        y_label="mean MAE",
        out_path=fig_dir / "figure_mae_by_anchor_faceted.png",
        modality_order=modality_order,
        timeline_order=timeline_order,
    )
    _plot_anchor_metric_faceted(
        df=metrics_summary,
        value_col="mean_pearson",
        y_label="mean Pearson",
        out_path=fig_dir / "figure_pearson_by_anchor_faceted.png",
        modality_order=modality_order,
        timeline_order=timeline_order,
    )


def _plot_year_metrics(
    metrics_by_year: pd.DataFrame,
    fig_dir: Path,
    modality_order: List[str],
    timeline_order: List[str],
) -> None:
    if metrics_by_year.empty:
        return

    for year in sorted(metrics_by_year["year"].unique()):
        sub = metrics_by_year[metrics_by_year["year"] == year].copy()
        _plot_anchor_metric_faceted(
            df=sub,
            value_col="r2",
            y_label="R2",
            out_path=fig_dir / f"figure_r2_by_anchor_faceted_year{int(year)}.png",
            modality_order=modality_order,
            timeline_order=timeline_order,
            title_suffix=f" (year={int(year)})",
        )
        _plot_anchor_metric_faceted(
            df=sub,
            value_col="rmse",
            y_label="RMSE",
            out_path=fig_dir / f"figure_rmse_by_anchor_faceted_year{int(year)}.png",
            modality_order=modality_order,
            timeline_order=timeline_order,
            title_suffix=f" (year={int(year)})",
        )
        _plot_anchor_metric_faceted(
            df=sub,
            value_col="mae",
            y_label="MAE",
            out_path=fig_dir / f"figure_mae_by_anchor_faceted_year{int(year)}.png",
            modality_order=modality_order,
            timeline_order=timeline_order,
            title_suffix=f" (year={int(year)})",
        )
        _plot_anchor_metric_faceted(
            df=sub,
            value_col="pearson",
            y_label="Pearson",
            out_path=fig_dir / f"figure_pearson_by_anchor_faceted_year{int(year)}.png",
            modality_order=modality_order,
            timeline_order=timeline_order,
            title_suffix=f" (year={int(year)})",
        )


def _safe_filename_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_").lower()
    return token or "item"


def _plot_anchor_metric_grouped(
    df: pd.DataFrame,
    *,
    value_col: str,
    y_label: str,
    out_path: Path,
    line_col: str,
    line_order: List[str],
    group_title: str,
    title_suffix: str = "",
) -> None:
    if df.empty or value_col not in df.columns or line_col not in df.columns:
        return

    predictors = sorted(df["predictor"].unique())
    if not predictors:
        return

    n = len(predictors)
    ncols = 2
    nrows = int(math.ceil(n / ncols))

    present_lines = [x for x in line_order if x in set(df[line_col].astype(str).unique())]
    if not present_lines:
        present_lines = sorted(df[line_col].astype(str).unique())

    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    color_map = {name: cmap(i % 10) for i, name in enumerate(present_lines)}
    marker_map = {name: markers[i % len(markers)] for i, name in enumerate(present_lines)}

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 4.2 * nrows), sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)

    for i, pred in enumerate(predictors):
        ax = axes[i]
        sub = df[df["predictor"] == pred].copy().sort_values("anchor_idx")
        for line_name in present_lines:
            part = sub[sub[line_col].astype(str) == str(line_name)].sort_values("anchor_idx")
            if part.empty:
                continue
            ax.plot(
                part["anchor_idx"],
                part[value_col],
                color=color_map[line_name],
                marker=marker_map[line_name],
                linewidth=1.6,
                markersize=3.2,
                alpha=0.95,
                label=str(line_name),
            )

        ax.set_title(f"{pred} | {group_title}{title_suffix}")
        ax.set_xlabel("anchor_idx")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles = [
        Line2D([0], [0], color=color_map[name], marker=marker_map[name], lw=2, label=str(name))
        for name in present_lines
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(4, max(1, len(handles))),
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_timeline_group_views(
    metrics_summary: pd.DataFrame,
    metrics_by_year: pd.DataFrame,
    fig_dir: Path,
    modality_order: List[str],
    timeline_order: List[str],
) -> None:
    base_dir = fig_dir / "by_timeline"
    metric_specs = [
        ("mean_r2", "mean R2", "figure_r2_by_anchor"),
        ("mean_rmse", "mean RMSE", "figure_rmse_by_anchor"),
        ("mean_mae", "mean MAE", "figure_mae_by_anchor"),
        ("mean_pearson", "mean Pearson", "figure_pearson_by_anchor"),
    ]
    year_metric_specs = [
        ("r2", "R2", "figure_r2_by_anchor"),
        ("rmse", "RMSE", "figure_rmse_by_anchor"),
        ("mae", "MAE", "figure_mae_by_anchor"),
        ("pearson", "Pearson", "figure_pearson_by_anchor"),
    ]

    for timeline in timeline_order:
        timeline_df = metrics_summary[metrics_summary["timeline"] == timeline].copy()
        if timeline_df.empty:
            continue

        timeline_dir = base_dir / _safe_filename_token(timeline)
        overall_dir = timeline_dir / "overall"
        year_dir = timeline_dir / "yearwise"

        for value_col, y_label, file_prefix in metric_specs:
            _plot_anchor_metric_grouped(
                timeline_df,
                value_col=value_col,
                y_label=y_label,
                out_path=overall_dir / f"{file_prefix}_timeline_{_safe_filename_token(timeline)}.png",
                line_col="modality_combo",
                line_order=modality_order,
                group_title=f"timeline={timeline}",
            )

        if metrics_by_year.empty:
            continue

        for year in sorted(metrics_by_year["year"].unique()):
            sub = metrics_by_year[
                (metrics_by_year["timeline"] == timeline) & (metrics_by_year["year"] == year)
            ].copy()
            if sub.empty:
                continue
            for value_col, y_label, file_prefix in year_metric_specs:
                _plot_anchor_metric_grouped(
                    sub,
                    value_col=value_col,
                    y_label=y_label,
                    out_path=year_dir / f"{file_prefix}_timeline_{_safe_filename_token(timeline)}_year{int(year)}.png",
                    line_col="modality_combo",
                    line_order=modality_order,
                    group_title=f"timeline={timeline}",
                    title_suffix=f" (year={int(year)})",
                )


def _plot_modality_group_views(
    metrics_summary: pd.DataFrame,
    metrics_by_year: pd.DataFrame,
    fig_dir: Path,
    modality_order: List[str],
    timeline_order: List[str],
) -> None:
    base_dir = fig_dir / "by_modality"
    metric_specs = [
        ("mean_r2", "mean R2", "figure_r2_by_anchor"),
        ("mean_rmse", "mean RMSE", "figure_rmse_by_anchor"),
        ("mean_mae", "mean MAE", "figure_mae_by_anchor"),
        ("mean_pearson", "mean Pearson", "figure_pearson_by_anchor"),
    ]
    year_metric_specs = [
        ("r2", "R2", "figure_r2_by_anchor"),
        ("rmse", "RMSE", "figure_rmse_by_anchor"),
        ("mae", "MAE", "figure_mae_by_anchor"),
        ("pearson", "Pearson", "figure_pearson_by_anchor"),
    ]

    for modality_combo in modality_order:
        combo_df = metrics_summary[metrics_summary["modality_combo"] == modality_combo].copy()
        if combo_df.empty:
            continue

        combo_dir = base_dir / _safe_filename_token(modality_combo)
        overall_dir = combo_dir / "overall"
        year_dir = combo_dir / "yearwise"

        for value_col, y_label, file_prefix in metric_specs:
            _plot_anchor_metric_grouped(
                combo_df,
                value_col=value_col,
                y_label=y_label,
                out_path=overall_dir / f"{file_prefix}_modality_{_safe_filename_token(modality_combo)}.png",
                line_col="timeline",
                line_order=timeline_order,
                group_title=f"modality={modality_combo}",
            )

        if metrics_by_year.empty:
            continue

        for year in sorted(metrics_by_year["year"].unique()):
            sub = metrics_by_year[
                (metrics_by_year["modality_combo"] == modality_combo) & (metrics_by_year["year"] == year)
            ].copy()
            if sub.empty:
                continue
            for value_col, y_label, file_prefix in year_metric_specs:
                _plot_anchor_metric_grouped(
                    sub,
                    value_col=value_col,
                    y_label=y_label,
                    out_path=year_dir / f"{file_prefix}_modality_{_safe_filename_token(modality_combo)}_year{int(year)}.png",
                    line_col="timeline",
                    line_order=timeline_order,
                    group_title=f"modality={modality_combo}",
                    title_suffix=f" (year={int(year)})",
                )


def _plot_global_metric_by_modality(
    pred_table: pd.DataFrame,
    value_col: str,
    y_label: str,
    out_path: Path,
    modality_order: List[str],
    timeline_order: List[str],
) -> None:
    if pred_table.empty:
        return

    predictors = sorted(pred_table["predictor"].unique())
    n = len(predictors)
    ncols = 2
    nrows = int(math.ceil(n / ncols))

    cmap = plt.get_cmap("Set2")
    timeline_colors = {t: cmap(i % 8) for i, t in enumerate(timeline_order)}

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 4.2 * nrows), sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)

    x = np.arange(len(modality_order))
    width = min(0.22, 0.75 / max(1, len(timeline_order)))

    for i, pred in enumerate(predictors):
        ax = axes[i]
        sub = pred_table[pred_table["predictor"] == pred]
        for j, tl in enumerate(timeline_order):
            vals = []
            for combo in modality_order:
                part = sub[(sub["timeline"] == tl) & (sub["modality_combo"] == combo)]
                vals.append(float(part[value_col].iloc[0]) if not part.empty else np.nan)
            ax.bar(x + (j - (len(timeline_order) - 1) / 2) * width, vals, width=width, color=timeline_colors[tl], label=tl)

        ax.set_xticks(x)
        ax.set_xticklabels(modality_order, rotation=30)
        ax.set_title(pred)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25, axis="y")
        ax.legend()

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_rank_summary(
    global_table: pd.DataFrame,
    out_path: Path,
    modality_order: List[str],
    timeline_order: List[str],
) -> None:
    if global_table.empty:
        return

    cmap = plt.get_cmap("Set2")
    timeline_colors = {t: cmap(i % 8) for i, t in enumerate(timeline_order)}

    x = np.arange(len(modality_order))
    width = min(0.22, 0.75 / max(1, len(timeline_order)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, (metric_col, title) in enumerate(
        [
            ("global_avg_rank_r2", "Average Rank by R2 (lower better)"),
            ("global_avg_rank_pearson", "Average Rank by Pearson (lower better)"),
        ]
    ):
        ax = axes[ax_idx]
        for j, tl in enumerate(timeline_order):
            vals = []
            for combo in modality_order:
                part = global_table[(global_table["timeline"] == tl) & (global_table["modality_combo"] == combo)]
                vals.append(float(part[metric_col].iloc[0]) if not part.empty else np.nan)
            ax.bar(x + (j - (len(timeline_order) - 1) / 2) * width, vals, width=width, color=timeline_colors[tl], label=tl)

        ax.set_xticks(x)
        ax.set_xticklabels(modality_order, rotation=30)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_stability_summary(
    pred_table: pd.DataFrame,
    out_path: Path,
    modality_order: List[str],
    timeline_order: List[str],
) -> None:
    if pred_table.empty:
        return

    agg = (
        pred_table.groupby(["timeline", "modality_combo"], as_index=False)
        .agg(
            std_r2=("avg_std_r2_across_anchors", "mean"),
            std_pearson=("avg_std_pearson_across_anchors", "mean"),
        )
        .sort_values(["timeline", "modality_combo"])
    )

    cmap = plt.get_cmap("Set2")
    timeline_colors = {t: cmap(i % 8) for i, t in enumerate(timeline_order)}

    x = np.arange(len(modality_order))
    width = min(0.22, 0.75 / max(1, len(timeline_order)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, (metric_col, title) in enumerate(
        [
            ("std_r2", "Stability: std(R2) across outer folds"),
            ("std_pearson", "Stability: std(Pearson) across outer folds"),
        ]
    ):
        ax = axes[ax_idx]
        for j, tl in enumerate(timeline_order):
            vals = []
            for combo in modality_order:
                part = agg[(agg["timeline"] == tl) & (agg["modality_combo"] == combo)]
                vals.append(float(part[metric_col].iloc[0]) if not part.empty else np.nan)
            ax.bar(x + (j - (len(timeline_order) - 1) / 2) * width, vals, width=width, color=timeline_colors[tl], label=tl)

        ax.set_xticks(x)
        ax.set_xticklabels(modality_order, rotation=30)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_summary_md(
    out_dir: Path,
    run_cfg: dict,
    timeline_anchor_df: pd.DataFrame,
    availability_df: pd.DataFrame,
    metrics_summary: pd.DataFrame,
    metrics_by_year: pd.DataFrame,
    pred_table: pd.DataFrame,
    global_table: pd.DataFrame,
    metrics_fold: pd.DataFrame,
) -> str:
    def _table(df: pd.DataFrame, max_rows: int = 12) -> str:
        if df.empty:
            return "(空)"
        show = df.head(max_rows)
        text = show.to_string(index=False)
        if len(df) > max_rows:
            text += f"\n... (仅展示前 {max_rows} 行, 共 {len(df)} 行)"
        return text

    validation_scenario = _normalize_validation_scenario(
        run_cfg.get("runtime", {}).get("validation_scenario", run_cfg.get("experiment", {}).get("validation_scenario", "within_season"))
    )
    result_tag = str(run_cfg.get("experiment", {}).get("result_tag", "Result2.4"))
    if validation_scenario == "loso":
        title = f"# {result_tag}: LOSO Temporal Multi-timeline Multimodal Ablation Summary"
        purpose = "比较 day / gdd / gdd_rel_heading 三条时间轴下，不同多组学组合（H/C/G 及其融合）在 temporal 框架中的预测准确性与稳定性（LOSO）。"
        split_head = "## 6. LOSO 说明"
        split_line = "使用 `loso_{target_year}_fold{k}.json`；test 为留出年份，5 个 fold 子模型集成预测。"
        ensemble_line = "每个 target_year 用 5 个 fold 子模型分别预测同一 test year，并取均值作为最终预测。"
        run_cmd = (
            "python scripts/run_loso_temporal_gdd_relheading_multimodal_ablation.py "
            "--config configs/loso_temporal_gdd_relheading_multimodal_ablation.yaml"
        )
    else:
        title = f"# {result_tag}: Within-Season Temporal Multi-timeline Multimodal Ablation Summary"
        purpose = "比较 day / gdd / gdd_rel_heading 三条时间轴下，不同多组学组合（H/C/G 及其融合）在 temporal 框架中的预测准确性与稳定性。"
        split_head = "## 6. within-season nested CV 说明"
        split_line = "使用 `within_season_{year}_outer{i}_inner{j}.json`；outer 用于最终测试，inner 用于调参与集成。"
        ensemble_line = "每个 outer fold 用 4 个 inner 子模型分别预测 outer test，并取均值作为最终 outer 预测。"
        run_cmd = (
            "python scripts/run_within_season_temporal_gdd_relheading_multimodal_ablation.py "
            "--config configs/within_season_temporal_gdd_relheading_multimodal_ablation.yaml"
        )

    lines: List[str] = []
    lines.append(title)
    lines.append("")
    lines.append("## 1. 实验目的")
    lines.append(purpose)
    lines.append("")
    lines.append("## 2. 为什么本实验强调 gdd 与 gdd_rel_heading")
    lines.append("gdd 与 gdd_rel_heading 是热时间体系的核心对照；同时保留 day 作为基准轴，便于判断热时间轴相对日历轴的收益或代价。")
    lines.append("")
    lines.append("## 3. timeline 定义")
    lines.append("- day: 日历天步长 1")
    lines.append("- gdd: 绝对积温步长 5")
    lines.append("- gdd_rel_heading: 相对抽穗期积温步长 5")
    lines.append("")
    lines.append("## 4. 多组学组合定义")
    lines.append(f"- 组合集合: {run_cfg['experiment']['modality_combos']}")
    lines.append("")
    lines.append("## 5. temporal prefix 特征构造")
    lines.append("每个 timeline 的每个 anchor 仅使用 `tb<=anchor_tb` 的 H/C 前缀；G 使用全部静态 `grm_pc_*`。")
    lines.append("")
    lines.append(split_head)
    lines.append(split_line)
    lines.append("")
    lines.append("## 7. 缺失值处理策略")
    lines.append("统一使用 `SimpleImputer(strategy=\"median\", add_indicator=True)`，且所有预处理均在训练折内 fit。")
    lines.append("`gdd_rel_heading` 采用统一列空间 + 样本允许缺失（软对齐），不做硬裁剪。")
    lines.append("")
    lines.append("## 8. 模型与 Optuna 搜索")
    lines.append(f"- 运行预测器: {run_cfg['experiment']['predictors_run']}")
    lines.append(f"- n_trials: {run_cfg['optuna']['n_trials']}, lambda_gap: {run_cfg['optuna']['lambda_gap']}")
    lines.append("- objective = mean_train_loss + mean_val_loss + lambda_gap * max(0, mean_val_loss - mean_train_loss)")
    lines.append("")
    lines.append("## 9. outer test 集成方式")
    lines.append(ensemble_line)
    lines.append("")
    lines.append("## 10. 总体准确性结果")
    lines.append(_table(global_table[["timeline", "modality_combo", "global_avg_r2", "global_avg_rmse", "global_avg_mae", "global_avg_pearson"]]))
    lines.append("")
    lines.append("## 11. 分年结果")
    lines.append(_table(metrics_by_year[["timeline", "modality_combo", "predictor", "year", "r2", "rmse", "mae", "pearson", "n_samples"]]))
    lines.append("")
    lines.append("## 12. 多组学消融结果")
    lines.append(_table(pred_table[["timeline", "modality_combo", "predictor", "avg_r2_across_anchors", "avg_pearson_across_anchors"]]))
    lines.append("")
    lines.append("## 13. gdd vs gdd_rel_heading 对比")
    pair = global_table[global_table["timeline"].isin(["gdd", "gdd_rel_heading"])].copy()
    lines.append(_table(pair[["timeline", "modality_combo", "global_avg_r2", "global_avg_pearson"]]))
    lines.append("")
    lines.append("## 14. 稳定性结果")
    lines.append(_table(pred_table[["timeline", "modality_combo", "predictor", "avg_std_r2_across_anchors", "avg_std_pearson_across_anchors"]]))
    lines.append("")
    lines.append("## 15. 缺失模式与结果解释")
    rel = availability_df[availability_df["timeline"] == "gdd_rel_heading"]
    if not rel.empty:
        lines.append(
            f"gdd_rel_heading overall_missing_ratio 平均={rel['overall_missing_ratio'].mean():.4f}, "
            f"最大={rel['overall_missing_ratio'].max():.4f}; "
            f"severely_missing_sample_count 平均={rel['severely_missing_sample_count'].mean():.1f}, "
            f"最大={int(rel['severely_missing_sample_count'].max())}。"
        )
    else:
        lines.append("(无 gdd_rel_heading 可用样本)")
    lines.append("")
    lines.append("## 16. 过拟合检查")
    if metrics_fold.empty:
        lines.append("(空)")
    else:
        overfit = (
            metrics_fold.groupby(["timeline", "modality_combo", "predictor"], as_index=False)[
                ["best_mean_train_loss", "best_mean_val_loss", "best_gap"]
            ]
            .mean()
            .sort_values(["timeline", "modality_combo", "predictor"])
        )
        lines.append(_table(overfit))
    lines.append("")
    lines.append("## 17. 局限性")
    lines.append("高维前缀特征在后期 anchor 会带来较高方差；当某些分组样本方差为 0 时，Pearson 按规则返回 NaN。")
    lines.append("")
    lines.append("## 18. 可复现性信息")
    lines.append(f"- 输出目录: `{out_dir}`")
    lines.append(f"- 运行时间: {now_iso()}")
    lines.append(f"- 随机种子: {run_cfg['experiment']['random_seed']}")
    lines.append(f"- 运行命令: `{run_cmd}`")
    lines.append("- 关键文件: run_config.yaml / metrics_by_fold.csv / metrics_summary.csv / metrics_by_year.csv")
    lines.append("")
    lines.append("## 19. 最终结论")

    if not global_table.empty:
        best_r2 = global_table.sort_values("global_avg_r2", ascending=False).iloc[0]
        best_pr = global_table.sort_values("global_avg_pearson", ascending=False).iloc[0]
        lines.append(
            f"- 按 R2 最佳: timeline={best_r2['timeline']}, modality_combo={best_r2['modality_combo']}, "
            f"global_avg_r2={best_r2['global_avg_r2']:.4f}"
        )
        lines.append(
            f"- 按 Pearson 最佳: timeline={best_pr['timeline']}, modality_combo={best_pr['modality_combo']}, "
            f"global_avg_pearson={best_pr['global_avg_pearson']:.4f}"
        )
        if (best_r2["timeline"], best_r2["modality_combo"]) != (best_pr["timeline"], best_pr["modality_combo"]):
            lines.append("- 主指标与辅助指标最优组合不一致，建议按应用目标（误差 vs 相关性）做权衡。")

    return "\n".join(lines)


def run_experiment(config_path: Path) -> Path:
    cfg = read_yaml(config_path)

    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})
    optuna_cfg = cfg.get("optuna", {})
    preprocess_cfg = cfg.get("preprocessing", {})
    output_cfg = cfg.get("output", {})

    output_dir = _resolve_output_dir(output_cfg)
    fig_dir = output_dir / "figures"
    ensure_dir(fig_dir)

    seed = int(exp_cfg.get("random_seed", 42))
    _set_global_seed(seed)

    validation_scenario = _normalize_validation_scenario(exp_cfg.get("validation_scenario", "within_season"))
    if validation_scenario not in {"within_season", "loso"}:
        raise ValueError(f"不支持的 validation_scenario={validation_scenario}，仅支持 within_season 或 loso。")

    progress_enabled = bool(output_cfg.get("progress_log", True))
    _append_progress(
        output_dir,
        enabled=progress_enabled,
        event="run_start",
        payload={
            "validation_scenario": validation_scenario,
            "n_anchor_bins": int(exp_cfg.get("n_anchor_bins", 20)),
        },
    )

    split_dir = Path(data_cfg.get("split_dir", "data/processed/splits"))
    target_col = str(data_cfg.get("target_col", "ActualYD"))
    genotype_representation = str(exp_cfg.get("genotype_representation", "grm_pca"))
    n_anchor_bins = int(exp_cfg.get("n_anchor_bins", 20))
    modality_combos = _parse_modality_combos(exp_cfg.get("modality_combos", DEFAULT_MODALITY_COMBOS))

    timeline_dirs = data_cfg.get(
        "timeline_dirs",
        {
            "day": "data/processed/model_inputs/day",
            "gdd": "data/processed/model_inputs/gdd_abs",
            "gdd_rel_heading": "data/processed/model_inputs/gdd_rel_heading",
        },
    )
    timeline_order = list(timeline_dirs.keys())

    predictors_supported = [
        str(x).lower() for x in exp_cfg.get("predictors_supported", ["ridge", "lasso", "elasticnet", "lightgbm", "extratrees"])
    ]
    predictors_run = [str(x).lower() for x in exp_cfg.get("predictors_run", predictors_supported)]

    model_cfg = exp_cfg.get("model_backends", {})
    skip_lightgbm_runtime = bool(exp_cfg.get("skip_lightgbm_runtime", True))

    active_predictors: List[str] = []
    for p in predictors_run:
        if p not in predictors_supported:
            continue
        if p == "lightgbm" and skip_lightgbm_runtime:
            continue
        if p == "lightgbm" and str(model_cfg.get("lightgbm_backend", "sklearn_gbrt")).lower() == "native" and not HAS_LIGHTGBM:
            continue
        active_predictors.append(p)

    if not active_predictors:
        raise RuntimeError("没有可运行的预测器，请检查 predictors_run / skip_lightgbm_runtime 配置。")

    n_trials = int(optuna_cfg.get("n_trials", 3))
    lambda_gap = float(optuna_cfg.get("lambda_gap", 1.0))
    pruner_startup_trials = int(optuna_cfg.get("pruner_startup_trials", 1))

    raw_split_groups = load_split_groups(split_dir, validation_scenario=validation_scenario)

    timeline_anchor_rows = []
    availability_rows = []
    metrics_fold_rows = []
    oof_rows = []
    inner_pred_rows = []
    trial_rows = []

    timeline_stats = {}
    total_tasks = len(timeline_dirs) * n_anchor_bins * len(modality_combos) * max(1, len(active_predictors))
    task_idx = 0

    combo_order = list(modality_combos)

    for timeline_name, timeline_path in timeline_dirs.items():
        input_dir = Path(timeline_path)
        print(f"[TIMELINE] {timeline_name} -> {input_dir}", flush=True)

        bundle = load_day_model_inputs(
            input_dir=input_dir,
            target_col=target_col,
            genotype_representation=genotype_representation,
        )

        prepared_groups, split_stats = _prepare_split_groups(raw_split_groups, set(bundle.meta.index))
        if not prepared_groups:
            raise RuntimeError(f"timeline={timeline_name} 无可用 {validation_scenario} split。")

        anchors = build_anchor_definitions(bundle, n_anchor_bins=n_anchor_bins)

        timeline_stats[timeline_name] = {
            "input_dir": input_dir.as_posix(),
            "n_samples": int(bundle.meta.shape[0]),
            "n_common_tbs": int(len(bundle.common_tbs)),
            "n_anchors": int(len(anchors)),
            "split_filter_stats": split_stats,
        }

        for anchor in anchors:
            x_all, x_info = build_feature_matrix(bundle, anchor.anchor_key, input_type="temporal")
            x_all = sanitize_feature_matrix(x_all)

            timeline_anchor_rows.append(
                {
                    "timeline": timeline_name,
                    "anchor_idx": int(anchor.anchor_idx),
                    "anchor_tb": int(anchor.anchor_tb),
                    "anchor_tb_token": str(anchor.anchor_tb_token),
                    "h_prefix_col_count": int(x_info["n_features_h"]),
                    "c_prefix_col_count": int(x_info["n_features_c"]),
                    "g_feature_count": int(x_info["n_features_g"]),
                }
            )

            y_full = bundle.y
            meta_full = bundle.meta

            for modality_combo in combo_order:
                x_mod, mod_info = _build_modality_feature_matrix(
                    x_all=x_all,
                    x_info=x_info,
                    modality_combo=modality_combo,
                    sample_index=bundle.meta.index,
                )
                x_mod = sanitize_feature_matrix(x_mod)

                availability_rows.append(
                    _availability_row(
                        timeline=timeline_name,
                        anchor_idx=anchor.anchor_idx,
                        anchor_tb=anchor.anchor_tb,
                        modality_combo=modality_combo,
                        x=x_mod,
                        n_h=mod_info["n_features_h"],
                        n_c=mod_info["n_features_c"],
                        n_g=mod_info["n_features_g"],
                    )
                )

                for predictor in active_predictors:
                    task_idx += 1
                    print(
                        f"[TASK {task_idx}/{total_tasks}] tl={timeline_name} anchor={anchor.anchor_tb} combo={modality_combo} predictor={predictor} "
                        f"features={x_mod.shape[1]}",
                        flush=True,
                    )
                    _append_progress(
                        output_dir,
                        enabled=progress_enabled,
                        event="task_start",
                        payload={
                            "task_idx": task_idx,
                            "total_tasks": total_tasks,
                            "timeline": timeline_name,
                            "anchor_idx": int(anchor.anchor_idx),
                            "anchor_tb": float(anchor.anchor_tb),
                            "modality_combo": modality_combo,
                            "predictor": predictor,
                            "n_features": int(x_mod.shape[1]),
                        },
                    )

                    fold_rows_for_combo = []

                    for (target_year, outer_fold), group_splits in prepared_groups.items():
                        payload = _build_outer_payload(x_mod, y_full, meta_full, group_splits)
                        inner_payload = payload["inner_payload"]
                        x_test = payload["x_test"]
                        y_test = payload["y_test"]
                        meta_test = payload["meta_test"]

                        if len(x_test) == 0 or not inner_payload:
                            continue

                        t_idx = timeline_order.index(timeline_name)
                        c_idx = combo_order.index(modality_combo)
                        p_idx = active_predictors.index(predictor)
                        sampler_seed = (
                            seed
                            + t_idx * 100000
                            + c_idx * 10000
                            + p_idx * 1000
                            + int(anchor.anchor_idx) * 10
                            + int(target_year % 100)
                            + int(outer_fold)
                        )

                        sampler = optuna.samplers.TPESampler(seed=int(sampler_seed))
                        pruner = optuna.pruners.MedianPruner(n_startup_trials=pruner_startup_trials, n_warmup_steps=1)
                        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

                        def objective(trial):
                            params = suggest_params(trial, predictor, model_cfg=model_cfg)
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
                                    model_cfg=model_cfg,
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
                        for t in sorted_trials:
                            trial_rows.append(
                                {
                                    "timeline": timeline_name,
                                    "anchor_idx": int(anchor.anchor_idx),
                                    "anchor_tb": int(anchor.anchor_tb),
                                    "modality_combo": modality_combo,
                                    "predictor": predictor,
                                    "outer_fold": int(outer_fold),
                                    "target_year": int(target_year),
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
                                }
                            )

                        best = study.best_trial

                        inner_preds = []
                        for inner in inner_payload:
                            pipe = build_pipeline(
                                predictor=predictor,
                                params=best.params,
                                n_features=inner["x_train"].shape[1],
                                preprocess_cfg=preprocess_cfg,
                                seed=int(sampler_seed) + int(inner["inner_fold"]) + 17,
                                model_cfg=model_cfg,
                            )
                            pipe.fit(inner["x_train"], inner["y_train"])
                            pred = pipe.predict(x_test)
                            inner_preds.append(pred)

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
                                        "outer_fold": int(outer_fold),
                                        "inner_fold": int(inner["inner_fold"]),
                                        "timeline": timeline_name,
                                        "anchor_idx": int(anchor.anchor_idx),
                                        "anchor_tb": int(anchor.anchor_tb),
                                        "modality_combo": modality_combo,
                                        "predictor": predictor,
                                        "y_true": y_true_i,
                                        "y_pred_inner_model": float(pred_mat[i, j]),
                                        "y_pred_ensemble_mean": y_ens_i,
                                    }
                                )

                            oof_rows.append(
                                {
                                    "plot_id": pid,
                                    "year": int(meta_test.loc[pid, "year"]),
                                    "outer_fold": int(outer_fold),
                                    "timeline": timeline_name,
                                    "anchor_idx": int(anchor.anchor_idx),
                                    "anchor_tb": int(anchor.anchor_tb),
                                    "modality_combo": modality_combo,
                                    "predictor": predictor,
                                    "y_true": y_true_i,
                                    "y_pred": y_ens_i,
                                }
                            )

                        fold_row = {
                                "timeline": timeline_name,
                                "anchor_idx": int(anchor.anchor_idx),
                                "anchor_tb": int(anchor.anchor_tb),
                                "modality_combo": modality_combo,
                                "predictor": predictor,
                                "target_year": int(target_year),
                                "outer_fold": int(outer_fold),
                                "best_trial_number": int(best.number),
                                "best_objective": float(best.value),
                                "best_mean_train_loss": float(best.user_attrs.get("mean_train_loss", np.nan)),
                                "best_mean_val_loss": float(best.user_attrs.get("mean_val_loss", np.nan)),
                                "best_gap": float(best.user_attrs.get("gap", np.nan)),
                                "train_r2": float(best.user_attrs.get("mean_train_r2", np.nan)),
                                "train_rmse": float(best.user_attrs.get("mean_train_rmse", np.nan)),
                                "train_mae": float(best.user_attrs.get("mean_train_mae", np.nan)),
                                "train_pearson": float(best.user_attrs.get("mean_train_pearson", np.nan)),
                                "val_r2": float(best.user_attrs.get("mean_val_r2", np.nan)),
                                "val_rmse": float(best.user_attrs.get("mean_val_rmse", np.nan)),
                                "val_mae": float(best.user_attrs.get("mean_val_mae", np.nan)),
                                "val_pearson": float(best.user_attrs.get("mean_val_pearson", np.nan)),
                                "test_r2": float(fold_metrics["r2"]),
                                "test_rmse": float(fold_metrics["rmse"]),
                                "test_mae": float(fold_metrics["mae"]),
                                "test_pearson": float(fold_metrics["pearson_r"]),
                                "n_train": int(np.mean([len(inner["train_ids"]) for inner in inner_payload])),
                                "n_val": int(np.mean([len(inner["val_ids"]) for inner in inner_payload])),
                                "n_test": int(len(payload["test_ids"])),
                                "n_features_total": int(mod_info["n_features_total"]),
                                "n_features_h": int(mod_info["n_features_h"]),
                                "n_features_c": int(mod_info["n_features_c"]),
                                "n_features_g": int(mod_info["n_features_g"]),
                                "n_inner_folds": int(len(inner_payload)),
                                "n_optuna_trials": int(n_trials),
                            }
                        metrics_fold_rows.append(fold_row)
                        fold_rows_for_combo.append(fold_row)

                    if fold_rows_for_combo:
                        combo_df = pd.DataFrame(fold_rows_for_combo)
                        _append_progress(
                            output_dir,
                            enabled=progress_enabled,
                            event="task_end",
                            payload={
                                "task_idx": task_idx,
                                "total_tasks": total_tasks,
                                "timeline": timeline_name,
                                "anchor_idx": int(anchor.anchor_idx),
                                "anchor_tb": float(anchor.anchor_tb),
                                "modality_combo": modality_combo,
                                "predictor": predictor,
                                "mean_r2": float(combo_df["test_r2"].mean()),
                            },
                        )
                    else:
                        _append_progress(
                            output_dir,
                            enabled=progress_enabled,
                            event="task_end",
                            payload={
                                "task_idx": task_idx,
                                "total_tasks": total_tasks,
                                "timeline": timeline_name,
                                "anchor_idx": int(anchor.anchor_idx),
                                "anchor_tb": float(anchor.anchor_tb),
                                "modality_combo": modality_combo,
                                "predictor": predictor,
                                "mean_r2": None,
                            },
                        )

    timeline_anchor_df = pd.DataFrame(timeline_anchor_rows)
    availability_df = pd.DataFrame(availability_rows)
    metrics_fold_df = pd.DataFrame(metrics_fold_rows)
    oof_df = pd.DataFrame(oof_rows)
    inner_pred_df = pd.DataFrame(inner_pred_rows)
    trial_df = pd.DataFrame(trial_rows)

    metrics_summary_df = _build_metrics_summary(metrics_fold_df)
    metrics_by_year_df = _build_metrics_by_year(oof_df)
    pred_table_df, global_table_df, rank_df = _build_rank_tables(metrics_summary_df)

    # 输出文件
    write_json(timeline_anchor_rows, output_dir / "timeline_anchor_bins.json")
    metrics_fold_df.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    metrics_summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    metrics_by_year_df.to_csv(output_dir / "metrics_by_year.csv", index=False)
    pred_table_df.to_csv(output_dir / "metrics_by_timeline_modality_predictor.csv", index=False)
    global_table_df.to_csv(output_dir / "metrics_by_timeline_modality_global.csv", index=False)
    oof_df.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    inner_pred_df.to_parquet(output_dir / "outer_test_inner_ensemble_predictions.parquet", index=False)
    availability_df.to_csv(output_dir / "timeline_anchor_modality_availability.csv", index=False)
    trial_df.to_csv(output_dir / "optuna_trials.csv", index=False)

    # 绘图
    _plot_overall_metrics(
        metrics_summary=metrics_summary_df,
        fig_dir=fig_dir,
        modality_order=combo_order,
        timeline_order=timeline_order,
    )
    _plot_year_metrics(
        metrics_by_year=metrics_by_year_df,
        fig_dir=fig_dir,
        modality_order=combo_order,
        timeline_order=timeline_order,
    )
    _plot_timeline_group_views(
        metrics_summary=metrics_summary_df,
        metrics_by_year=metrics_by_year_df,
        fig_dir=fig_dir,
        modality_order=combo_order,
        timeline_order=timeline_order,
    )
    _plot_modality_group_views(
        metrics_summary=metrics_summary_df,
        metrics_by_year=metrics_by_year_df,
        fig_dir=fig_dir,
        modality_order=combo_order,
        timeline_order=timeline_order,
    )

    _plot_global_metric_by_modality(
        pred_table=pred_table_df,
        value_col="avg_r2_across_anchors",
        y_label="global mean R2 across anchors",
        out_path=fig_dir / "figure_global_r2_by_modality_timeline.png",
        modality_order=combo_order,
        timeline_order=timeline_order,
    )
    _plot_global_metric_by_modality(
        pred_table=pred_table_df,
        value_col="avg_pearson_across_anchors",
        y_label="global mean Pearson across anchors",
        out_path=fig_dir / "figure_global_pearson_by_modality_timeline.png",
        modality_order=combo_order,
        timeline_order=timeline_order,
    )
    _plot_rank_summary(
        global_table=global_table_df,
        out_path=fig_dir / "figure_rank_summary.png",
        modality_order=combo_order,
        timeline_order=timeline_order,
    )
    _plot_stability_summary(
        pred_table=pred_table_df,
        out_path=fig_dir / "figure_stability_summary.png",
        modality_order=combo_order,
        timeline_order=timeline_order,
    )

    run_snapshot = {
        **cfg,
        "runtime": {
            "timestamp": now_iso(),
            "git_commit_hash": get_git_commit_hash(Path.cwd()),
            "validation_scenario": validation_scenario,
            "objective_definition": "mean_train_loss + mean_val_loss + lambda_gap * max(0, mean_val_loss - mean_train_loss)",
            "missing_policy": "SimpleImputer(strategy='median', add_indicator=True) for all timelines and modality combos",
            "gdd_rel_heading_soft_alignment": "统一列空间+样本允许缺失；不做硬裁剪",
            "active_predictors": active_predictors,
            "timeline_data_stats": timeline_stats,
            "modality_combos": combo_order,
        },
    }
    write_yaml(run_snapshot, output_dir / "run_config.yaml")

    summary_md = _build_summary_md(
        out_dir=output_dir,
        run_cfg=run_snapshot,
        timeline_anchor_df=timeline_anchor_df,
        availability_df=availability_df,
        metrics_summary=metrics_summary_df,
        metrics_by_year=metrics_by_year_df,
        pred_table=pred_table_df,
        global_table=global_table_df,
        metrics_fold=metrics_fold_df,
    )
    (output_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    print(f"[DONE] 输出目录: {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run within-season temporal gdd/day/gdd_rel_heading multimodal ablation")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/within_season_temporal_gdd_relheading_multimodal_ablation.yaml"),
        help="配置文件路径",
    )
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
