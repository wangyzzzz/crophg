from __future__ import annotations

import json
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.anchorwise.data_loader import MultiTargetDataBundle, load_multitarget_model_inputs
from models.anchorwise.feature_builder import AnchorDefinition, build_anchor_definitions
from models.anchorwise.modeling import sanitize_feature_matrix
from models.common.io_utils import ensure_dir, get_git_commit_hash, now_iso, read_yaml, write_json, write_yaml
from models.result31.runner import _resolve_split_groups_for_scenario
from models.result32.runner import _prepare_split_groups
from crophg.common.report_utils import markdown_table
from models.result33.runner import (
    VEGETATION_INDEX_TOKENS,
    _append_progress,
    _build_metrics_by_year,
    _build_metrics_summary,
    _h_feature_base_name,
    _normalize_modality_variant,
    _resolve_active_predictors,
    _resolve_output_dir,
    _resolve_timeline_dirs,
    _resolve_vi_names,
    _run_single_task,
    _set_global_seed,
    _tail_fill_hyperspectral_tail_only,
    _to_records_df,
    _write_summary,
    _normalize_vi_name,
)

warnings.filterwarnings("ignore", message="Skipping features without any observed values")

KNOWN_VI_NAMES = {f"vi_{token.lower()}" for token in VEGETATION_INDEX_TOKENS}


def _normalize_anchor_vi_base_name(col_name: str) -> str | None:
    base = _h_feature_base_name(str(col_name)).strip()
    if not base:
        return None
    token = base.lower().replace("-", "_").replace(" ", "_")
    if token.startswith("vi_"):
        return token if token in KNOWN_VI_NAMES else None
    try:
        normalized = _normalize_vi_name(token)
    except ValueError:
        return None
    return normalized if normalized in KNOWN_VI_NAMES else None


def load_best_anchor_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "scenario",
        "target",
        "timeline",
        "predictor",
        "best_anchor",
        "best_anchor_tb",
        "best_anchor_phase",
        "best_anchor_band",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"best anchor 文件缺少必要列: {missing}")

    out = df.loc[
        :,
        [
            "scenario",
            "target",
            "timeline",
            "predictor",
            "best_anchor",
            "best_anchor_tb",
            "best_anchor_phase",
            "best_anchor_band",
        ],
    ].copy()
    out = out.rename(
        columns={
            "best_anchor": "anchor_idx",
            "best_anchor_tb": "anchor_tb",
            "best_anchor_phase": "anchor_phase",
            "best_anchor_band": "anchor_band",
        }
    )
    out["anchor_idx"] = out["anchor_idx"].astype(int)
    return out.sort_values(["scenario", "target", "timeline", "predictor"]).reset_index(drop=True)


def build_best_anchor_task_table(
    best_anchor_df: pd.DataFrame,
    *,
    requested_vi_names: list[str] | None = None,
    available_vi_names: dict[tuple[str, int], set[str]] | None = None,
) -> pd.DataFrame:
    if best_anchor_df.empty:
        return pd.DataFrame()

    normalized_vi_names = None
    if requested_vi_names:
        normalized_vi_names = [_normalize_vi_name(x) for x in requested_vi_names]

    rows: list[dict[str, Any]] = []
    for row in best_anchor_df.to_dict("records"):
        timeline = str(row["timeline"])
        anchor_idx = int(row["anchor_idx"])
        allowed = None
        if available_vi_names is not None:
            allowed = available_vi_names.get((timeline, anchor_idx), set())
        vi_names = normalized_vi_names
        if vi_names is None and allowed is not None:
            vi_names = sorted(allowed)
        if not vi_names:
            continue
        for vi_name in vi_names:
            if allowed is not None and vi_name not in allowed:
                continue
            for modality_variant in ["H_SINGLE_VI", "GH_SINGLE_VI"]:
                rows.append(
                    {
                        **row,
                        "anchor_idx": anchor_idx,
                        "vi_name": vi_name,
                        "modality_variant": modality_variant,
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["scenario", "target", "timeline", "predictor", "anchor_idx", "vi_name", "modality_variant"]
    ).reset_index(drop=True)


def build_result33bc_light_tasks_from_config(
    cfg: dict[str, Any],
    best_anchor_df: pd.DataFrame,
    *,
    available_vi_names: dict[tuple[str, int], set[str]] | None = None,
) -> pd.DataFrame:
    if best_anchor_df.empty:
        return pd.DataFrame()

    exp_cfg = cfg.get("experiment", {})
    timeline_dirs = _resolve_timeline_dirs(cfg.get("data", {}))
    targets = {str(x) for x in exp_cfg.get("targets", [])}
    predictors = {str(x).lower() for x in _resolve_active_predictors(exp_cfg)}

    work = best_anchor_df.copy()
    work["target"] = work["target"].astype(str)
    work["predictor"] = work["predictor"].astype(str).str.lower()
    work["timeline"] = work["timeline"].astype(str)
    work = work[work["timeline"].isin(set(timeline_dirs.keys()))]
    if targets:
        work = work[work["target"].isin(targets)]
    if predictors:
        work = work[work["predictor"].isin(predictors)]
    if work.empty:
        return pd.DataFrame()

    return build_best_anchor_task_table(
        work,
        requested_vi_names=_resolve_vi_names(exp_cfg),
        available_vi_names=available_vi_names,
    )


def summarize_available_vi_names(
    timeline_name: str,
    available_vi_names: dict[int, set[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for anchor_idx, vi_names in sorted(available_vi_names.items()):
        for vi_name in sorted(vi_names):
            rows.append(
                {
                    "timeline": str(timeline_name),
                    "anchor_idx": int(anchor_idx),
                    "vi_name": str(vi_name),
                }
            )
    return pd.DataFrame(rows)


def collect_available_vi_names_by_anchor(
    bundle: MultiTargetDataBundle,
    anchors: list[AnchorDefinition],
    anchor_indices: list[int] | None = None,
) -> dict[int, set[str]]:
    selected = None if anchor_indices is None else {int(x) for x in anchor_indices}
    out: dict[int, set[str]] = {}
    for anchor in anchors:
        anchor_idx = int(anchor.anchor_idx)
        if selected is not None and anchor_idx not in selected:
            continue
        h_cols = bundle.hyperspectral_tb_map.get(anchor.anchor_key, [])
        vi_names = set()
        for col in h_cols:
            normalized = _normalize_anchor_vi_base_name(str(col))
            if normalized is None:
                continue
            vi_names.add(normalized)
        out[anchor_idx] = vi_names
    return out


def build_result33bc_light_feature_matrix(
    *,
    bundle: MultiTargetDataBundle,
    anchors: list[AnchorDefinition],
    anchor_idx: int,
    vi_name: str,
    use_genotype: bool,
) -> tuple[pd.DataFrame, dict]:
    modality_variant = _normalize_modality_variant("GH_SINGLE_VI" if use_genotype else "H_SINGLE_VI")
    anchor_idx = int(anchor_idx)
    if anchor_idx < 0 or anchor_idx >= len(anchors):
        raise IndexError(f"anchor_idx 超出范围: {anchor_idx} / {len(anchors)}")

    filled_bundle = getattr(bundle, "_result33_tail_filled_bundle", None)
    if filled_bundle is None:
        filled_bundle = _tail_fill_hyperspectral_tail_only(bundle)
        setattr(bundle, "_result33_tail_filled_bundle", filled_bundle)
    bundle = filled_bundle

    target_vi_name = _normalize_vi_name(vi_name)
    anchor = anchors[anchor_idx]
    static_h_cols = bundle.hyperspectral_tb_map.get(anchor.anchor_key, [])
    h_cols = [col for col in static_h_cols if _normalize_anchor_vi_base_name(str(col)) == target_vi_name]
    if not h_cols:
        raise ValueError(f"{modality_variant} 在 anchor_idx={anchor_idx} 未找到 VI 列: {[target_vi_name]}")

    g_cols = list(bundle.x_genotype.columns) if use_genotype else []
    x_parts = [bundle.x_hyperspectral.loc[bundle.meta.index, h_cols].copy()]
    if g_cols:
        x_parts.append(bundle.x_genotype.loc[bundle.meta.index, g_cols].copy())
    x = sanitize_feature_matrix(pd.concat(x_parts, axis=1))
    info = {
        "modality_variant": modality_variant,
        "n_features_total": int(x.shape[1]),
        "n_features_h": int(len(h_cols)),
        "n_features_g": int(len(g_cols)),
        "h_cols": list(h_cols),
        "g_cols": list(g_cols),
        "anchor_tb": int(anchor.anchor_tb),
        "anchor_phase": anchor.anchor_phase,
        "vi_name": target_vi_name,
        "vi_names": [target_vi_name],
    }
    return x, info


def _write_light_summary(
    *,
    output_dir: Path,
    cfg: dict[str, Any],
    best_anchor_df: pd.DataFrame,
    available_vi_df: pd.DataFrame,
    tasks_df: pd.DataFrame,
    split_usage_df: pd.DataFrame,
    metrics_summary_df: pd.DataFrame,
) -> None:
    lines = [
        "# Result 3.3B/C Light Summary",
        "",
        "## 1. Config",
        "",
        f"- config_path: `{cfg.get('config_path', '')}`",
        f"- timelines: `{', '.join(cfg.get('data', {}).get('timeline_dirs', {}).keys())}`",
        f"- scenarios: `{', '.join(cfg.get('data', {}).get('scenarios', {}).keys())}`",
        f"- targets: `{', '.join(cfg.get('experiment', {}).get('targets', []))}`",
        f"- predictors: `{', '.join(cfg.get('experiment', {}).get('predictors_run', []))}`",
        "",
        "## 2. Best Anchor",
        "",
        markdown_table(best_anchor_df, max_rows=30),
        "",
        "## 3. Available VI",
        "",
        markdown_table(available_vi_df, max_rows=30),
        "",
        "## 4. Task Table",
        "",
        markdown_table(tasks_df, max_rows=30),
        "",
        "## 5. Split Usage",
        "",
        markdown_table(split_usage_df, max_rows=30),
        "",
        "## 6. Metrics Summary",
        "",
        markdown_table(metrics_summary_df, max_rows=30),
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_result33bc_light(config_path: Path, *, dry_run: bool = False) -> Path | None:
    cfg = read_yaml(config_path)
    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})
    optuna_cfg = cfg.get("optuna", {})
    preprocess_cfg = cfg.get("preprocessing", {})
    output_cfg = cfg.get("output", {})

    seed = int(exp_cfg.get("random_seed", 42))
    _set_global_seed(seed)

    timeline_dirs = _resolve_timeline_dirs(data_cfg)
    scenarios_cfg = data_cfg.get("scenarios", {})
    genotype_representation = str(exp_cfg.get("genotype_representation", "grm_pca"))
    n_anchor_bins = int(exp_cfg.get("n_anchor_bins", 20))
    best_anchor_csv = Path(str(exp_cfg["best_anchor_csv"]))

    best_anchor_df = load_best_anchor_table(best_anchor_csv)
    output_dir = None if dry_run else _resolve_output_dir(output_cfg)
    progress_enabled = False if dry_run else bool(output_cfg.get("progress_log", True))
    _append_progress(output_dir, enabled=progress_enabled, event="run_start", payload={"config_path": str(config_path)})

    timeline_bundles: dict[str, MultiTargetDataBundle] = {}
    timeline_anchors: dict[str, list[AnchorDefinition]] = {}
    timeline_available_vi_names: dict[str, dict[int, set[str]]] = {}
    available_vi_frames: list[pd.DataFrame] = []

    selected_anchor_pairs_by_timeline: dict[str, set[int]] = {}
    for row in best_anchor_df.to_dict("records"):
        selected_anchor_pairs_by_timeline.setdefault(str(row["timeline"]), set()).add(int(row["anchor_idx"]))

    for timeline_name, input_dir_str in timeline_dirs.items():
        bundle = load_multitarget_model_inputs(Path(str(input_dir_str)), genotype_representation=genotype_representation)
        anchors = build_anchor_definitions(bundle, n_anchor_bins=n_anchor_bins)
        anchor_indices = sorted(selected_anchor_pairs_by_timeline.get(timeline_name, set()))
        available_vi_names = collect_available_vi_names_by_anchor(bundle, anchors, anchor_indices=anchor_indices)
        timeline_bundles[timeline_name] = bundle
        timeline_anchors[timeline_name] = anchors
        timeline_available_vi_names[timeline_name] = available_vi_names
        available_vi_frames.append(summarize_available_vi_names(timeline_name, available_vi_names))

    available_vi_df = pd.concat(available_vi_frames, ignore_index=True) if available_vi_frames else pd.DataFrame()
    tasks_df = build_result33bc_light_tasks_from_config(
        cfg,
        best_anchor_df,
        available_vi_names={
            (timeline_name, int(anchor_idx)): set(vi_names)
            for timeline_name, anchor_map in timeline_available_vi_names.items()
            for anchor_idx, vi_names in anchor_map.items()
        },
    )
    if dry_run:
        summary = {
            "config_path": str(config_path),
            "best_anchor_rows": int(len(best_anchor_df)),
            "task_count": int(len(tasks_df)),
            "available_vi_rows": int(len(available_vi_df)),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return None
    if tasks_df.empty:
        raise RuntimeError("3.3B/C light 未生成任何有效任务。")

    split_usage_rows: list[dict[str, Any]] = []
    prepared_split_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    unique_triplets = tasks_df[["scenario", "timeline", "target"]].drop_duplicates()
    for row in unique_triplets.to_dict("records"):
        scenario_name = str(row["scenario"])
        timeline_name = str(row["timeline"])
        target_col = str(row["target"])
        bundle = timeline_bundles[timeline_name]
        raw_split_groups = _resolve_split_groups_for_scenario(
            scenario_name=scenario_name,
            scenario_cfg=scenarios_cfg[scenario_name],
            meta_df=bundle.meta,
            random_seed=seed,
        )
        if target_col not in bundle.y_df.columns:
            info = {"scenario": scenario_name, "timeline": timeline_name, "target": target_col, "status": "missing_target_column"}
            split_usage_rows.append(info)
            prepared_split_cache[(scenario_name, timeline_name, target_col)] = info
            continue
        y_full = bundle.y_df[target_col].dropna().copy()
        available_ids = set(bundle.meta.index.intersection(y_full.index))
        prepared_groups, split_stats = _prepare_split_groups(raw_split_groups, available_ids)
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

    metrics_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    total_tasks = int(len(tasks_df))
    for task_idx, task in enumerate(tasks_df.to_dict("records"), start=1):
        scenario_name = str(task["scenario"])
        timeline_name = str(task["timeline"])
        target_col = str(task["target"])
        predictor = str(task["predictor"]).lower()
        anchor_idx = int(task["anchor_idx"])
        vi_name = str(task["vi_name"])
        modality_variant = str(task["modality_variant"])
        split_info = prepared_split_cache[(scenario_name, timeline_name, target_col)]
        if split_info.get("status") != "ok":
            continue

        bundle = timeline_bundles[timeline_name]
        anchors = timeline_anchors[timeline_name]
        anchor = anchors[anchor_idx]
        x, info = build_result33bc_light_feature_matrix(
            bundle=bundle,
            anchors=anchors,
            anchor_idx=anchor_idx,
            vi_name=vi_name,
            use_genotype=(modality_variant == "GH_SINGLE_VI"),
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
                "modality_variant": modality_variant,
                "anchor_idx": anchor_idx,
                "vi_name": vi_name,
                "n_features": int(x.shape[1]),
            },
        )

        task_metrics, task_oof, task_trials = _run_single_task(
            x=x,
            y_full=split_info["y_full"],
            meta_full=bundle.meta,
            prepared_groups=split_info["prepared_groups"],
            scenario_name=scenario_name,
            timeline_name=timeline_name,
            target_col=target_col,
            predictor=predictor,
            modality_variant=modality_variant,
            anchor_idx=anchor_idx,
            anchor_tb=int(anchor.anchor_tb),
            anchor_phase=anchor.anchor_phase,
            anchor_band=str(task["anchor_band"]),
            vi_name=vi_name,
            vi_names=[vi_name],
            n_features_h=int(info.get("n_features_h", 0)),
            n_features_g=int(info.get("n_features_g", 0)),
            n_features_total=int(info.get("n_features_total", x.shape[1])),
            seed=seed,
            optuna_cfg=optuna_cfg,
            preprocess_cfg=preprocess_cfg,
            model_cfg=exp_cfg.get("model_backends", {}),
            ablation_group="FULL",
        )
        metrics_rows.extend(task_metrics)
        oof_rows.extend(task_oof)
        trial_rows.extend(task_trials)
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
                "modality_variant": modality_variant,
                "anchor_idx": anchor_idx,
                "vi_name": vi_name,
                "n_outer_results": int(len(task_metrics)),
            },
        )

    split_usage_df = _to_records_df(split_usage_rows)
    metrics_fold_df = _to_records_df(metrics_rows)
    oof_df = _to_records_df(oof_rows)
    trial_df = _to_records_df(trial_rows)
    metrics_summary_df = _build_metrics_summary(metrics_fold_df)
    metrics_by_year_df = _build_metrics_by_year(oof_df)

    best_anchor_df.to_csv(output_dir / "best_anchor_input.csv", index=False)
    available_vi_df.to_csv(output_dir / "available_vi_by_anchor.csv", index=False)
    tasks_df.to_csv(output_dir / "task_table.csv", index=False)
    split_usage_df.to_csv(output_dir / "split_usage_summary.csv", index=False)
    metrics_fold_df.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    metrics_summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    metrics_by_year_df.to_csv(output_dir / "metrics_by_year.csv", index=False)
    if not oof_df.empty:
        oof_df.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    if not trial_df.empty:
        trial_df.to_csv(output_dir / "optuna_trials.csv", index=False)

    config_snapshot = {
        "config_path": str(config_path),
        "generated_at": now_iso(),
        "git_commit": get_git_commit_hash(Path.cwd()),
        "data": data_cfg,
        "experiment": exp_cfg,
        "optuna": optuna_cfg,
        "preprocessing": preprocess_cfg,
        "output_dir": output_dir.as_posix(),
    }
    write_yaml(config_snapshot, output_dir / "run_config.yaml")
    write_json(config_snapshot, output_dir / "run_config.json")
    _write_light_summary(
        output_dir=output_dir,
        cfg=config_snapshot,
        best_anchor_df=best_anchor_df,
        available_vi_df=available_vi_df,
        tasks_df=tasks_df,
        split_usage_df=split_usage_df,
        metrics_summary_df=metrics_summary_df,
    )
    _append_progress(output_dir, enabled=progress_enabled, event="run_end", payload={"output_dir": output_dir.as_posix()})
    return output_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Light runner for Result 3.3B/C best-anchor single-VI experiments.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_result33bc_light(Path(args.config), dry_run=bool(args.dry_run))
