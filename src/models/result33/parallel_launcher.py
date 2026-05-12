from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from models.common.io_utils import ensure_dir, now_iso, read_yaml, write_json, write_yaml
from models.result33.runner import (
    _build_index_importance,
    _build_g_conditioned_single_vi_summary,
    _build_single_vi_transferability_summary,
    _resolve_active_predictors,
    _resolve_modality_variants,
    _resolve_output_dir,
    _resolve_scenarios_cfg,
    _resolve_timeline_dirs,
    _build_top_positive_anchor_summary,
    _to_records_df,
    _write_summary,
    _build_best_anchor_summary,
    _build_factor_group_importance,
    _build_metrics_by_year,
    _build_metrics_summary,
    enumerate_result33_tasks_from_config,
)


def _append_launcher_progress(out_dir: Path, event: str, payload: dict[str, Any]) -> None:
    path = out_dir / "progress.jsonl"
    record = {"ts": now_iso(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _task_weight(task: dict[str, Any]) -> float:
    predictor_weight = {
        "ridge": 1.0,
        "lasso": 1.05,
        "elasticnet": 1.1,
        "random_forest": 2.2,
        "lightgbm": 2.8,
    }
    scenario_weight = {
        "reference": 1.0,
        "within_season": 1.0,
        "loso": 1.15,
        "loso_genotype": 1.25,
    }
    variant_weight = {
        "G": 0.5,
        "GH_FULL": 1.0,
        "H_SINGLE": 0.8,
        "GH_SINGLE": 1.0,
        "H_SINGLE_VI": 0.7,
        "GH_SINGLE_VI": 0.8,
    }
    base = (
        predictor_weight.get(str(task["predictor"]).lower(), 1.5)
        * scenario_weight.get(str(task["scenario"]), 1.0)
        * variant_weight.get(str(task["modality_variant"]), 1.0)
    )
    if task.get("anchor_idx") is not None:
        base *= 1.0
    return base


def _task_unit_key(task: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(task["scenario"]),
        str(task["timeline"]),
        str(task["target"]),
        str(task["predictor"]).lower(),
    )


def group_tasks_by_unit(tasks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for task in tasks:
        grouped.setdefault(_task_unit_key(task), []).append(task)

    out = []
    for key in sorted(grouped.keys()):
        unit_tasks = grouped[key]
        unit_tasks = sorted(
            unit_tasks,
            key=lambda x: (
                {"G": 0, "GH_FULL": 1, "H_SINGLE": 2, "GH_SINGLE": 3, "H_SINGLE_VI": 4, "GH_SINGLE_VI": 5}.get(str(x["modality_variant"]), 9),
                -1 if x.get("anchor_idx") is None else int(x.get("anchor_idx")),
                str(x.get("vi_name") or ""),
            ),
        )
        out.append(unit_tasks)
    return out


def shard_task_groups(task_groups: list[list[dict[str, Any]]], n_workers: int) -> list[list[dict[str, Any]]]:
    if n_workers <= 0:
        raise ValueError("n_workers 必须大于 0。")
    if not task_groups:
        return []

    shard_count = min(int(n_workers), len(task_groups))
    shards: list[list[list[dict[str, Any]]]] = [[] for _ in range(shard_count)]
    shard_weights = [0.0 for _ in range(shard_count)]

    weighted_groups = []
    for group in task_groups:
        weight = sum(_task_weight(task) for task in group)
        weighted_groups.append((weight, group))

    for weight, group in sorted(weighted_groups, key=lambda x: x[0], reverse=True):
        idx = min(range(shard_count), key=lambda i: (shard_weights[i], len(shards[i])))
        shards[idx].append(group)
        shard_weights[idx] += weight

    out: list[list[dict[str, Any]]] = []
    for shard in shards:
        flat = [task for group in shard for task in group]
        if flat:
            out.append(flat)
    return out


def build_worker_config(
    base_cfg: dict[str, Any],
    *,
    worker_idx: int,
    worker_tasks: list[dict[str, Any]],
    worker_output_dir: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    data_cfg = cfg.setdefault("data", {})
    exp_cfg = cfg.setdefault("experiment", {})
    output_cfg = cfg.setdefault("output", {})

    used_scenarios = list(dict.fromkeys(task["scenario"] for task in worker_tasks))
    used_timelines = list(dict.fromkeys(task["timeline"] for task in worker_tasks))
    used_targets = list(dict.fromkeys(task["target"] for task in worker_tasks))
    used_predictors = list(dict.fromkeys(str(task["predictor"]).lower() for task in worker_tasks))
    used_variants = list(dict.fromkeys(task["modality_variant"] for task in worker_tasks))
    used_vi_names = list(
        dict.fromkeys(
            str(task["vi_name"])
            for task in worker_tasks
            if task.get("vi_name") not in (None, "", "null")
        )
    )
    used_anchor_indices = sorted(
        {
            int(task["anchor_idx"])
            for task in worker_tasks
            if task.get("anchor_idx") not in (None, "", "null")
        }
    )

    scenario_cfg = _resolve_scenarios_cfg(data_cfg)
    timeline_cfg = _resolve_timeline_dirs(data_cfg)
    data_cfg["scenarios"] = {key: scenario_cfg[key] for key in used_scenarios if key in scenario_cfg}
    data_cfg["timeline_dirs"] = {key: timeline_cfg[key] for key in used_timelines if key in timeline_cfg}

    exp_cfg["targets"] = used_targets
    exp_cfg["predictors_run"] = used_predictors
    exp_cfg["modality_variants"] = used_variants
    exp_cfg["anchor_indices"] = used_anchor_indices
    if used_vi_names:
        exp_cfg["vi_names"] = used_vi_names
    exp_cfg["task_filters"] = {
        "task_specs": worker_tasks,
        "scenario_order": list(_resolve_scenarios_cfg(base_cfg.get("data", {})).keys()),
        "timeline_order": list(_resolve_timeline_dirs(base_cfg.get("data", {})).keys()),
        "target_order": [str(x) for x in base_cfg.get("experiment", {}).get("targets", used_targets)],
        "predictor_order": _resolve_active_predictors(base_cfg.get("experiment", {})),
    }
    exp_cfg["parallel_worker_index"] = int(worker_idx)

    output_cfg["output_dir_base"] = worker_output_dir.as_posix()
    output_cfg["append_timestamp"] = False
    output_cfg["allow_overwrite"] = True
    output_cfg["latest_pointer_file"] = None
    output_cfg["progress_log"] = True

    return cfg


def _worker_log_path(log_dir: Path, worker_idx: int) -> Path:
    return log_dir / f"worker_{worker_idx:02d}.log"


def _worker_output_path(base_output_dir: Path, worker_idx: int) -> Path:
    return base_output_dir / "workers" / f"worker_{worker_idx:02d}"


def _worker_config_path(base_output_dir: Path, worker_idx: int) -> Path:
    return base_output_dir / "worker_configs" / f"worker_{worker_idx:02d}.yaml"


def _launch_workers(*, repo_root: Path, base_output_dir: Path, worker_cfg_paths: list[Path]) -> list[dict[str, Any]]:
    log_dir = ensure_dir(base_output_dir / "logs")
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    env.setdefault("PYTHONUNBUFFERED", "1")

    procs: list[dict[str, Any]] = []
    for worker_idx, cfg_path in enumerate(worker_cfg_paths):
        log_path = _worker_log_path(log_dir, worker_idx)
        log_fh = log_path.open("w", encoding="utf-8", buffering=1)
        cmd = [
            sys.executable,
            "scripts/run_result3_3_single_anchor_multitrait.py",
            "--config",
            str(cfg_path),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        _append_launcher_progress(
            base_output_dir,
            "worker_start",
            {"worker_idx": worker_idx, "pid": proc.pid, "config_path": str(cfg_path), "log_path": str(log_path)},
        )
        procs.append(
            {
                "worker_idx": worker_idx,
                "process": proc,
                "log_fh": log_fh,
                "log_path": log_path,
                "config_path": cfg_path,
            }
        )

    results = []
    try:
        remaining = procs[:]
        while remaining:
            still_running = []
            for item in remaining:
                proc = item["process"]
                code = proc.poll()
                if code is None:
                    still_running.append(item)
                    continue

                item["log_fh"].close()
                result = {
                    "worker_idx": item["worker_idx"],
                    "returncode": int(code),
                    "log_path": str(item["log_path"]),
                    "config_path": str(item["config_path"]),
                }
                results.append(result)
                _append_launcher_progress(base_output_dir, "worker_end", result)
            remaining = still_running
            if remaining:
                time.sleep(2.0)
    except KeyboardInterrupt:
        for item in procs:
            proc = item["process"]
            if proc.poll() is None:
                proc.terminate()
        for item in procs:
            item["log_fh"].close()
        raise

    return sorted(results, key=lambda x: x["worker_idx"])


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def merge_worker_outputs(
    *,
    base_cfg: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    worker_output_dirs: list[Path],
    worker_plans: list[dict[str, Any]],
    n_workers: int,
) -> None:
    anchor_frames = []
    feature_overview_frames = []
    split_usage_frames = []
    metrics_fold_frames = []
    metrics_by_year_frames = []
    delta_frames = []
    best_anchor_frames = []
    factor_importance_frames = []
    oof_frames = []
    trial_frames = []
    factor_fold_frames = []
    factor_summary_frames = []
    factor_oof_frames = []
    factor_trial_frames = []
    top_positive_anchor_frames = []
    index_fold_frames = []
    index_summary_frames = []
    index_oof_frames = []
    index_trial_frames = []
    index_importance_frames = []
    single_vi_transferability_frames = []
    g_conditioned_single_vi_frames = []

    for worker_dir in worker_output_dirs:
        anchor_frames.append(_read_csv(worker_dir / "anchor_bins.csv"))
        feature_overview_frames.append(_read_csv(worker_dir / "feature_overview.csv"))
        split_usage_frames.append(_read_csv(worker_dir / "split_usage_summary.csv"))
        metrics_fold_frames.append(_read_csv(worker_dir / "metrics_by_fold.csv"))
        metrics_by_year_frames.append(_read_csv(worker_dir / "metrics_by_year.csv"))
        delta_frames.append(_read_csv(worker_dir / "single_anchor_delta.csv"))
        best_anchor_frames.append(_read_csv(worker_dir / "best_anchor_by_trait_scenario.csv"))
        factor_importance_frames.append(_read_csv(worker_dir / "factor_group_importance.csv"))
        oof_frames.append(_read_parquet(worker_dir / "oof_predictions.parquet"))
        trial_frames.append(_read_csv(worker_dir / "optuna_trials.csv"))
        factor_fold_frames.append(_read_csv(worker_dir / "factor_group_metrics_by_fold.csv"))
        factor_summary_frames.append(_read_csv(worker_dir / "factor_group_metrics_summary.csv"))
        factor_oof_frames.append(_read_parquet(worker_dir / "factor_group_oof_predictions.parquet"))
        factor_trial_frames.append(_read_csv(worker_dir / "factor_group_optuna_trials.csv"))
        top_positive_anchor_frames.append(_read_csv(worker_dir / "top_positive_anchors.csv"))
        index_fold_frames.append(_read_csv(worker_dir / "index_ablation_metrics_by_fold.csv"))
        index_summary_frames.append(_read_csv(worker_dir / "index_ablation_metrics_summary.csv"))
        index_oof_frames.append(_read_parquet(worker_dir / "index_ablation_oof_predictions.parquet"))
        index_trial_frames.append(_read_csv(worker_dir / "index_ablation_optuna_trials.csv"))
        index_importance_frames.append(_read_csv(worker_dir / "index_importance.csv"))
        single_vi_transferability_frames.append(_read_csv(worker_dir / "single_vi_transferability.csv"))
        g_conditioned_single_vi_frames.append(_read_csv(worker_dir / "g_conditioned_single_vi_increment.csv"))

    anchor_df = pd.concat(anchor_frames, ignore_index=True) if anchor_frames else pd.DataFrame()
    if not anchor_df.empty:
        anchor_df = anchor_df.drop_duplicates(subset=["timeline", "anchor_idx"], keep="first").reset_index(drop=True)

    feature_overview_df = pd.concat(feature_overview_frames, ignore_index=True) if feature_overview_frames else pd.DataFrame()
    if not feature_overview_df.empty:
        feature_overview_df = feature_overview_df.drop_duplicates(
            subset=["timeline", "modality_variant", "anchor_idx"],
            keep="first",
        ).reset_index(drop=True)

    split_usage_df = pd.concat(split_usage_frames, ignore_index=True) if split_usage_frames else pd.DataFrame()
    if not split_usage_df.empty:
        split_usage_df = split_usage_df.drop_duplicates(subset=["scenario", "timeline", "target"], keep="first").reset_index(drop=True)

    metrics_fold_df = pd.concat(metrics_fold_frames, ignore_index=True) if metrics_fold_frames else pd.DataFrame()
    if not metrics_fold_df.empty:
        metrics_fold_df = metrics_fold_df.drop_duplicates(
            subset=[
                "scenario",
                "target",
                "timeline",
                "modality_variant",
                "predictor",
                "anchor_idx",
                "vi_name",
                "vi_names",
                "ablation_group",
                "target_year",
                "outer_fold",
            ],
            keep="first",
        ).reset_index(drop=True)

    oof_df = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    if not oof_df.empty:
        oof_df = oof_df.drop_duplicates(
            subset=[
                "plot_id",
                "scenario",
                "target",
                "timeline",
                "modality_variant",
                "predictor",
                "anchor_idx",
                "vi_name",
                "vi_names",
                "ablation_group",
                "target_year",
                "outer_fold",
            ],
            keep="first",
        ).reset_index(drop=True)

    trial_df = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
    if not trial_df.empty:
        trial_df = trial_df.drop_duplicates(
            subset=[
                "scenario",
                "target",
                "timeline",
                "modality_variant",
                "predictor",
                "anchor_idx",
                "vi_name",
                "vi_names",
                "ablation_group",
                "target_year",
                "outer_fold",
                "trial_number",
            ],
            keep="first",
        ).reset_index(drop=True)

    metrics_summary_df = _build_metrics_summary(metrics_fold_df)
    metrics_by_year_df = _build_metrics_by_year(oof_df)
    single_anchor_delta_df, best_anchor_df = _build_best_anchor_summary(metrics_summary_df)
    single_vi_transferability_df = pd.concat(single_vi_transferability_frames, ignore_index=True) if single_vi_transferability_frames else pd.DataFrame()
    if single_vi_transferability_df.empty and not metrics_summary_df.empty:
        single_vi_transferability_df = _build_single_vi_transferability_summary(metrics_summary_df)
    elif not single_vi_transferability_df.empty:
        single_vi_transferability_df = single_vi_transferability_df.drop_duplicates().reset_index(drop=True)
    g_conditioned_single_vi_df = pd.concat(g_conditioned_single_vi_frames, ignore_index=True) if g_conditioned_single_vi_frames else pd.DataFrame()
    if g_conditioned_single_vi_df.empty and not metrics_summary_df.empty:
        g_conditioned_single_vi_df = _build_g_conditioned_single_vi_summary(metrics_summary_df)
    elif not g_conditioned_single_vi_df.empty:
        g_conditioned_single_vi_df = g_conditioned_single_vi_df.drop_duplicates().reset_index(drop=True)
    top_positive_anchor_df = pd.concat(top_positive_anchor_frames, ignore_index=True) if top_positive_anchor_frames else pd.DataFrame()
    if top_positive_anchor_df.empty and not metrics_summary_df.empty:
        top_positive_anchor_df = _build_top_positive_anchor_summary(metrics_summary_df, top_k=int(base_cfg.get("experiment", {}).get("index_ablation", {}).get("top_k", 5)))
    elif not top_positive_anchor_df.empty:
        top_positive_anchor_df = top_positive_anchor_df.drop_duplicates().reset_index(drop=True)

    factor_group_metrics_by_fold_df = pd.concat(factor_fold_frames, ignore_index=True) if factor_fold_frames else pd.DataFrame()
    if not factor_group_metrics_by_fold_df.empty:
        factor_group_metrics_by_fold_df = factor_group_metrics_by_fold_df.drop_duplicates().reset_index(drop=True)
    factor_group_metrics_summary_df = pd.concat(factor_summary_frames, ignore_index=True) if factor_summary_frames else pd.DataFrame()
    if factor_group_metrics_summary_df.empty and not factor_group_metrics_by_fold_df.empty:
        factor_group_metrics_summary_df = _build_metrics_summary(factor_group_metrics_by_fold_df)
    factor_group_oof_df = pd.concat(factor_oof_frames, ignore_index=True) if factor_oof_frames else pd.DataFrame()
    if not factor_group_oof_df.empty:
        factor_group_oof_df = factor_group_oof_df.drop_duplicates().reset_index(drop=True)
    factor_group_trial_df = pd.concat(factor_trial_frames, ignore_index=True) if factor_trial_frames else pd.DataFrame()
    if not factor_group_trial_df.empty:
        factor_group_trial_df = factor_group_trial_df.drop_duplicates().reset_index(drop=True)
    factor_group_importance_df = pd.concat(factor_importance_frames, ignore_index=True) if factor_importance_frames else pd.DataFrame()
    if factor_group_importance_df.empty and not factor_group_metrics_summary_df.empty and not best_anchor_df.empty:
        factor_group_importance_df = _build_factor_group_importance(factor_group_metrics_summary_df, best_anchor_df)

    index_metrics_by_fold_df = pd.concat(index_fold_frames, ignore_index=True) if index_fold_frames else pd.DataFrame()
    if not index_metrics_by_fold_df.empty:
        index_metrics_by_fold_df = index_metrics_by_fold_df.drop_duplicates().reset_index(drop=True)
    index_metrics_summary_df = pd.concat(index_summary_frames, ignore_index=True) if index_summary_frames else pd.DataFrame()
    if index_metrics_summary_df.empty and not index_metrics_by_fold_df.empty:
        index_metrics_summary_df = _build_metrics_summary(index_metrics_by_fold_df)
    index_oof_df = pd.concat(index_oof_frames, ignore_index=True) if index_oof_frames else pd.DataFrame()
    if not index_oof_df.empty:
        index_oof_df = index_oof_df.drop_duplicates().reset_index(drop=True)
    index_trial_df = pd.concat(index_trial_frames, ignore_index=True) if index_trial_frames else pd.DataFrame()
    if not index_trial_df.empty:
        index_trial_df = index_trial_df.drop_duplicates().reset_index(drop=True)
    index_importance_df = pd.concat(index_importance_frames, ignore_index=True) if index_importance_frames else pd.DataFrame()
    if index_importance_df.empty and not index_metrics_summary_df.empty and not top_positive_anchor_df.empty:
        index_importance_df = _build_index_importance(index_metrics_summary_df, top_positive_anchor_df)

    final_cfg = copy.deepcopy(base_cfg)
    final_cfg["config_path"] = str(config_path)
    final_cfg.setdefault("experiment", {})
    final_cfg["experiment"]["active_predictors"] = sorted(
        {str(task["predictor"]).lower() for plan in worker_plans for task in plan.get("tasks", [])}
    )
    final_cfg["experiment"]["predictors_run"] = final_cfg["experiment"]["active_predictors"]
    final_cfg["experiment"]["modality_variants"] = list(
        dict.fromkeys(str(task["modality_variant"]) for plan in worker_plans for task in plan.get("tasks", []))
    )
    final_cfg.setdefault("data", {})
    timeline_cfg = _resolve_timeline_dirs(base_cfg.get("data", {}))
    used_timelines = list(dict.fromkeys(str(task["timeline"]) for plan in worker_plans for task in plan.get("tasks", [])))
    final_cfg["data"]["timeline_dirs"] = {key: timeline_cfg[key] for key in used_timelines if key in timeline_cfg}
    final_cfg["launcher"] = {
        "n_workers": int(n_workers),
        "worker_count_actual": int(len(worker_output_dirs)),
        "worker_plans": worker_plans,
    }

    anchor_df.to_csv(output_dir / "anchor_bins.csv", index=False)
    feature_overview_df.to_csv(output_dir / "feature_overview.csv", index=False)
    split_usage_df.to_csv(output_dir / "split_usage_summary.csv", index=False)
    metrics_fold_df.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    metrics_summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    metrics_by_year_df.to_csv(output_dir / "metrics_by_year.csv", index=False)
    single_anchor_delta_df.to_csv(output_dir / "single_anchor_delta.csv", index=False)
    best_anchor_df.to_csv(output_dir / "best_anchor_by_trait_scenario.csv", index=False)
    if not top_positive_anchor_df.empty:
        top_positive_anchor_df.to_csv(output_dir / "top_positive_anchors.csv", index=False)
    if not single_vi_transferability_df.empty:
        single_vi_transferability_df.to_csv(output_dir / "single_vi_transferability.csv", index=False)
    if not g_conditioned_single_vi_df.empty:
        g_conditioned_single_vi_df.to_csv(output_dir / "g_conditioned_single_vi_increment.csv", index=False)
    if not factor_group_metrics_by_fold_df.empty:
        factor_group_metrics_by_fold_df.to_csv(output_dir / "factor_group_metrics_by_fold.csv", index=False)
    if not factor_group_metrics_summary_df.empty:
        factor_group_metrics_summary_df.to_csv(output_dir / "factor_group_metrics_summary.csv", index=False)
    if not factor_group_importance_df.empty:
        factor_group_importance_df.to_csv(output_dir / "factor_group_importance.csv", index=False)
    if not oof_df.empty:
        oof_df.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    if not trial_df.empty:
        trial_df.to_csv(output_dir / "optuna_trials.csv", index=False)
    if not factor_group_oof_df.empty:
        factor_group_oof_df.to_parquet(output_dir / "factor_group_oof_predictions.parquet", index=False)
    if not factor_group_trial_df.empty:
        factor_group_trial_df.to_csv(output_dir / "factor_group_optuna_trials.csv", index=False)
    if not index_metrics_by_fold_df.empty:
        index_metrics_by_fold_df.to_csv(output_dir / "index_ablation_metrics_by_fold.csv", index=False)
    if not index_metrics_summary_df.empty:
        index_metrics_summary_df.to_csv(output_dir / "index_ablation_metrics_summary.csv", index=False)
    if not index_oof_df.empty:
        index_oof_df.to_parquet(output_dir / "index_ablation_oof_predictions.parquet", index=False)
    if not index_trial_df.empty:
        index_trial_df.to_csv(output_dir / "index_ablation_optuna_trials.csv", index=False)
    if not index_importance_df.empty:
        index_importance_df.to_csv(output_dir / "index_importance.csv", index=False)

    config_snapshot = {
        "config_path": str(config_path),
        "generated_at": now_iso(),
        "data": final_cfg.get("data", {}),
        "experiment": final_cfg.get("experiment", {}),
        "optuna": final_cfg.get("optuna", {}),
        "preprocessing": final_cfg.get("preprocessing", {}),
        "output_dir": output_dir.as_posix(),
        "launcher": final_cfg.get("launcher", {}),
    }
    write_yaml(config_snapshot, output_dir / "run_config.yaml")
    write_json(config_snapshot, output_dir / "run_config.json")
    _write_summary(
        output_dir=output_dir,
        cfg=config_snapshot,
        anchor_df=anchor_df,
        feature_overview_df=feature_overview_df,
        split_usage_df=split_usage_df,
        metrics_summary_df=metrics_summary_df,
        delta_df=single_anchor_delta_df,
        best_anchor_df=best_anchor_df,
        factor_importance_df=factor_group_importance_df,
        top_positive_anchor_df=top_positive_anchor_df,
        index_importance_df=index_importance_df,
    )


def run_result3_3_parallel_launcher(config_path: Path, *, n_workers: int) -> Path:
    cfg = read_yaml(config_path)
    output_dir = _resolve_output_dir(cfg.get("output", {}))
    ensure_dir(output_dir / "worker_configs")
    ensure_dir(output_dir / "workers")
    ensure_dir(output_dir / "logs")
    _append_launcher_progress(output_dir, "launcher_start", {"config_path": str(config_path), "n_workers": int(n_workers)})

    tasks = enumerate_result33_tasks_from_config(cfg)
    if not tasks:
        raise RuntimeError("launcher 未枚举到任何任务。")

    task_groups = group_tasks_by_unit(tasks)
    shards = shard_task_groups(task_groups, n_workers=n_workers)
    worker_cfg_paths = []
    worker_plans = []
    worker_output_dirs = []

    for worker_idx, worker_tasks in enumerate(shards):
        worker_output_dir = _worker_output_path(output_dir, worker_idx)
        worker_cfg = build_worker_config(
            cfg,
            worker_idx=worker_idx,
            worker_tasks=worker_tasks,
            worker_output_dir=worker_output_dir,
        )
        worker_cfg_path = _worker_config_path(output_dir, worker_idx)
        write_yaml(worker_cfg, worker_cfg_path)
        worker_cfg_paths.append(worker_cfg_path)
        worker_output_dirs.append(worker_output_dir)
        worker_plans.append(
            {
                "worker_idx": worker_idx,
                "task_count": len(worker_tasks),
                "unit_count": len({_task_unit_key(task) for task in worker_tasks}),
                "tasks": worker_tasks,
                "weight_sum": sum(_task_weight(task) for task in worker_tasks),
                "config_path": worker_cfg_path.as_posix(),
                "output_dir": worker_output_dir.as_posix(),
            }
        )

    write_json({"tasks_total": len(tasks), "task_units_total": len(task_groups), "workers": worker_plans}, output_dir / "worker_plan.json")
    _append_launcher_progress(
        output_dir,
        "worker_plan_ready",
        {"tasks_total": len(tasks), "task_units_total": len(task_groups), "worker_count": len(worker_plans)},
    )

    results = _launch_workers(repo_root=Path.cwd(), base_output_dir=output_dir, worker_cfg_paths=worker_cfg_paths)
    write_json({"results": results}, output_dir / "worker_results.json")

    failed = [item for item in results if int(item["returncode"]) != 0]
    if failed:
        raise RuntimeError(
            "存在 worker 失败，请查看日志：" + "; ".join(f"worker={item['worker_idx']} log={item['log_path']}" for item in failed)
        )

    merge_worker_outputs(
        base_cfg=cfg,
        config_path=config_path,
        output_dir=output_dir,
        worker_output_dirs=worker_output_dirs,
        worker_plans=worker_plans,
        n_workers=n_workers,
    )
    _append_launcher_progress(output_dir, "launcher_end", {"output_dir": output_dir.as_posix()})
    print(f"Result3.3 parallel launcher completed: {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel launcher for Result3.3 single-anchor experiments.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-workers", type=int, required=True)
    args = parser.parse_args()
    run_result3_3_parallel_launcher(Path(args.config), n_workers=int(args.n_workers))


if __name__ == "__main__":
    main()
