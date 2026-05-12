from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from models.common.io_utils import ensure_dir, now_iso, read_yaml, write_json, write_yaml
from models.result31.parallel_launcher import shard_tasks
from models.result31.runner import _resolve_output_dir, _resolve_scenarios_cfg, _resolve_timeline_dirs
from models.result34.runner import (
    _resolve_input_variants,
    enumerate_result34_tasks_from_config,
    finalize_result34_outputs,
)


def _append_launcher_progress(out_dir: Path, event: str, payload: dict[str, Any]) -> None:
    path = out_dir / "progress.jsonl"
    record = {"ts": now_iso(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _task_weight(task: dict[str, str]) -> float:
    predictor_weight = {
        "ridge": 1.0,
        "lasso": 1.05,
        "elasticnet": 1.1,
        "random_forest": 2.2,
        "lightgbm": 2.8,
    }
    scenario_weight = {
        "reference": 1.0,
        "within_season": 1.05,
        "loso": 1.15,
        "loso_genotype": 1.2,
    }
    input_variant_weight = {
        "G": 0.9,
        "H_FULL": 1.0,
        "G+H_FULL": 1.1,
        "H_ANCHOR_LOCAL": 1.4,
        "G+H_ANCHOR_LOCAL": 1.6,
        "H_ANCHOR_VI": 1.1,
        "G+H_ANCHOR_VI": 1.3,
        "G+FULLH": 3.2,
    }
    prefix_weight = 0.9 + 0.02 * float(task.get("anchor_order", 20) or 20)
    return (
        predictor_weight.get(str(task["predictor"]).lower(), 1.5)
        * scenario_weight.get(str(task["scenario"]), 1.0)
        * input_variant_weight.get(str(task["input_variant"]).upper(), 1.0)
        * prefix_weight
    )


def _resolve_anchor_order_order(base_cfg: dict[str, Any], worker_tasks: list[dict[str, str]]) -> list[int]:
    exp_cfg = base_cfg.get("experiment", {})
    prefix_cfg = exp_cfg.get("growth_prefix", exp_cfg.get("prefix_mode", {}))
    if isinstance(prefix_cfg, dict) and prefix_cfg.get("anchor_orders") not in (None, "", []):
        return [int(x) for x in prefix_cfg.get("anchor_orders", [])]
    worker_orders = sorted({int(task["anchor_order"]) for task in worker_tasks if task.get("anchor_order") is not None})
    return worker_orders


def build_worker_config(
    base_cfg: dict[str, Any],
    *,
    worker_idx: int,
    worker_tasks: list[dict[str, str]],
    worker_output_dir: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    data_cfg = cfg.setdefault("data", {})
    exp_cfg = cfg.setdefault("experiment", {})
    output_cfg = cfg.setdefault("output", {})

    used_scenarios = list(dict.fromkeys(task["scenario"] for task in worker_tasks))
    used_timelines = list(dict.fromkeys(task["timeline"] for task in worker_tasks))
    used_targets = list(dict.fromkeys(task["target"] for task in worker_tasks))
    used_input_variants = list(dict.fromkeys(str(task["input_variant"]).upper() for task in worker_tasks))
    used_predictors = list(dict.fromkeys(str(task["predictor"]).lower() for task in worker_tasks))

    scenario_cfg = _resolve_scenarios_cfg(data_cfg)
    timeline_cfg = _resolve_timeline_dirs(data_cfg)
    data_cfg["scenarios"] = {key: scenario_cfg[key] for key in used_scenarios if key in scenario_cfg}
    data_cfg["timeline_dirs"] = {key: timeline_cfg[key] for key in used_timelines if key in timeline_cfg}

    exp_cfg["targets"] = used_targets
    exp_cfg["input_variants"] = used_input_variants
    exp_cfg["predictors_run"] = used_predictors
    exp_cfg["task_filters"] = {
        "task_specs": worker_tasks,
        "scenario_order": list(_resolve_scenarios_cfg(base_cfg.get("data", {})).keys()),
        "timeline_order": list(_resolve_timeline_dirs(base_cfg.get("data", {})).keys()),
        "target_order": [str(x) for x in base_cfg.get("experiment", {}).get("targets", used_targets)],
        "input_variant_order": _resolve_input_variants(base_cfg.get("experiment", {})),
        "predictor_order": [str(x).lower() for x in base_cfg.get("experiment", {}).get("predictors_run", used_predictors)],
        "anchor_order_order": _resolve_anchor_order_order(base_cfg, worker_tasks),
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


def _launch_workers(
    *,
    repo_root: Path,
    base_output_dir: Path,
    worker_cfg_paths: list[Path],
    worker_script: str = "scripts/run_result3_4_phase_state.py",
) -> list[dict[str, Any]]:
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
            worker_script,
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


def _dedup_subset(df: pd.DataFrame, cols: list[str]) -> list[str]:
    prefix_cols = ["anchor_order", "anchor_idx", "anchor_tb", "anchor_phase", "anchor_token"]
    return [col for col in [*cols, *prefix_cols] if col in df.columns]


def merge_worker_outputs(
    *,
    base_cfg: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    worker_output_dirs: list[Path],
    worker_plans: list[dict[str, Any]],
    n_workers: int,
) -> None:
    representation_overview_frames = []
    representation_feature_spec_frames = []
    feature_overview_frames = []
    split_usage_frames = []
    metrics_fold_frames = []
    oof_frames = []
    inner_pred_frames = []
    trial_frames = []

    for worker_dir in worker_output_dirs:
        representation_overview_frames.append(_read_csv(worker_dir / "reduced_h_feature_overview.csv"))
        representation_feature_spec_frames.append(_read_csv(worker_dir / "reduced_h_feature_spec.csv"))
        feature_overview_frames.append(_read_csv(worker_dir / "feature_overview.csv"))
        split_usage_frames.append(_read_csv(worker_dir / "split_usage_summary.csv"))
        metrics_fold_frames.append(_read_csv(worker_dir / "metrics_by_fold.csv"))
        oof_frames.append(_read_parquet(worker_dir / "oof_predictions.parquet"))
        inner_pred_frames.append(_read_parquet(worker_dir / "outer_test_inner_ensemble_predictions.parquet"))
        trial_frames.append(_read_csv(worker_dir / "optuna_trials.csv"))

    representation_overview_df = (
        pd.concat(representation_overview_frames, ignore_index=True) if representation_overview_frames else pd.DataFrame()
    )
    if not representation_overview_df.empty:
        subset = [col for col in ["timeline", "representation"] if col in representation_overview_df.columns]
        if subset:
            representation_overview_df = representation_overview_df.drop_duplicates(subset=subset, keep="first")

    representation_feature_spec_df = (
        pd.concat(representation_feature_spec_frames, ignore_index=True) if representation_feature_spec_frames else pd.DataFrame()
    )
    if not representation_feature_spec_df.empty:
        subset = [col for col in ["timeline", "representation", "feature"] if col in representation_feature_spec_df.columns]
        if subset:
            representation_feature_spec_df = representation_feature_spec_df.drop_duplicates(subset=subset, keep="first")

    feature_overview_df = pd.concat(feature_overview_frames, ignore_index=True) if feature_overview_frames else pd.DataFrame()
    if not feature_overview_df.empty:
        subset = [col for col in ["timeline", "input_variant"] if col in feature_overview_df.columns]
        if subset:
            feature_overview_df = feature_overview_df.drop_duplicates(subset=subset, keep="first")

    split_usage_df = pd.concat(split_usage_frames, ignore_index=True) if split_usage_frames else pd.DataFrame()
    if not split_usage_df.empty:
        subset = [col for col in ["scenario", "timeline", "target"] if col in split_usage_df.columns]
        if subset:
            split_usage_df = split_usage_df.drop_duplicates(subset=subset, keep="first")

    metrics_fold_df = pd.concat(metrics_fold_frames, ignore_index=True) if metrics_fold_frames else pd.DataFrame()
    if not metrics_fold_df.empty:
        metrics_fold_df = metrics_fold_df.drop_duplicates(
            subset=_dedup_subset(
                metrics_fold_df,
                ["scenario", "timeline", "target", "input_variant", "predictor", "target_year", "outer_fold"],
            ),
            keep="first",
        )

    oof_df = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    if not oof_df.empty:
        oof_df = oof_df.drop_duplicates(
            subset=_dedup_subset(
                oof_df,
                ["plot_id", "scenario", "timeline", "target", "input_variant", "predictor", "target_year", "outer_fold"],
            ),
            keep="first",
        )

    inner_pred_df = pd.concat(inner_pred_frames, ignore_index=True) if inner_pred_frames else pd.DataFrame()
    if not inner_pred_df.empty:
        inner_pred_df = inner_pred_df.drop_duplicates(
            subset=_dedup_subset(
                inner_pred_df,
                [
                    "plot_id",
                    "scenario",
                    "timeline",
                    "target",
                    "input_variant",
                    "predictor",
                    "target_year",
                    "outer_fold",
                    "inner_fold",
                ],
            ),
            keep="first",
        )

    trial_df = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
    if not trial_df.empty:
        trial_df = trial_df.drop_duplicates(
            subset=_dedup_subset(
                trial_df,
                [
                    "scenario",
                    "timeline",
                    "target",
                    "input_variant",
                    "predictor",
                    "target_year",
                    "outer_fold",
                    "trial_number",
                ],
            ),
            keep="first",
        )

    final_cfg = copy.deepcopy(base_cfg)
    final_cfg["config_path"] = str(config_path)
    final_cfg.setdefault("experiment", {})
    final_cfg["experiment"]["active_predictors"] = list(
        dict.fromkeys(str(task["predictor"]).lower() for plan in worker_plans for task in plan.get("tasks", []))
    )
    final_cfg["experiment"]["predictors_run"] = final_cfg["experiment"]["active_predictors"]
    final_cfg["experiment"]["input_variants"] = list(
        dict.fromkeys(str(task["input_variant"]).upper() for plan in worker_plans for task in plan.get("tasks", []))
    )
    final_cfg["launcher"] = {
        "n_workers": int(n_workers),
        "worker_count_actual": int(len(worker_output_dirs)),
        "worker_plans": worker_plans,
    }

    finalize_result34_outputs(
        output_dir=output_dir,
        cfg=final_cfg,
        representation_overview_df=representation_overview_df,
        representation_feature_spec_df=representation_feature_spec_df,
        feature_overview_df=feature_overview_df,
        split_usage_df=split_usage_df,
        metrics_fold_df=metrics_fold_df,
        oof_df=oof_df,
        inner_pred_df=inner_pred_df,
        trial_df=trial_df,
    )


def run_result34_parallel_launcher(config_path: Path, *, n_workers: int) -> Path:
    cfg = read_yaml(config_path)
    output_dir = _resolve_output_dir(cfg.get("output", {}))
    ensure_dir(output_dir / "worker_configs")
    ensure_dir(output_dir / "workers")
    ensure_dir(output_dir / "logs")
    _append_launcher_progress(output_dir, "launcher_start", {"config_path": str(config_path), "n_workers": int(n_workers)})

    tasks = enumerate_result34_tasks_from_config(cfg)
    if not tasks:
        raise RuntimeError("launcher 未枚举到任何任务。")

    shards = shard_tasks(tasks, n_workers=n_workers)
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
                "tasks": worker_tasks,
                "weight_sum": sum(_task_weight(task) for task in worker_tasks),
                "config_path": worker_cfg_path.as_posix(),
                "output_dir": worker_output_dir.as_posix(),
            }
        )

    write_json({"tasks_total": len(tasks), "workers": worker_plans}, output_dir / "worker_plan.json")
    _append_launcher_progress(output_dir, "worker_plan_ready", {"tasks_total": len(tasks), "worker_count": len(worker_plans)})

    results = _launch_workers(repo_root=Path.cwd(), base_output_dir=output_dir, worker_cfg_paths=worker_cfg_paths)
    write_json({"results": results}, output_dir / "worker_results.json")

    failed = [item for item in results if int(item["returncode"]) != 0]
    if failed:
        raise RuntimeError(
            "存在 worker 失败，请查看日志："
            + "; ".join(f"worker={item['worker_idx']} log={item['log_path']}" for item in failed)
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
    print(f"Result3.4 parallel launcher completed: {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel launcher for Result3.4A reduced-H experiments.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-workers", type=int, required=True)
    args = parser.parse_args()
    run_result34_parallel_launcher(Path(args.config), n_workers=int(args.n_workers))


if __name__ == "__main__":
    main()
