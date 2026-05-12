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
from models.result32.runner import (
    _resolve_active_predictors,
    _resolve_modality_combos,
    _resolve_output_dir,
    _resolve_scenarios_cfg,
    _resolve_timeline_dirs,
    enumerate_result32_tasks_from_config,
    finalize_result3_2_outputs,
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
    timeline_weight = {
        "gdd_abs": 1.0,
        "gdd_rel_heading": 1.15,
        "day": 1.0,
        "day_rel_heading": 1.05,
    }
    modality_weight = {
        "G": 0.7,
        "C": 0.9,
        "H": 1.0,
        "C+G": 1.0,
        "H+G": 1.1,
        "H+C": 1.15,
        "H+C+G": 1.2,
    }
    scenario_weight = {
        "within_season": 1.0,
        "loso": 1.05,
        "loso_genotype": 1.1,
    }
    return (
        predictor_weight.get(str(task["predictor"]).lower(), 1.5)
        * timeline_weight.get(str(task.get("timeline", "")), 1.0)
        * modality_weight.get(str(task.get("modality_combo", "")), 1.0)
        * scenario_weight.get(str(task["scenario"]), 1.0)
    )


def shard_tasks(tasks: list[dict[str, str]], n_workers: int) -> list[list[dict[str, str]]]:
    if n_workers <= 0:
        raise ValueError("n_workers 必须大于 0。")
    if not tasks:
        return []

    shard_count = min(int(n_workers), len(tasks))
    shards: list[list[dict[str, str]]] = [[] for _ in range(shard_count)]
    shard_weights = [0.0 for _ in range(shard_count)]

    for task in sorted(tasks, key=_task_weight, reverse=True):
        idx = min(range(shard_count), key=lambda i: (shard_weights[i], len(shards[i])))
        shards[idx].append(task)
        shard_weights[idx] += _task_weight(task)

    return [shard for shard in shards if shard]


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
    used_combos = list(dict.fromkeys(task["modality_combo"] for task in worker_tasks))
    used_predictors = list(dict.fromkeys(str(task["predictor"]).lower() for task in worker_tasks))

    scenario_cfg = _resolve_scenarios_cfg(data_cfg)
    timeline_cfg = _resolve_timeline_dirs(data_cfg, exp_cfg)
    data_cfg["scenarios"] = {key: scenario_cfg[key] for key in used_scenarios if key in scenario_cfg}
    data_cfg["timeline_dirs"] = {key: timeline_cfg[key] for key in used_timelines if key in timeline_cfg}

    exp_cfg["targets"] = used_targets
    exp_cfg["modality_combos"] = used_combos
    exp_cfg["predictors_run"] = used_predictors
    exp_cfg["task_filters"] = {
        "task_specs": worker_tasks,
        "scenario_order": list(_resolve_scenarios_cfg(base_cfg.get("data", {})).keys()),
        "timeline_order": list(_resolve_timeline_dirs(base_cfg.get("data", {}), base_cfg.get("experiment", {})).keys()),
        "target_order": [str(x) for x in base_cfg.get("experiment", {}).get("targets", used_targets)],
        "modality_combo_order": _resolve_modality_combos(base_cfg.get("experiment", {})),
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
            "scripts/run_result3_2_multitimeline_multimodal_multitrait.py",
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


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def merge_worker_outputs(
    *,
    base_cfg: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    worker_output_dirs: list[Path],
    worker_plans: list[dict[str, Any]],
    n_workers: int,
) -> None:
    feature_overview_frames = []
    split_usage_frames = []
    metrics_fold_frames = []
    oof_frames = []
    inner_pred_frames = []
    trial_frames = []
    anchor_bins_list = []

    for worker_dir in worker_output_dirs:
        feature_overview_frames.append(_read_csv(worker_dir / "feature_overview.csv"))
        split_usage_frames.append(_read_csv(worker_dir / "split_usage_summary.csv"))
        metrics_fold_frames.append(_read_csv(worker_dir / "metrics_by_fold.csv"))
        oof_frames.append(_read_parquet(worker_dir / "oof_predictions.parquet"))
        inner_pred_frames.append(_read_parquet(worker_dir / "outer_test_inner_ensemble_predictions.parquet"))
        trial_frames.append(_read_csv(worker_dir / "optuna_trials.csv"))
        anchor_obj = _read_json(worker_dir / "anchor_bins.json")
        if isinstance(anchor_obj, list):
            anchor_bins_list.extend(anchor_obj)

    anchor_bins_obj = None
    if anchor_bins_list:
        seen = set()
        deduped = []
        for row in anchor_bins_list:
            key = (row.get("timeline"), row.get("anchor_idx"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        anchor_bins_obj = deduped

    feature_overview_df = pd.concat(feature_overview_frames, ignore_index=True) if feature_overview_frames else pd.DataFrame()
    if not feature_overview_df.empty:
        feature_overview_df = feature_overview_df.drop_duplicates(
            subset=["timeline", "anchor_idx", "modality_combo"],
            keep="first",
        )

    split_usage_df = pd.concat(split_usage_frames, ignore_index=True) if split_usage_frames else pd.DataFrame()
    if not split_usage_df.empty:
        split_usage_df = split_usage_df.drop_duplicates(subset=["scenario", "timeline", "target"], keep="first").reset_index(drop=True)

    metrics_fold_df = pd.concat(metrics_fold_frames, ignore_index=True) if metrics_fold_frames else pd.DataFrame()
    if not metrics_fold_df.empty:
        metrics_fold_df = metrics_fold_df.drop_duplicates(
            subset=["scenario", "timeline", "target", "modality_combo", "predictor", "anchor_idx", "target_year", "outer_fold"],
            keep="first",
        )

    oof_df = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    if not oof_df.empty:
        oof_df = oof_df.drop_duplicates(
            subset=["plot_id", "scenario", "timeline", "target", "modality_combo", "predictor", "anchor_idx", "target_year", "outer_fold"],
            keep="first",
        )

    inner_pred_df = pd.concat(inner_pred_frames, ignore_index=True) if inner_pred_frames else pd.DataFrame()
    if not inner_pred_df.empty:
        inner_pred_df = inner_pred_df.drop_duplicates(
            subset=["plot_id", "scenario", "timeline", "target", "modality_combo", "predictor", "anchor_idx", "target_year", "outer_fold", "inner_fold"],
            keep="first",
        )

    trial_df = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
    if not trial_df.empty:
        trial_df = trial_df.drop_duplicates(
            subset=["scenario", "timeline", "target", "modality_combo", "predictor", "anchor_idx", "target_year", "outer_fold", "trial_number"],
            keep="first",
        )

    final_cfg = copy.deepcopy(base_cfg)
    final_cfg["config_path"] = str(config_path)
    final_cfg.setdefault("experiment", {})
    final_cfg["experiment"]["active_predictors"] = sorted(
        {str(task["predictor"]).lower() for plan in worker_plans for task in plan.get("tasks", [])}
    )
    final_cfg["experiment"]["predictors_run"] = final_cfg["experiment"]["active_predictors"]
    final_cfg["experiment"]["modality_combos"] = list(
        dict.fromkeys(str(task["modality_combo"]) for plan in worker_plans for task in plan.get("tasks", []))
    )
    final_cfg.setdefault("data", {})
    timeline_cfg = _resolve_timeline_dirs(base_cfg.get("data", {}), base_cfg.get("experiment", {}))
    used_timelines = list(dict.fromkeys(str(task["timeline"]) for plan in worker_plans for task in plan.get("tasks", [])))
    final_cfg["data"]["timeline_dirs"] = {key: timeline_cfg[key] for key in used_timelines if key in timeline_cfg}
    final_cfg["launcher"] = {
        "n_workers": int(n_workers),
        "worker_count_actual": int(len(worker_output_dirs)),
        "worker_plans": worker_plans,
    }

    finalize_result3_2_outputs(
        output_dir=output_dir,
        cfg=final_cfg,
        anchor_bins_obj=anchor_bins_obj,
        feature_overview_df=feature_overview_df,
        split_usage_df=split_usage_df,
        metrics_fold_df=metrics_fold_df,
        oof_df=oof_df,
        inner_pred_df=inner_pred_df,
        trial_df=trial_df,
    )


def run_result3_2_parallel_launcher(config_path: Path, *, n_workers: int) -> Path:
    cfg = read_yaml(config_path)
    output_dir = _resolve_output_dir(cfg.get("output", {}))
    ensure_dir(output_dir / "worker_configs")
    ensure_dir(output_dir / "workers")
    ensure_dir(output_dir / "logs")
    _append_launcher_progress(output_dir, "launcher_start", {"config_path": str(config_path), "n_workers": int(n_workers)})

    tasks = enumerate_result32_tasks_from_config(cfg)
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
    print(f"Result3.2 parallel launcher completed: {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel launcher for Result3.2 task-level multiprocessing.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-workers", type=int, required=True)
    args = parser.parse_args()
    run_result3_2_parallel_launcher(Path(args.config), n_workers=int(args.n_workers))


if __name__ == "__main__":
    main()
