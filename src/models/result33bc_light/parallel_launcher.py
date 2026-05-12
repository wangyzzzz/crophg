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
from models.result33.parallel_launcher import _read_csv, _read_parquet
from models.result33bc_light.runner import (
    _write_light_summary,
    build_result33bc_light_tasks_from_config,
    collect_available_vi_names_by_anchor,
    load_best_anchor_table,
    summarize_available_vi_names,
)
from models.anchorwise.data_loader import load_multitarget_model_inputs
from models.anchorwise.feature_builder import build_anchor_definitions
from models.result31.runner import _resolve_split_groups_for_scenario
from models.result32.runner import _prepare_split_groups
from models.result33.runner import (
    _build_metrics_by_year,
    _build_metrics_summary,
    _resolve_active_predictors,
    _resolve_output_dir,
    _resolve_timeline_dirs,
)


def _append_launcher_progress(out_dir: Path, event: str, payload: dict[str, Any]) -> None:
    path = out_dir / "progress.jsonl"
    record = {"ts": now_iso(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _row_weight(row: dict[str, Any]) -> float:
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
    return predictor_weight.get(str(row["predictor"]).lower(), 1.5) * scenario_weight.get(str(row["scenario"]), 1.0)


def shard_best_anchor_rows(best_anchor_df: pd.DataFrame, n_workers: int) -> list[pd.DataFrame]:
    if n_workers <= 0:
        raise ValueError("n_workers 必须大于 0。")
    if best_anchor_df.empty:
        return []

    shard_count = min(int(n_workers), len(best_anchor_df))
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    shard_weights = [0.0 for _ in range(shard_count)]

    rows = best_anchor_df.to_dict("records")
    for row in sorted(rows, key=_row_weight, reverse=True):
        idx = min(range(shard_count), key=lambda i: (shard_weights[i], len(shards[i])))
        shards[idx].append(row)
        shard_weights[idx] += _row_weight(row)

    out = []
    for shard in shards:
        if shard:
            out.append(pd.DataFrame(shard))
    return out


def build_worker_config(
    base_cfg: dict[str, Any],
    *,
    worker_idx: int,
    worker_best_anchor_csv: Path,
    worker_rows: pd.DataFrame,
    worker_output_dir: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    data_cfg = cfg.setdefault("data", {})
    exp_cfg = cfg.setdefault("experiment", {})
    output_cfg = cfg.setdefault("output", {})

    used_scenarios = list(dict.fromkeys(worker_rows["scenario"].astype(str).tolist()))
    used_timelines = list(dict.fromkeys(worker_rows["timeline"].astype(str).tolist()))
    used_targets = list(dict.fromkeys(worker_rows["target"].astype(str).tolist()))
    used_predictors = list(dict.fromkeys(worker_rows["predictor"].astype(str).str.lower().tolist()))

    timeline_cfg = _resolve_timeline_dirs(data_cfg)
    data_cfg["timeline_dirs"] = {k: timeline_cfg[k] for k in used_timelines if k in timeline_cfg}
    scenario_cfg = data_cfg.get("scenarios", {})
    data_cfg["scenarios"] = {k: scenario_cfg[k] for k in used_scenarios if k in scenario_cfg}

    exp_cfg["best_anchor_csv"] = worker_best_anchor_csv.as_posix()
    exp_cfg["targets"] = used_targets
    exp_cfg["predictors_run"] = used_predictors
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


def _worker_best_anchor_path(base_output_dir: Path, worker_idx: int) -> Path:
    return base_output_dir / "worker_best_anchor" / f"worker_{worker_idx:02d}.csv"


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
            "scripts/run_result3_3BC_light.py",
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


def merge_worker_outputs(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    worker_output_dirs: list[Path],
    worker_plans: list[dict[str, Any]],
    n_workers: int,
) -> None:
    best_anchor_frames = []
    available_vi_frames = []
    task_frames = []
    split_usage_frames = []
    metrics_fold_frames = []
    metrics_by_year_frames = []
    oof_frames = []
    trial_frames = []

    for worker_dir in worker_output_dirs:
        best_anchor_frames.append(_read_csv(worker_dir / "best_anchor_input.csv"))
        available_vi_frames.append(_read_csv(worker_dir / "available_vi_by_anchor.csv"))
        task_frames.append(_read_csv(worker_dir / "task_table.csv"))
        split_usage_frames.append(_read_csv(worker_dir / "split_usage_summary.csv"))
        metrics_fold_frames.append(_read_csv(worker_dir / "metrics_by_fold.csv"))
        metrics_by_year_frames.append(_read_csv(worker_dir / "metrics_by_year.csv"))
        oof_frames.append(_read_parquet(worker_dir / "oof_predictions.parquet"))
        trial_frames.append(_read_csv(worker_dir / "optuna_trials.csv"))

    best_anchor_df = pd.concat(best_anchor_frames, ignore_index=True) if best_anchor_frames else pd.DataFrame()
    if not best_anchor_df.empty:
        best_anchor_df = best_anchor_df.drop_duplicates().reset_index(drop=True)
    available_vi_df = pd.concat(available_vi_frames, ignore_index=True) if available_vi_frames else pd.DataFrame()
    if not available_vi_df.empty:
        available_vi_df = available_vi_df.drop_duplicates().reset_index(drop=True)
    tasks_df = pd.concat(task_frames, ignore_index=True) if task_frames else pd.DataFrame()
    if not tasks_df.empty:
        tasks_df = tasks_df.drop_duplicates().reset_index(drop=True)
    split_usage_df = pd.concat(split_usage_frames, ignore_index=True) if split_usage_frames else pd.DataFrame()
    if not split_usage_df.empty:
        split_usage_df = split_usage_df.drop_duplicates(subset=["scenario", "timeline", "target"], keep="first").reset_index(drop=True)
    metrics_fold_df = pd.concat(metrics_fold_frames, ignore_index=True) if metrics_fold_frames else pd.DataFrame()
    if not metrics_fold_df.empty:
        metrics_fold_df = metrics_fold_df.drop_duplicates().reset_index(drop=True)
    oof_df = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    if not oof_df.empty:
        oof_df = oof_df.drop_duplicates().reset_index(drop=True)
    trial_df = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
    if not trial_df.empty:
        trial_df = trial_df.drop_duplicates().reset_index(drop=True)

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
        "data": cfg.get("data", {}),
        "experiment": cfg.get("experiment", {}),
        "optuna": cfg.get("optuna", {}),
        "preprocessing": cfg.get("preprocessing", {}),
        "output_dir": output_dir.as_posix(),
        "launcher": {
            "n_workers": int(n_workers),
            "worker_count_actual": int(len(worker_output_dirs)),
            "worker_plans": worker_plans,
        },
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


def run_result33bc_light_parallel_launcher(config_path: Path, *, n_workers: int) -> Path:
    cfg = read_yaml(config_path)
    output_dir = _resolve_output_dir(cfg.get("output", {}))
    ensure_dir(output_dir / "worker_configs")
    ensure_dir(output_dir / "workers")
    ensure_dir(output_dir / "logs")
    ensure_dir(output_dir / "worker_best_anchor")
    _append_launcher_progress(output_dir, "launcher_start", {"config_path": str(config_path), "n_workers": int(n_workers)})

    best_anchor_csv = Path(str(cfg.get("experiment", {}).get("best_anchor_csv", "")))
    best_anchor_df = load_best_anchor_table(best_anchor_csv)
    if best_anchor_df.empty:
        raise RuntimeError("light launcher 未读取到任何 best anchor 行。")

    shards = shard_best_anchor_rows(best_anchor_df, n_workers=n_workers)
    worker_cfg_paths: list[Path] = []
    worker_output_dirs: list[Path] = []
    worker_plans: list[dict[str, Any]] = []

    for worker_idx, worker_rows in enumerate(shards):
        worker_best_anchor_csv = _worker_best_anchor_path(output_dir, worker_idx)
        worker_rows_export = worker_rows.rename(
            columns={
                "anchor_idx": "best_anchor",
                "anchor_tb": "best_anchor_tb",
                "anchor_phase": "best_anchor_phase",
                "anchor_band": "best_anchor_band",
            }
        )
        worker_rows_export.to_csv(worker_best_anchor_csv, index=False)
        worker_output_dir = _worker_output_path(output_dir, worker_idx)
        worker_cfg = build_worker_config(
            cfg,
            worker_idx=worker_idx,
            worker_best_anchor_csv=worker_best_anchor_csv,
            worker_rows=worker_rows,
            worker_output_dir=worker_output_dir,
        )
        worker_cfg_path = _worker_config_path(output_dir, worker_idx)
        write_yaml(worker_cfg, worker_cfg_path)
        worker_cfg_paths.append(worker_cfg_path)
        worker_output_dirs.append(worker_output_dir)
        worker_plans.append(
            {
                "worker_idx": worker_idx,
                "best_anchor_rows": int(len(worker_rows)),
                "config_path": worker_cfg_path.as_posix(),
                "best_anchor_csv": worker_best_anchor_csv.as_posix(),
                "output_dir": worker_output_dir.as_posix(),
                "scenarios": sorted(worker_rows["scenario"].astype(str).unique().tolist()),
                "targets": sorted(worker_rows["target"].astype(str).unique().tolist()),
                "predictors": sorted(worker_rows["predictor"].astype(str).str.lower().unique().tolist()),
            }
        )

    write_json(
        {
            "best_anchor_rows_total": int(len(best_anchor_df)),
            "task_units_total": int(len(best_anchor_df)),
            "workers_total": int(len(worker_plans)),
            "workers": worker_plans,
        },
        output_dir / "worker_plan.json",
    )
    _append_launcher_progress(
        output_dir,
        "worker_plan_ready",
        {
            "best_anchor_rows_total": int(len(best_anchor_df)),
            "task_units_total": int(len(best_anchor_df)),
            "worker_count": int(len(worker_plans)),
        },
    )

    results = _launch_workers(repo_root=Path.cwd(), base_output_dir=output_dir, worker_cfg_paths=worker_cfg_paths)
    write_json({"results": results}, output_dir / "worker_results.json")
    failed = [item for item in results if int(item["returncode"]) != 0]
    if failed:
        raise RuntimeError("存在 worker 失败，请查看日志：" + "; ".join(f"worker={x['worker_idx']} log={x['log_path']}" for x in failed))

    merge_worker_outputs(
        cfg=cfg,
        config_path=config_path,
        output_dir=output_dir,
        worker_output_dirs=worker_output_dirs,
        worker_plans=worker_plans,
        n_workers=n_workers,
    )
    _append_launcher_progress(output_dir, "launcher_end", {"output_dir": output_dir.as_posix()})
    print(f"Result3.3B/C light parallel launcher completed: {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel launcher for Result3.3B/C light best-anchor experiments.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-workers", type=int, required=True)
    args = parser.parse_args()
    run_result33bc_light_parallel_launcher(Path(args.config), n_workers=int(args.n_workers))


if __name__ == "__main__":
    main()
