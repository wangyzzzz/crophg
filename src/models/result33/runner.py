from __future__ import annotations

import argparse
import json
import os
import random
import re
import warnings
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
from dataclasses import replace
from sklearn.exceptions import ConvergenceWarning

from models.anchorwise.data_loader import MultiTargetDataBundle, load_multitarget_model_inputs
from models.anchorwise.feature_builder import AnchorDefinition, build_anchor_definitions, build_feature_matrix
from models.anchorwise.modeling import HAS_LIGHTGBM, build_pipeline, sanitize_feature_matrix, suggest_params
from models.common.io_utils import ensure_dir, get_git_commit_hash, now_iso, read_yaml, write_json, write_yaml
from models.common.metrics import mean_regression_metrics, regression_metrics
from models.common.optuna_reuse import normalize_outer_groups, resolve_optuna_reuse_policy, should_search_on_outer_fold
from models.multimodal_ablation.runner import _build_modality_feature_matrix
from models.result31.runner import _resolve_split_groups_for_scenario
from models.result32.runner import _build_outer_payload, _prepare_split_groups
from crophg.common.report_utils import markdown_table

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Skipping features without any observed values")
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt", category=RuntimeWarning)

DEFAULT_MODALITY_VARIANTS = ["G", "GH_SINGLE", "GH_FULL"]
DEFAULT_TARGETS = ["ActualYD", "CM", "LM", "PHM", "Spike", "TKW"]
VEGETATION_INDEX_TOKENS = (
    "EVI2",
    "GNDVI",
    "GRVI",
    "MGRVI",
    "MSAVI",
    "MSR",
    "NDRE",
    "NDVI",
    "OSAVI",
    "RDVI",
    "SAVI",
    "VARI",
)
TEXTURE_TOKENS = ("GLCM", "TEXTURE", "ASM", "CON", "COR", "DIS", "ENT", "HOM")


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _default_data_cfg() -> dict:
    return {
        "timeline_dirs": {
            "gdd_rel_heading": "data/processed/model_inputs_engineered/gdd_rel_heading",
        },
        "scenarios": {
            "reference": {
                "custom_split_strategy": "reference",
                "genotype_fold_map_path": "outputs/reports/loso_genotype_nested_genotype_fold_map.csv",
            },
            "loso": {
                "custom_split_strategy": "loso_known_genotype",
                "genotype_fold_map_path": "outputs/reports/loso_genotype_nested_genotype_fold_map.csv",
            },
        },
    }


def _resolve_timeline_dirs(data_cfg: dict) -> dict:
    return data_cfg.get("timeline_dirs", _default_data_cfg()["timeline_dirs"])


def _resolve_scenarios_cfg(data_cfg: dict) -> dict:
    return data_cfg.get("scenarios", _default_data_cfg()["scenarios"])


def _resolve_output_dir(output_cfg: dict) -> Path:
    base = Path(
        output_cfg.get(
            "output_dir_base",
            output_cfg.get("output_dir", "outputs/experiments/result3_3_single_anchor_multitrait"),
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


def _append_progress(out_dir: Path | None, *, enabled: bool, event: str, payload: dict) -> None:
    if not enabled or out_dir is None:
        return
    path = out_dir / "progress.jsonl"
    record = {"ts": now_iso(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


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

    active_predictors: list[str] = []
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


def _normalize_modality_variant(value: str) -> str:
    token = str(value).strip().upper().replace("-", "_").replace("+", "_").replace(" ", "_")
    aliases = {
        "G": "G",
        "H_SINGLE": "H_SINGLE",
        "H": "H_SINGLE",
        "GH_SINGLE": "GH_SINGLE",
        "G_H_SINGLE": "GH_SINGLE",
        "HG_SINGLE": "GH_SINGLE",
        "H_SINGLE_VI": "H_SINGLE_VI",
        "H_VI": "H_SINGLE_VI",
        "H_SINGLE_INDEX": "H_SINGLE_VI",
        "GH_SINGLE_VI": "GH_SINGLE_VI",
        "G_H_SINGLE_VI": "GH_SINGLE_VI",
        "GH_VI": "GH_SINGLE_VI",
        "G_H_VI": "GH_SINGLE_VI",
        "GHFULL": "GH_FULL",
        "GH_FULL": "GH_FULL",
        "G_H_FULL": "GH_FULL",
        "HG_FULL": "GH_FULL",
    }
    if token in aliases:
        return aliases[token]
    raise ValueError(f"不支持的 modality_variant: {value}")


def _resolve_modality_variants(exp_cfg: dict) -> list[str]:
    raw = exp_cfg.get("modality_variants")
    if raw in (None, [], ""):
        raw = exp_cfg.get("modalities")
    if raw in (None, [], ""):
        raw = DEFAULT_MODALITY_VARIANTS
    out: list[str] = []
    seen = set()
    for item in raw:
        token = _normalize_modality_variant(item)
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _resolve_anchor_indices(exp_cfg: dict) -> list[int]:
    if isinstance(exp_cfg.get("anchor_indices"), list) and exp_cfg.get("anchor_indices"):
        return sorted({int(x) for x in exp_cfg["anchor_indices"] if int(x) >= 0})
    n_anchor_bins = int(exp_cfg.get("n_anchor_bins", 20))
    return list(range(max(0, n_anchor_bins)))


def _normalize_vi_name(value: Any) -> str:
    token = str(value).strip()
    if not token:
        raise ValueError("vi_name 不能为空。")
    token = token.lower().replace("-", "_").replace(" ", "_")
    if token.startswith("vi_"):
        return token
    return f"vi_{token}"


def _resolve_vi_names(exp_cfg: dict) -> list[str]:
    raw = exp_cfg.get("vi_names")
    if raw in (None, "", []):
        raw = exp_cfg.get("vegetation_indices")
    if raw in (None, "", []):
        return [f"vi_{token.lower()}" for token in VEGETATION_INDEX_TOKENS]
    out: list[str] = []
    seen = set()
    for item in raw:
        token = _normalize_vi_name(item)
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _feature_id_from_task_spec(spec: dict) -> str | None:
    vi_name = spec.get("vi_name")
    if vi_name not in (None, "", "null"):
        return _normalize_vi_name(vi_name)
    vi_names = spec.get("vi_names")
    if vi_names not in (None, "", [], "null"):
        if not isinstance(vi_names, list):
            raise TypeError("task_spec.vi_names 必须是列表。")
        return ",".join(_normalize_vi_name(x) for x in vi_names)
    return None


def _normalize_task_spec(spec: dict) -> tuple[str, str, str, str, str, int | None, str | None]:
    required = ["scenario", "timeline", "target", "predictor", "modality_variant"]
    missing = [k for k in required if k not in spec]
    if missing:
        raise KeyError(f"task_spec 缺少字段: {missing}")
    anchor_idx = spec.get("anchor_idx")
    return (
        str(spec["scenario"]),
        str(spec["timeline"]),
        str(spec["target"]),
        str(spec["predictor"]).lower(),
        _normalize_modality_variant(spec["modality_variant"]),
        None if anchor_idx in (None, "", "null") else int(anchor_idx),
        _feature_id_from_task_spec(spec),
    )


def _resolve_task_filter_set(exp_cfg: dict) -> set[tuple[str, str, str, str, str, int | None, str | None]]:
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


def enumerate_result33_tasks_from_config(cfg: dict) -> list[dict]:
    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})

    timeline_dirs = _resolve_timeline_dirs(data_cfg)
    scenarios_cfg = _resolve_scenarios_cfg(data_cfg)
    targets = [str(x) for x in exp_cfg.get("targets", DEFAULT_TARGETS)]
    active_predictors = _resolve_active_predictors(exp_cfg)
    modality_variants = _resolve_modality_variants(exp_cfg)
    anchor_indices = _resolve_anchor_indices(exp_cfg)
    vi_names = _resolve_vi_names(exp_cfg)
    task_filter_set = _resolve_task_filter_set(exp_cfg)

    tasks: list[dict] = []
    for scenario_name in scenarios_cfg.keys():
        for timeline_name in timeline_dirs.keys():
            for target_col in targets:
                for predictor in active_predictors:
                    for modality_variant in modality_variants:
                        if modality_variant in {"H_SINGLE", "GH_SINGLE"}:
                            for anchor_idx in anchor_indices:
                                task = {
                                    "scenario": scenario_name,
                                    "timeline": timeline_name,
                                    "target": target_col,
                                    "predictor": predictor,
                                    "modality_variant": modality_variant,
                                    "anchor_idx": int(anchor_idx),
                                }
                                task_key = (
                                    scenario_name,
                                    timeline_name,
                                    target_col,
                                    predictor,
                                    modality_variant,
                                    int(anchor_idx),
                                    None,
                                )
                                if task_filter_set and task_key not in task_filter_set:
                                    continue
                                tasks.append(task)
                            continue

                        if modality_variant in {"H_SINGLE_VI", "GH_SINGLE_VI"}:
                            for anchor_idx in anchor_indices:
                                for vi_name in vi_names:
                                    task = {
                                        "scenario": scenario_name,
                                        "timeline": timeline_name,
                                        "target": target_col,
                                        "predictor": predictor,
                                        "modality_variant": modality_variant,
                                        "anchor_idx": int(anchor_idx),
                                        "vi_name": vi_name,
                                    }
                                    task_key = (
                                        scenario_name,
                                        timeline_name,
                                        target_col,
                                        predictor,
                                        modality_variant,
                                        int(anchor_idx),
                                        vi_name,
                                    )
                                    if task_filter_set and task_key not in task_filter_set:
                                        continue
                                    tasks.append(task)
                            continue

                        task = {
                            "scenario": scenario_name,
                            "timeline": timeline_name,
                            "target": target_col,
                            "predictor": predictor,
                            "modality_variant": modality_variant,
                            "anchor_idx": None,
                        }
                        task_key = (scenario_name, timeline_name, target_col, predictor, modality_variant, None, None)
                        if task_filter_set and task_key not in task_filter_set:
                            continue
                        tasks.append(task)
    return tasks


def _stable_seed(seed: int, *parts: Any) -> int:
    text = "|".join(str(x) for x in parts)
    return int((seed + zlib.crc32(text.encode("utf-8"))) % (2**31 - 1))


def _anchor_band(anchor_idx: int, n_anchor_total: int) -> str:
    if n_anchor_total <= 1:
        return "mid"
    ratio = float(anchor_idx) / float(max(1, n_anchor_total - 1))
    if ratio < 0.25:
        return "early"
    if ratio < 0.50:
        return "mid"
    if ratio < 0.75:
        return "late"
    return "tail"


def _h_feature_base_name(col_name: str) -> str:
    if "__ph_" in col_name:
        return col_name.split("__ph_")[0]
    if "__tb_" in col_name:
        return col_name.split("__tb_")[0]
    return col_name


def _tail_fill_hyperspectral_tail_only(bundle: MultiTargetDataBundle) -> MultiTargetDataBundle:
    if bundle.x_hyperspectral.empty or not bundle.hyperspectral_tb_map:
        return bundle

    filled = bundle.x_hyperspectral.copy()
    phase_to_keys: dict[str | None, list[tuple[str | None, int]]] = {}
    for key in sorted(bundle.hyperspectral_tb_map.keys(), key=lambda x: (str(x[0]), int(x[1]))):
        phase_to_keys.setdefault(key[0], []).append(key)

    for phase, keys in phase_to_keys.items():
        ordered_keys = sorted(keys, key=lambda x: int(x[1]))
        key_cols = {
            key: [c for c in bundle.hyperspectral_tb_map.get(key, []) if c in filled.columns]
            for key in ordered_keys
        }
        key_cols = {k: v for k, v in key_cols.items() if v}
        if not key_cols:
            continue

        phase_cols = [c for key in ordered_keys for c in key_cols.get(key, [])]
        phase_df = filled.loc[:, phase_cols].copy()
        for row_label in phase_df.index:
            last_valid_state: dict[str, float] | None = None
            for key in ordered_keys:
                cols = key_cols.get(key, [])
                if not cols:
                    continue
                row_vals = phase_df.loc[row_label, cols]
                if row_vals.notna().any():
                    last_valid_state = {
                        _h_feature_base_name(col): float(row_vals[col])
                        for col in cols
                        if pd.notna(row_vals[col])
                    }
                elif last_valid_state is not None:
                    fill_values = []
                    for col in cols:
                        fill_values.append(last_valid_state.get(_h_feature_base_name(col), np.nan))
                    phase_df.loc[row_label, cols] = fill_values
        filled.loc[:, phase_cols] = phase_df
    return replace(bundle, x_hyperspectral=filled)


def _safe_std(s: pd.Series) -> float:
    return float(np.std(pd.to_numeric(s, errors="coerce"), ddof=0))


def _summarize_feature_info(
    *,
    timeline: str,
    modality_variant: str,
    anchor_idx: int | None,
    anchor_tb: int | None,
    anchor_phase: str | None,
    anchor_band: str | None,
    x: pd.DataFrame,
    info: dict,
) -> dict:
    return {
        "timeline": timeline,
        "modality_variant": modality_variant,
        "anchor_idx": anchor_idx,
        "anchor_tb": anchor_tb,
        "anchor_phase": anchor_phase,
        "anchor_band": anchor_band,
        "vi_name": info.get("vi_name"),
        "vi_names": ",".join(info.get("vi_names", [])) if info.get("vi_names") else None,
        "n_samples": int(x.shape[0]),
        "n_features_total": int(info.get("n_features_total", x.shape[1])),
        "n_features_h": int(info.get("n_features_h", 0)),
        "n_features_g": int(info.get("n_features_g", 0)),
        "h_cols": list(info.get("h_cols", [])),
        "g_cols": list(info.get("g_cols", [])),
    }


def _build_result33_feature_matrix(
    *,
    bundle: MultiTargetDataBundle,
    anchors: list[AnchorDefinition],
    modality_variant: str,
    anchor_idx: int | None,
    vi_name: str | None = None,
    vi_names: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    modality_variant = _normalize_modality_variant(modality_variant)
    if modality_variant == "G":
        x = sanitize_feature_matrix(bundle.x_genotype.copy())
        info = {
            "modality_variant": "G",
            "n_features_total": int(x.shape[1]),
            "n_features_h": 0,
            "n_features_g": int(x.shape[1]),
            "h_cols": [],
            "g_cols": list(x.columns),
        }
        return x, info

    filled_bundle = getattr(bundle, "_result33_tail_filled_bundle", None)
    if filled_bundle is None:
        filled_bundle = _tail_fill_hyperspectral_tail_only(bundle)
        setattr(bundle, "_result33_tail_filled_bundle", filled_bundle)
    bundle = filled_bundle

    if not anchors:
        raise ValueError("anchors 为空，无法构造 H 相关输入。")

    if modality_variant == "GH_FULL":
        anchor = anchors[-1]
        x_all, x_info = build_feature_matrix(bundle, anchor.anchor_key, input_type="temporal")
        x, info = _build_modality_feature_matrix(
            x_all=sanitize_feature_matrix(x_all),
            x_info=x_info,
            modality_combo="H+G",
            sample_index=bundle.meta.index,
        )
        x = sanitize_feature_matrix(x)
        info["modality_variant"] = modality_variant
        return x, info

    if anchor_idx is None:
        raise ValueError(f"{modality_variant} 需要提供 anchor_idx。")
    if anchor_idx < 0 or anchor_idx >= len(anchors):
        raise IndexError(f"anchor_idx 超出范围: {anchor_idx} / {len(anchors)}")

    anchor = anchors[int(anchor_idx)]
    x_all, x_info = build_feature_matrix(bundle, anchor.anchor_key, input_type="static")
    if modality_variant == "H_SINGLE":
        x, info = _build_modality_feature_matrix(
            x_all=sanitize_feature_matrix(x_all),
            x_info=x_info,
            modality_combo="H",
            sample_index=bundle.meta.index,
        )
        x = sanitize_feature_matrix(x)
        info["modality_variant"] = modality_variant
        return x, info

    if modality_variant in {"H_SINGLE_VI", "GH_SINGLE_VI"}:
        requested_vi_names = []
        if vi_names:
            requested_vi_names.extend(_normalize_vi_name(v) for v in vi_names)
        elif vi_name is not None:
            requested_vi_names.append(_normalize_vi_name(vi_name))
        else:
            raise ValueError(f"{modality_variant} 需要提供 vi_name 或 vi_names。")

        h_cols = [
            col
            for col in x_info.get("h_cols", [])
            if _h_feature_base_name(str(col)).lower() in set(requested_vi_names)
        ]
        if not h_cols:
            raise ValueError(
                f"{modality_variant} 在 anchor_idx={anchor_idx} 未找到 VI 列: {requested_vi_names}"
            )
        g_cols = list(x_info.get("g_cols", [])) if modality_variant == "GH_SINGLE_VI" else []
        use_cols = h_cols + g_cols
        x = sanitize_feature_matrix(x_all.loc[:, use_cols].copy())
        info = {
            "modality_variant": modality_variant,
            "n_features_total": int(x.shape[1]),
            "n_features_h": int(len(h_cols)),
            "n_features_g": int(len(g_cols)),
            "h_cols": list(h_cols),
            "g_cols": list(g_cols),
            "vi_name": requested_vi_names[0] if len(requested_vi_names) == 1 else None,
            "vi_names": list(requested_vi_names),
        }
        return x, info

    x, info = _build_modality_feature_matrix(
        x_all=sanitize_feature_matrix(x_all),
        x_info=x_info,
        modality_combo="H+G",
        sample_index=bundle.meta.index,
    )
    x = sanitize_feature_matrix(x)
    info["modality_variant"] = modality_variant
    return x, info


def _build_metrics_summary(metrics_fold_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_fold_df.empty:
        return pd.DataFrame()

    group_cols = [
        col
        for col in [
            "scenario",
            "target",
            "timeline",
            "modality_variant",
            "predictor",
            "anchor_idx",
            "anchor_tb",
            "anchor_phase",
            "anchor_band",
            "vi_name",
            "vi_names",
            "ablation_group",
        ]
        if col in metrics_fold_df.columns
    ]
    out = (
        metrics_fold_df.groupby(group_cols, dropna=False, as_index=False)
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
            n_features_g=("n_features_g", "mean"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    for col in ["n_outer_folds", "n_inner_folds", "n_optuna_trials", "n_features_total", "n_features_h", "n_features_g"]:
        if col in out.columns:
            out[col] = out[col].astype(int)
    return out


def _build_metrics_by_year(oof_df: pd.DataFrame) -> pd.DataFrame:
    if oof_df.empty:
        return pd.DataFrame()

    rows = []
    group_cols = [
        col
        for col in [
            "scenario",
            "target",
            "timeline",
            "modality_variant",
            "predictor",
            "anchor_idx",
            "anchor_tb",
            "anchor_phase",
            "anchor_band",
            "vi_name",
            "vi_names",
            "ablation_group",
            "year",
        ]
        if col in oof_df.columns
    ]
    for keys, sub in oof_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {col: keys[idx] for idx, col in enumerate(group_cols)}
        metrics = regression_metrics(sub["y_true"].to_numpy(dtype=float), sub["y_pred"].to_numpy(dtype=float))
        record.update(
            {
                "r2": float(metrics["r2"]),
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "pearson": float(metrics["pearson_r"]),
                "n_samples": int(len(sub)),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _build_best_anchor_summary(metrics_summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics_summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = metrics_summary_df.copy()
    if "ablation_group" in work.columns:
        work["ablation_group"] = work["ablation_group"].fillna("FULL")
        work = work[work["ablation_group"] == "FULL"].copy()

    base_cols = ["scenario", "target", "timeline", "predictor"]
    g_df = (
        work[work["modality_variant"] == "G"]
        .loc[:, base_cols + ["mean_pearson", "mean_r2"]]
        .rename(columns={"mean_pearson": "g_mean_pearson", "mean_r2": "g_mean_r2"})
    )
    single_df = work[work["modality_variant"] == "GH_SINGLE"].copy()
    if g_df.empty or single_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    delta_df = single_df.merge(g_df, on=base_cols, how="left")
    delta_df["delta_pearson"] = delta_df["mean_pearson"] - delta_df["g_mean_pearson"]
    delta_df["delta_r2"] = delta_df["mean_r2"] - delta_df["g_mean_r2"]
    delta_df["anchor_idx"] = delta_df["anchor_idx"].astype(int)

    band_values = []
    for _, sub in delta_df.groupby(base_cols, dropna=False):
        n_anchor_total = int(sub["anchor_idx"].nunique())
        band_values.extend([_anchor_band(int(idx), n_anchor_total) for idx in sub["anchor_idx"].tolist()])
    delta_df["best_anchor_band"] = band_values

    best_rows = []
    for keys, sub in delta_df.groupby(base_cols, dropna=False):
        part = sub.sort_values(
            ["delta_pearson", "delta_r2", "mean_pearson", "anchor_idx"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        top = part.iloc[0]
        best_rows.append(
            {
                "scenario": keys[0],
                "target": keys[1],
                "timeline": keys[2],
                "predictor": keys[3],
                "best_anchor": int(top["anchor_idx"]),
                "best_anchor_tb": top["anchor_tb"],
                "best_anchor_phase": top.get("anchor_phase"),
                "best_anchor_band": top["best_anchor_band"],
                "best_delta": float(top["delta_pearson"]),
                "best_delta_r2": float(top["delta_r2"]),
                "best_mean_pearson": float(top["mean_pearson"]),
                "best_mean_r2": float(top["mean_r2"]),
                "g_mean_pearson": float(top["g_mean_pearson"]),
                "g_mean_r2": float(top["g_mean_r2"]),
            }
        )

    best_df = pd.DataFrame(best_rows).sort_values(base_cols).reset_index(drop=True)
    delta_df = delta_df.sort_values(base_cols + ["anchor_idx"]).reset_index(drop=True)
    return delta_df, best_df


def _build_single_vi_transferability_summary(
    metrics_summary_df: pd.DataFrame,
    *,
    easy_scenarios: tuple[str, ...] = ("reference", "within_season"),
    hard_scenarios: tuple[str, ...] = ("loso", "loso_genotype"),
) -> pd.DataFrame:
    if metrics_summary_df.empty or "vi_name" not in metrics_summary_df.columns:
        return pd.DataFrame()

    work = metrics_summary_df[metrics_summary_df["modality_variant"] == "H_SINGLE_VI"].copy()
    if work.empty:
        return pd.DataFrame()
    work["scenario_group"] = np.where(
        work["scenario"].isin(easy_scenarios),
        "easy",
        np.where(work["scenario"].isin(hard_scenarios), "hard", "other"),
    )
    work = work[work["scenario_group"].isin(["easy", "hard"])].copy()
    if work.empty:
        return pd.DataFrame()

    group_cols = [
        col
        for col in ["target", "timeline", "predictor", "anchor_idx", "anchor_tb", "anchor_phase", "anchor_band", "vi_name"]
        if col in work.columns
    ]
    pivot = (
        work.groupby(group_cols + ["scenario_group"], dropna=False, as_index=False)
        .agg(
            mean_pearson=("mean_pearson", "mean"),
            mean_r2=("mean_r2", "mean"),
            n_scenarios=("scenario", "nunique"),
        )
        .pivot(index=group_cols, columns="scenario_group")
    )
    pivot.columns = ["_".join(str(x) for x in col if str(x)) for col in pivot.columns.to_flat_index()]
    out = pivot.reset_index()
    rename_map = {
        "mean_pearson_easy": "easy_mean_pearson",
        "mean_pearson_hard": "hard_mean_pearson",
        "mean_r2_easy": "easy_mean_r2",
        "mean_r2_hard": "hard_mean_r2",
        "n_scenarios_easy": "easy_n_scenarios",
        "n_scenarios_hard": "hard_n_scenarios",
    }
    out = out.rename(columns=rename_map)
    for col in ["easy_mean_pearson", "hard_mean_pearson", "easy_mean_r2", "hard_mean_r2"]:
        if col not in out.columns:
            out[col] = np.nan
    out["transfer_loss_pearson"] = out["easy_mean_pearson"] - out["hard_mean_pearson"]
    out["transfer_loss_r2"] = out["easy_mean_r2"] - out["hard_mean_r2"]
    out["hard_retention_pearson"] = out["hard_mean_pearson"] / out["easy_mean_pearson"].replace(0, np.nan)

    out["rank_easy"] = (
        out.groupby(["target", "timeline", "predictor", "anchor_idx"], dropna=False)["easy_mean_pearson"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    out["rank_hard"] = (
        out.groupby(["target", "timeline", "predictor", "anchor_idx"], dropna=False)["hard_mean_pearson"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    out["rank_shift"] = out["rank_hard"].astype("float") - out["rank_easy"].astype("float")
    return out.sort_values(
        ["target", "timeline", "predictor", "anchor_idx", "transfer_loss_pearson"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)


def _build_g_conditioned_single_vi_summary(metrics_summary_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_summary_df.empty or "vi_name" not in metrics_summary_df.columns:
        return pd.DataFrame()

    base_cols = ["scenario", "target", "timeline", "predictor"]
    g_df = (
        metrics_summary_df[metrics_summary_df["modality_variant"] == "G"]
        .loc[:, base_cols + ["mean_pearson", "mean_r2"]]
        .drop_duplicates()
        .rename(columns={"mean_pearson": "g_mean_pearson", "mean_r2": "g_mean_r2"})
    )
    single_df = metrics_summary_df[metrics_summary_df["modality_variant"] == "GH_SINGLE_VI"].copy()
    if g_df.empty or single_df.empty:
        return pd.DataFrame()

    out = single_df.merge(g_df, on=base_cols, how="left")
    out["delta_g_pearson"] = out["mean_pearson"] - out["g_mean_pearson"]
    out["delta_g_r2"] = out["mean_r2"] - out["g_mean_r2"]
    out["rank_delta_g_pearson"] = (
        out.groupby(base_cols + ["anchor_idx"], dropna=False)["delta_g_pearson"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    keep_cols = [
        "scenario",
        "target",
        "timeline",
        "predictor",
        "anchor_idx",
        "anchor_tb",
        "anchor_phase",
        "anchor_band",
        "vi_name",
        "mean_pearson",
        "mean_r2",
        "g_mean_pearson",
        "g_mean_r2",
        "delta_g_pearson",
        "delta_g_r2",
        "rank_delta_g_pearson",
    ]
    return out.loc[:, [c for c in keep_cols if c in out.columns]].sort_values(
        ["scenario", "target", "timeline", "predictor", "anchor_idx", "delta_g_pearson"],
        ascending=[True, True, True, True, True, False],
    ).reset_index(drop=True)


def _build_top_positive_anchor_summary(
    metrics_summary_df: pd.DataFrame,
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    if metrics_summary_df.empty:
        return pd.DataFrame()

    work = metrics_summary_df.copy()
    if "ablation_group" in work.columns:
        work["ablation_group"] = work["ablation_group"].fillna("FULL")
        work = work[work["ablation_group"] == "FULL"].copy()

    base_cols = ["scenario", "target", "timeline", "predictor"]
    g_df = (
        work[work["modality_variant"] == "G"]
        .loc[:, base_cols + ["mean_pearson", "mean_r2"]]
        .rename(columns={"mean_pearson": "g_mean_pearson", "mean_r2": "g_mean_r2"})
    )
    single_df = work[work["modality_variant"] == "GH_SINGLE"].copy()
    if g_df.empty or single_df.empty:
        return pd.DataFrame()

    merged = single_df.merge(g_df, on=base_cols, how="left")
    merged["delta_pearson"] = merged["mean_pearson"] - merged["g_mean_pearson"]
    merged["delta_r2"] = merged["mean_r2"] - merged["g_mean_r2"]
    merged["anchor_idx"] = merged["anchor_idx"].astype(int)
    merged = merged[merged["delta_pearson"] > 0].copy()
    if merged.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for _, sub in merged.groupby(base_cols, dropna=False):
        top = (
            sub.sort_values(
                ["delta_pearson", "delta_r2", "mean_pearson", "anchor_idx"],
                ascending=[False, False, False, True],
            )
            .head(int(top_k))
            .copy()
        )
        top["rank_within_trait_scenario_predictor"] = range(1, len(top) + 1)
        rows.append(top)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    keep_cols = [
        "scenario",
        "target",
        "timeline",
        "predictor",
        "anchor_idx",
        "anchor_tb",
        "anchor_phase",
        "anchor_band",
        "delta_pearson",
        "delta_r2",
        "mean_pearson",
        "mean_r2",
        "g_mean_pearson",
        "g_mean_r2",
        "rank_within_trait_scenario_predictor",
    ]
    return out.loc[:, keep_cols].sort_values(
        ["scenario", "target", "predictor", "rank_within_trait_scenario_predictor"]
    ).reset_index(drop=True)


def _infer_h_factor_group(col_name: str) -> str:
    token = str(col_name).split("__")[0].upper()
    if any(x in token for x in VEGETATION_INDEX_TOKENS):
        return "VI"
    if any(x in token for x in TEXTURE_TOKENS) or token.startswith("TEX_"):
        return "TEXTURE"
    if re.search(r"(RED|RE|NIR|SWIR|BAND|WL|WAVE|R)\_?\d{3,4}$", token) or re.fullmatch(r"\d{3,4}", token):
        return "SPECTRAL_BAND"
    return "OTHER"


def _group_h_columns_for_anchor(bundle: MultiTargetDataBundle, anchors: list[AnchorDefinition], anchor_idx: int) -> dict[str, list[str]]:
    _, info = _build_result33_feature_matrix(
        bundle=bundle,
        anchors=anchors,
        modality_variant="GH_SINGLE",
        anchor_idx=anchor_idx,
    )
    groups: dict[str, list[str]] = {}
    for col in info.get("h_cols", []):
        group = _infer_h_factor_group(col)
        groups.setdefault(group, []).append(col)
    return {k: sorted(v) for k, v in sorted(groups.items()) if v}


def _single_h_columns_for_anchor(bundle: MultiTargetDataBundle, anchors: list[AnchorDefinition], anchor_idx: int) -> dict[str, list[str]]:
    _, info = _build_result33_feature_matrix(
        bundle=bundle,
        anchors=anchors,
        modality_variant="GH_SINGLE",
        anchor_idx=anchor_idx,
    )
    out: dict[str, list[str]] = {}
    for col in info.get("h_cols", []):
        out[str(col).split("__")[0]] = [col]
    return {k: v for k, v in sorted(out.items()) if v}


def _h_single_vi_columns_for_anchor(
    bundle: MultiTargetDataBundle,
    anchors: list[AnchorDefinition],
    anchor_idx: int,
    vi_name: str,
) -> list[str]:
    _, info = _build_result33_feature_matrix(
        bundle=bundle,
        anchors=anchors,
        modality_variant="GH_SINGLE",
        anchor_idx=anchor_idx,
    )
    target = _normalize_vi_name(vi_name)
    cols = [
        col
        for col in info.get("h_cols", [])
        if _h_feature_base_name(str(col)).lower() == target
    ]
    return sorted(cols)


def _available_single_vi_names_by_anchor(
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
        _, info = _build_result33_feature_matrix(
            bundle=bundle,
            anchors=anchors,
            modality_variant="GH_SINGLE",
            anchor_idx=anchor_idx,
        )
        vi_names = {
            _h_feature_base_name(str(col)).lower()
            for col in info.get("h_cols", [])
            if _h_feature_base_name(str(col)).lower().startswith("vi_")
        }
        out[anchor_idx] = vi_names
    return out


def _build_factor_group_importance(metrics_summary_df: pd.DataFrame, best_anchor_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_summary_df.empty or best_anchor_df.empty or "ablation_group" not in metrics_summary_df.columns:
        return pd.DataFrame()

    work = metrics_summary_df.copy()
    work["ablation_group"] = work["ablation_group"].fillna("FULL")
    best_cols = ["scenario", "target", "timeline", "predictor", "best_anchor", "best_anchor_tb"]
    best_ref = best_anchor_df.loc[:, best_cols].copy()
    merged = work.merge(
        best_ref,
        left_on=["scenario", "target", "timeline", "predictor", "anchor_idx"],
        right_on=["scenario", "target", "timeline", "predictor", "best_anchor"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    full_df = (
        merged[merged["ablation_group"] == "FULL"]
        .loc[
            :,
            [
                "scenario",
                "target",
                "timeline",
                "predictor",
                "best_anchor",
                "best_anchor_tb",
                "mean_pearson",
                "mean_r2",
            ],
        ]
        .rename(columns={"mean_pearson": "full_mean_pearson", "mean_r2": "full_mean_r2"})
    )
    ablated_df = merged[merged["ablation_group"] != "FULL"].copy()
    if full_df.empty or ablated_df.empty:
        return pd.DataFrame()

    out = ablated_df.merge(
        full_df,
        on=["scenario", "target", "timeline", "predictor", "best_anchor", "best_anchor_tb"],
        how="left",
    )
    out["importance"] = out["full_mean_pearson"] - out["mean_pearson"]
    out["importance_r2"] = out["full_mean_r2"] - out["mean_r2"]
    out = out.rename(columns={"ablation_group": "factor_group"})
    keep_cols = [
        "scenario",
        "target",
        "timeline",
        "predictor",
        "best_anchor",
        "best_anchor_tb",
        "factor_group",
        "importance",
        "importance_r2",
        "full_mean_pearson",
        "mean_pearson",
        "full_mean_r2",
        "mean_r2",
    ]
    return out.loc[:, keep_cols].sort_values(
        ["scenario", "target", "timeline", "predictor", "importance"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)


def _build_index_importance(metrics_summary_df: pd.DataFrame, selected_anchor_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_summary_df.empty or selected_anchor_df.empty or "ablation_group" not in metrics_summary_df.columns:
        return pd.DataFrame()

    work = metrics_summary_df.copy()
    work["ablation_group"] = work["ablation_group"].fillna("FULL")
    ref_cols = ["scenario", "target", "timeline", "predictor", "anchor_idx", "anchor_tb"]
    selected_ref = selected_anchor_df.loc[:, ref_cols].drop_duplicates().copy()

    merged = work.merge(
        selected_ref,
        on=["scenario", "target", "timeline", "predictor", "anchor_idx", "anchor_tb"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    full_df = (
        merged[merged["ablation_group"] == "FULL"]
        .loc[
            :,
            [
                "scenario",
                "target",
                "timeline",
                "predictor",
                "anchor_idx",
                "anchor_tb",
                "mean_pearson",
                "mean_r2",
            ],
        ]
        .rename(columns={"mean_pearson": "full_mean_pearson", "mean_r2": "full_mean_r2"})
    )
    ablated_df = merged[merged["ablation_group"] != "FULL"].copy()
    if full_df.empty or ablated_df.empty:
        return pd.DataFrame()

    out = ablated_df.merge(
        full_df,
        on=["scenario", "target", "timeline", "predictor", "anchor_idx", "anchor_tb"],
        how="left",
    )
    out["importance_pearson"] = out["full_mean_pearson"] - out["mean_pearson"]
    out["importance_r2"] = out["full_mean_r2"] - out["mean_r2"]
    out = out.rename(columns={"ablation_group": "vegetation_index", "mean_pearson": "loo_mean_pearson", "mean_r2": "loo_mean_r2"})
    keep_cols = [
        "scenario",
        "target",
        "timeline",
        "predictor",
        "anchor_idx",
        "anchor_tb",
        "vegetation_index",
        "importance_pearson",
        "importance_r2",
        "full_mean_pearson",
        "loo_mean_pearson",
        "full_mean_r2",
        "loo_mean_r2",
    ]
    return out.loc[:, keep_cols].sort_values(
        ["scenario", "target", "timeline", "predictor", "anchor_idx", "importance_pearson"],
        ascending=[True, True, True, True, True, False],
    ).reset_index(drop=True)


def _to_records_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _run_single_task(
    *,
    x: pd.DataFrame,
    y_full: pd.Series,
    meta_full: pd.DataFrame,
    prepared_groups: Dict[Tuple[int, int], List[dict]],
    scenario_name: str,
    timeline_name: str,
    target_col: str,
    predictor: str,
    modality_variant: str,
    anchor_idx: int | None,
    anchor_tb: int | None,
    anchor_phase: str | None,
    anchor_band: str | None,
    vi_name: str | None,
    vi_names: list[str] | None,
    n_features_h: int,
    n_features_g: int,
    n_features_total: int,
    seed: int,
    optuna_cfg: dict,
    preprocess_cfg: dict,
    model_cfg: dict,
    ablation_group: str = "FULL",
) -> tuple[list[dict], list[dict], list[dict]]:
    metrics_rows: list[dict] = []
    oof_rows: list[dict] = []
    trial_rows: list[dict] = []

    n_trials = int(optuna_cfg.get("n_trials", 10))
    lambda_gap = float(optuna_cfg.get("lambda_gap", 1.0))
    pruner_startup_trials = int(optuna_cfg.get("pruner_startup_trials", 1))
    ordered_outer_items = normalize_outer_groups(prepared_groups)
    ordered_outer_keys = [key for key, _ in ordered_outer_items]
    reuse_policy = resolve_optuna_reuse_policy(optuna_cfg)
    cached_best_params = None
    cached_source = None

    for (target_year, outer_fold), group_splits in ordered_outer_items:
        payload = _build_outer_payload(x, y_full, meta_full, group_splits)
        inner_payload = payload["inner_payload"]
        x_test = payload["x_test"]
        y_test = payload["y_test"]
        meta_test = payload["meta_test"]
        if len(x_test) == 0 or not inner_payload:
            continue

        sampler_seed = _stable_seed(
            seed,
            scenario_name,
            timeline_name,
            target_col,
            predictor,
            modality_variant,
            anchor_idx,
            target_year,
            outer_fold,
            ablation_group,
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
                train_metrics_list = []
                val_metrics_list = []
                for fold_i, inner in enumerate(inner_payload):
                    pipe = build_pipeline(
                        predictor=predictor,
                        params=params,
                        n_features=inner["x_train"].shape[1],
                        preprocess_cfg=preprocess_cfg,
                        seed=int(sampler_seed) + int(trial.number) + int(fold_i),
                        model_cfg=model_cfg,
                    )
                    pipe.fit(inner["x_train"], inner["y_train"])
                    pred_train = pipe.predict(inner["x_train"])
                    pred_val = pipe.predict(inner["x_val"])
                    train_metrics = regression_metrics(inner["y_train"].to_numpy(), pred_train)
                    val_metrics = regression_metrics(inner["y_val"].to_numpy(), pred_val)
                    train_metrics_list.append(train_metrics)
                    val_metrics_list.append(val_metrics)

                mean_train = mean_regression_metrics(train_metrics_list)
                mean_val = mean_regression_metrics(val_metrics_list)
                gap = max(0.0, float(mean_val["rmse"]) - float(mean_train["rmse"]))
                loss = float(mean_train["rmse"]) + float(mean_val["rmse"]) + lambda_gap * gap

                trial.set_user_attr("mean_train_loss", float(mean_train["rmse"]))
                trial.set_user_attr("mean_val_loss", float(mean_val["rmse"]))
                trial.set_user_attr("gap", float(gap))
                trial.set_user_attr("objective", float(loss))
                trial.set_user_attr("mean_train_rmse", float(mean_train["rmse"]))
                trial.set_user_attr("mean_train_mae", float(mean_train["mae"]))
                trial.set_user_attr("mean_train_r2", float(mean_train["r2"]))
                trial.set_user_attr("mean_train_pearson", float(mean_train["pearson_r"]))
                trial.set_user_attr("mean_val_rmse", float(mean_val["rmse"]))
                trial.set_user_attr("mean_val_mae", float(mean_val["mae"]))
                trial.set_user_attr("mean_val_r2", float(mean_val["r2"]))
                trial.set_user_attr("mean_val_pearson", float(mean_val["pearson_r"]))
                return loss

            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            complete_trials = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
            if not complete_trials:
                continue
            best_trial = study.best_trial
            cached_best_params = dict(best_trial.params)
            cached_source = {
                "target_year": int(target_year),
                "outer_fold": int(outer_fold),
                "trial_number": int(best_trial.number),
                "objective": float(best_trial.value),
                "attrs": dict(best_trial.user_attrs),
                "n_trials": int(len(complete_trials)),
            }
        else:
            if cached_best_params is None or cached_source is None:
                raise RuntimeError(
                    "Optuna 参数复用已启用，但当前任务缺少源 outer-fold 的最佳参数缓存。"
                )
            study = None
            complete_trials = []

        best_params = dict(cached_best_params)
        source_target_year = cached_source["target_year"] if cached_source else np.nan
        source_outer_fold = cached_source["outer_fold"] if cached_source else np.nan
        source_trial_number = cached_source["trial_number"] if cached_source else np.nan
        source_objective = cached_source["objective"] if cached_source else np.nan
        best_attrs = dict(cached_source["attrs"]) if optuna_search_performed and cached_source else {}

        val_metrics_list = []
        test_pred_matrix = []
        for fold_i, inner in enumerate(inner_payload):
            pipe = build_pipeline(
                predictor=predictor,
                params=best_params,
                n_features=inner["x_train"].shape[1],
                preprocess_cfg=preprocess_cfg,
                seed=int(sampler_seed) + 10_000 + int(fold_i),
                model_cfg=model_cfg,
            )
            pipe.fit(inner["x_train"], inner["y_train"])
            if optuna_search_performed:
                pred_val = pipe.predict(inner["x_val"])
                val_metrics_list.append(regression_metrics(inner["y_val"].to_numpy(), pred_val))
            test_pred_matrix.append(np.asarray(pipe.predict(x_test), dtype=float).reshape(-1))

        mean_val_metrics = mean_regression_metrics(val_metrics_list) if optuna_search_performed else {
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "pearson_r": np.nan,
        }
        y_pred = np.mean(np.vstack(test_pred_matrix), axis=0)
        test_metrics = regression_metrics(y_test.to_numpy(dtype=float), y_pred)

        metrics_rows.append(
            {
                "scenario": scenario_name,
                "target": target_col,
                "timeline": timeline_name,
                "modality_variant": modality_variant,
                "predictor": predictor,
                "anchor_idx": anchor_idx,
                "anchor_tb": anchor_tb,
                "anchor_phase": anchor_phase,
                "anchor_band": anchor_band,
                "vi_name": vi_name,
                "vi_names": ",".join(vi_names) if vi_names else None,
                "ablation_group": ablation_group,
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
                "n_inner_folds": int(len(inner_payload)),
                "n_optuna_trials": int(len(complete_trials)) if optuna_search_performed else 0,
                "optuna_search_performed": bool(optuna_search_performed),
                "optuna_reuse_scope": reuse_policy["mode"],
                "optuna_source_target_year": source_target_year,
                "optuna_source_outer_fold": source_outer_fold,
                "optuna_source_trial_number": source_trial_number,
                "optuna_source_objective": source_objective,
                "n_features_total": int(n_features_total),
                "n_features_h": int(n_features_h),
                "n_features_g": int(n_features_g),
                "val_r2": float(mean_val_metrics["r2"]),
                "val_rmse": float(mean_val_metrics["rmse"]),
                "val_mae": float(mean_val_metrics["mae"]),
                "val_pearson": float(mean_val_metrics["pearson_r"]),
                "test_r2": float(test_metrics["r2"]),
                "test_rmse": float(test_metrics["rmse"]),
                "test_mae": float(test_metrics["mae"]),
                "test_pearson": float(test_metrics["pearson_r"]),
            }
        )

        for plot_id, row_meta, y_true, pred in zip(meta_test.index.tolist(), meta_test.itertuples(index=False), y_test.tolist(), y_pred.tolist()):
            year = getattr(row_meta, "year", np.nan)
            oof_rows.append(
                {
                    "scenario": scenario_name,
                    "target": target_col,
                    "timeline": timeline_name,
                    "modality_variant": modality_variant,
                    "predictor": predictor,
                    "anchor_idx": anchor_idx,
                    "anchor_tb": anchor_tb,
                    "anchor_phase": anchor_phase,
                    "anchor_band": anchor_band,
                    "vi_name": vi_name,
                    "vi_names": ",".join(vi_names) if vi_names else None,
                    "ablation_group": ablation_group,
                    "target_year": int(target_year),
                    "outer_fold": int(outer_fold),
                    "plot_id": str(plot_id),
                    "year": int(year) if pd.notna(year) else np.nan,
                    "y_true": float(y_true),
                    "y_pred": float(pred),
                }
            )

        if optuna_search_performed and study is not None:
            for trial in study.get_trials(deepcopy=False):
                trial_rows.append(
                    {
                        "scenario": scenario_name,
                        "target": target_col,
                        "timeline": timeline_name,
                        "modality_variant": modality_variant,
                        "predictor": predictor,
                        "anchor_idx": anchor_idx,
                        "anchor_tb": anchor_tb,
                        "anchor_phase": anchor_phase,
                        "anchor_band": anchor_band,
                        "vi_name": vi_name,
                        "vi_names": ",".join(vi_names) if vi_names else None,
                        "ablation_group": ablation_group,
                        "target_year": int(target_year),
                        "outer_fold": int(outer_fold),
                        "trial_number": int(trial.number),
                        "state": trial.state.name,
                        "value": float(trial.value) if trial.value is not None else np.nan,
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
                        "optuna_search_performed": True,
                        "optuna_reuse_scope": reuse_policy["mode"],
                        "optuna_source_target_year": source_target_year,
                        "optuna_source_outer_fold": source_outer_fold,
                        "optuna_source_trial_number": source_trial_number,
                        "optuna_source_objective": source_objective,
                    }
                )

    return metrics_rows, oof_rows, trial_rows


def _build_feature_cache(
    *,
    timeline_name: str,
    bundle: MultiTargetDataBundle,
    anchors: list[AnchorDefinition],
    modality_variants: list[str],
    vi_names: list[str] | None = None,
    anchor_indices: list[int] | None = None,
    available_single_vi_names_by_anchor: dict[int, set[str]] | None = None,
) -> tuple[dict[tuple[str, str, int | None], tuple[pd.DataFrame, dict]], list[dict]]:
    cache: dict[tuple[str, str, int | None], tuple[pd.DataFrame, dict]] = {}
    rows: list[dict] = []
    selected_anchor_indices = None if anchor_indices is None else {int(x) for x in anchor_indices}

    if "G" in modality_variants:
        x, info = _build_result33_feature_matrix(bundle=bundle, anchors=anchors, modality_variant="G", anchor_idx=None)
        cache[(timeline_name, "G", None)] = (x, info)
        rows.append(
            _summarize_feature_info(
                timeline=timeline_name,
                modality_variant="G",
                anchor_idx=None,
                anchor_tb=None,
                anchor_phase=None,
                anchor_band=None,
                x=x,
                info=info,
            )
        )

    if "GH_FULL" in modality_variants:
        x, info = _build_result33_feature_matrix(bundle=bundle, anchors=anchors, modality_variant="GH_FULL", anchor_idx=None)
        cache[(timeline_name, "GH_FULL", None)] = (x, info)
        full_anchor = anchors[-1] if anchors else None
        rows.append(
            _summarize_feature_info(
                timeline=timeline_name,
                modality_variant="GH_FULL",
                anchor_idx=None,
                anchor_tb=int(full_anchor.anchor_tb) if full_anchor is not None else None,
                anchor_phase=full_anchor.anchor_phase if full_anchor is not None else None,
                anchor_band="full_season",
                x=x,
                info=info,
            )
        )

    if "GH_SINGLE" in modality_variants or "H_SINGLE" in modality_variants:
        n_anchor_total = int(len(anchors))
        for anchor in anchors:
            if anchor_indices is not None and int(anchor.anchor_idx) not in set(anchor_indices):
                continue
            for variant in [v for v in ["H_SINGLE", "GH_SINGLE"] if v in modality_variants]:
                x, info = _build_result33_feature_matrix(
                    bundle=bundle,
                    anchors=anchors,
                    modality_variant=variant,
                    anchor_idx=int(anchor.anchor_idx),
                )
                cache[(timeline_name, variant, int(anchor.anchor_idx))] = (x, info)
                rows.append(
                    _summarize_feature_info(
                        timeline=timeline_name,
                        modality_variant=variant,
                        anchor_idx=int(anchor.anchor_idx),
                        anchor_tb=int(anchor.anchor_tb),
                        anchor_phase=anchor.anchor_phase,
                        anchor_band=_anchor_band(int(anchor.anchor_idx), n_anchor_total),
                        x=x,
                        info=info,
                    )
                )

    single_vi_variants = [v for v in ["H_SINGLE_VI", "GH_SINGLE_VI"] if v in modality_variants]
    if single_vi_variants:
        n_anchor_total = int(len(anchors))
        resolved_vi_names = vi_names or [f"vi_{token.lower()}" for token in VEGETATION_INDEX_TOKENS]
        if available_single_vi_names_by_anchor is None:
            available_single_vi_names_by_anchor = _available_single_vi_names_by_anchor(
                bundle,
                anchors,
                anchor_indices=anchor_indices,
            )
        for anchor in anchors:
            anchor_idx = int(anchor.anchor_idx)
            if selected_anchor_indices is not None and anchor_idx not in selected_anchor_indices:
                continue
            anchor_available_vi = None
            if available_single_vi_names_by_anchor is not None:
                anchor_available_vi = available_single_vi_names_by_anchor.get(anchor_idx, set())
            for variant in single_vi_variants:
                for vi_name in resolved_vi_names:
                    if anchor_available_vi is not None and _normalize_vi_name(vi_name) not in anchor_available_vi:
                        continue
                    x, info = _build_result33_feature_matrix(
                        bundle=bundle,
                        anchors=anchors,
                        modality_variant=variant,
                        anchor_idx=anchor_idx,
                        vi_name=vi_name,
                    )
                    cache_key = (timeline_name, variant, anchor_idx, str(info["vi_name"]))
                    cache[cache_key] = (x, info)
                    rows.append(
                        _summarize_feature_info(
                            timeline=timeline_name,
                            modality_variant=variant,
                            anchor_idx=anchor_idx,
                            anchor_tb=int(anchor.anchor_tb),
                            anchor_phase=anchor.anchor_phase,
                            anchor_band=_anchor_band(anchor_idx, n_anchor_total),
                            x=x,
                            info=info,
                        )
                    )

    return cache, rows


def _runtime_tasks(
    *,
    timeline_dirs: dict,
    scenarios_cfg: dict,
    targets: list[str],
    active_predictors: list[str],
    modality_variants: list[str],
    vi_names: list[str],
    anchor_indices: list[int] | None,
    timeline_anchors: dict[str, list[AnchorDefinition]],
    prepared_split_cache: dict[tuple[str, str, str], dict],
    task_filter_set: set[tuple[str, str, str, str, str, int | None, str | None]],
    timeline_available_single_vi_names: dict[str, dict[int, set[str]]] | None = None,
) -> list[dict]:
    tasks: list[dict] = []
    for scenario_name in scenarios_cfg.keys():
        for timeline_name in timeline_dirs.keys():
            anchors = timeline_anchors[timeline_name]
            for target_col in targets:
                split_info = prepared_split_cache.get((scenario_name, timeline_name, target_col), {})
                if split_info.get("status") != "ok":
                    continue
                for predictor in active_predictors:
                    for modality_variant in modality_variants:
                        if modality_variant in {"H_SINGLE", "GH_SINGLE"}:
                            for anchor in anchors:
                                if anchor_indices is not None and int(anchor.anchor_idx) not in set(anchor_indices):
                                    continue
                                key = (
                                    scenario_name,
                                    timeline_name,
                                    target_col,
                                    predictor,
                                    modality_variant,
                                    int(anchor.anchor_idx),
                                    None,
                                )
                                if task_filter_set and key not in task_filter_set:
                                    continue
                                tasks.append(
                                    {
                                        "scenario": scenario_name,
                                        "timeline": timeline_name,
                                        "target": target_col,
                                        "predictor": predictor,
                                        "modality_variant": modality_variant,
                                        "anchor_idx": int(anchor.anchor_idx),
                                    }
                                )
                            continue

                        if modality_variant in {"H_SINGLE_VI", "GH_SINGLE_VI"}:
                            for anchor in anchors:
                                anchor_idx = int(anchor.anchor_idx)
                                if anchor_indices is not None and anchor_idx not in set(anchor_indices):
                                    continue
                                allowed_vi_names = None
                                if timeline_available_single_vi_names is not None:
                                    allowed_vi_names = timeline_available_single_vi_names.get(timeline_name, {}).get(anchor_idx, set())
                                for vi_name in vi_names:
                                    normalized_vi = _normalize_vi_name(vi_name)
                                    if allowed_vi_names is not None and normalized_vi not in allowed_vi_names:
                                        continue
                                    key = (
                                        scenario_name,
                                        timeline_name,
                                        target_col,
                                        predictor,
                                        modality_variant,
                                        anchor_idx,
                                        normalized_vi,
                                    )
                                    if task_filter_set and key not in task_filter_set:
                                        continue
                                    tasks.append(
                                        {
                                            "scenario": scenario_name,
                                            "timeline": timeline_name,
                                            "target": target_col,
                                            "predictor": predictor,
                                            "modality_variant": modality_variant,
                                            "anchor_idx": anchor_idx,
                                            "vi_name": normalized_vi,
                                        }
                                    )
                            continue

                        key = (scenario_name, timeline_name, target_col, predictor, modality_variant, None, None)
                        if task_filter_set and key not in task_filter_set:
                            continue
                        tasks.append(
                            {
                                "scenario": scenario_name,
                                "timeline": timeline_name,
                                "target": target_col,
                                "predictor": predictor,
                                "modality_variant": modality_variant,
                                "anchor_idx": None,
                            }
                        )
    return tasks


def _select_best_anchor_main_fold_rows(metrics_fold_df: pd.DataFrame, best_anchor_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_fold_df.empty or best_anchor_df.empty:
        return pd.DataFrame()
    out = metrics_fold_df.merge(
        best_anchor_df[["scenario", "target", "timeline", "predictor", "best_anchor", "best_anchor_tb"]],
        left_on=["scenario", "target", "timeline", "predictor", "anchor_idx"],
        right_on=["scenario", "target", "timeline", "predictor", "best_anchor"],
        how="inner",
    ).copy()
    out["ablation_group"] = "FULL"
    return out.drop(columns=["best_anchor"]).rename(columns={"best_anchor_tb": "best_anchor_tb_ref"})


def _run_factor_group_ablation(
    *,
    best_anchor_df: pd.DataFrame,
    timeline_bundles: dict[str, MultiTargetDataBundle],
    timeline_anchors: dict[str, list[AnchorDefinition]],
    feature_cache: dict[tuple[str, str, int | None], tuple[pd.DataFrame, dict]],
    prepared_split_cache: dict[tuple[str, str, str], dict],
    seed: int,
    optuna_cfg: dict,
    preprocess_cfg: dict,
    model_cfg: dict,
    progress_enabled: bool,
    output_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if best_anchor_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    metrics_rows: list[dict] = []
    oof_rows: list[dict] = []
    trial_rows: list[dict] = []
    total_jobs = int(len(best_anchor_df))
    for job_idx, (_, row) in enumerate(best_anchor_df.iterrows(), start=1):
        scenario_name = str(row["scenario"])
        target_col = str(row["target"])
        timeline_name = str(row["timeline"])
        predictor = str(row["predictor"])
        anchor_idx = int(row["best_anchor"])
        bundle = timeline_bundles[timeline_name]
        anchors = timeline_anchors[timeline_name]
        split_info = prepared_split_cache[(scenario_name, timeline_name, target_col)]
        x_full, info_full = feature_cache[(timeline_name, "GH_SINGLE", anchor_idx)]
        group_map = _group_h_columns_for_anchor(bundle, anchors, anchor_idx)
        if not group_map:
            continue

        _append_progress(
            output_dir,
            enabled=progress_enabled,
            event="factor_ablation_start",
            payload={
                "job_idx": job_idx,
                "total_jobs": total_jobs,
                "scenario": scenario_name,
                "target": target_col,
                "timeline": timeline_name,
                "predictor": predictor,
                "anchor_idx": anchor_idx,
            },
        )

        for factor_group, drop_cols in group_map.items():
            keep_cols = [c for c in x_full.columns if c not in set(drop_cols)]
            x_ablate = sanitize_feature_matrix(x_full.loc[:, keep_cols].copy())
            h_cols = [c for c in info_full.get("h_cols", []) if c not in set(drop_cols)]
            task_metrics, task_oof, task_trials = _run_single_task(
                x=x_ablate,
                y_full=split_info["y_full"],
                meta_full=bundle.meta,
                prepared_groups=split_info["prepared_groups"],
                scenario_name=scenario_name,
                timeline_name=timeline_name,
                target_col=target_col,
                predictor=predictor,
                modality_variant="GH_SINGLE",
                anchor_idx=anchor_idx,
                anchor_tb=int(row["best_anchor_tb"]) if pd.notna(row["best_anchor_tb"]) else None,
                anchor_phase=None,
                anchor_band=str(row.get("best_anchor_band", "mid")),
                vi_name=None,
                vi_names=None,
                n_features_h=len(h_cols),
                n_features_g=int(info_full.get("n_features_g", 0)),
                n_features_total=int(len(keep_cols)),
                seed=seed,
                optuna_cfg=optuna_cfg,
                preprocess_cfg=preprocess_cfg,
                model_cfg=model_cfg,
                ablation_group=factor_group,
            )
            metrics_rows.extend(task_metrics)
            oof_rows.extend(task_oof)
            trial_rows.extend(task_trials)

    return _to_records_df(metrics_rows), _to_records_df(oof_rows), _to_records_df(trial_rows)


def _select_top_anchor_main_fold_rows(metrics_fold_df: pd.DataFrame, top_anchor_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_fold_df.empty or top_anchor_df.empty:
        return pd.DataFrame()
    ref = top_anchor_df[
        ["scenario", "target", "timeline", "predictor", "anchor_idx", "anchor_tb"]
    ].drop_duplicates()
    out = metrics_fold_df.merge(
        ref,
        on=["scenario", "target", "timeline", "predictor", "anchor_idx", "anchor_tb"],
        how="inner",
    ).copy()
    out["ablation_group"] = "FULL"
    return out


def _run_index_ablation(
    *,
    selected_anchor_df: pd.DataFrame,
    timeline_bundles: dict[str, MultiTargetDataBundle],
    timeline_anchors: dict[str, list[AnchorDefinition]],
    feature_cache: dict[tuple[str, str, int | None], tuple[pd.DataFrame, dict]],
    prepared_split_cache: dict[tuple[str, str, str], dict],
    seed: int,
    optuna_cfg: dict,
    preprocess_cfg: dict,
    model_cfg: dict,
    progress_enabled: bool,
    output_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if selected_anchor_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    metrics_rows: list[dict] = []
    oof_rows: list[dict] = []
    trial_rows: list[dict] = []
    selected_ref = selected_anchor_df[
        ["scenario", "target", "timeline", "predictor", "anchor_idx", "anchor_tb", "anchor_phase", "anchor_band"]
    ].drop_duplicates()
    total_jobs = int(len(selected_ref))

    for job_idx, (_, row) in enumerate(selected_ref.iterrows(), start=1):
        scenario_name = str(row["scenario"])
        target_col = str(row["target"])
        timeline_name = str(row["timeline"])
        predictor = str(row["predictor"])
        anchor_idx = int(row["anchor_idx"])
        bundle = timeline_bundles[timeline_name]
        anchors = timeline_anchors[timeline_name]
        split_info = prepared_split_cache[(scenario_name, timeline_name, target_col)]
        x_full, info_full = feature_cache[(timeline_name, "GH_SINGLE", anchor_idx)]
        index_map = _single_h_columns_for_anchor(bundle, anchors, anchor_idx)
        if not index_map:
            continue

        _append_progress(
            output_dir,
            enabled=progress_enabled,
            event="index_ablation_start",
            payload={
                "job_idx": job_idx,
                "total_jobs": total_jobs,
                "scenario": scenario_name,
                "target": target_col,
                "timeline": timeline_name,
                "predictor": predictor,
                "anchor_idx": anchor_idx,
            },
        )

        for vi_name, drop_cols in index_map.items():
            keep_cols = [c for c in x_full.columns if c not in set(drop_cols)]
            x_ablate = sanitize_feature_matrix(x_full.loc[:, keep_cols].copy())
            h_cols = [c for c in info_full.get("h_cols", []) if c not in set(drop_cols)]
            task_metrics, task_oof, task_trials = _run_single_task(
                x=x_ablate,
                y_full=split_info["y_full"],
                meta_full=bundle.meta,
                prepared_groups=split_info["prepared_groups"],
                scenario_name=scenario_name,
                timeline_name=timeline_name,
                target_col=target_col,
                predictor=predictor,
                modality_variant="GH_SINGLE",
                anchor_idx=anchor_idx,
                anchor_tb=int(row["anchor_tb"]) if pd.notna(row["anchor_tb"]) else None,
                anchor_phase=str(row.get("anchor_phase")) if pd.notna(row.get("anchor_phase")) else None,
                anchor_band=str(row.get("anchor_band")) if pd.notna(row.get("anchor_band")) else None,
                vi_name=vi_name,
                vi_names=[vi_name],
                n_features_h=len(h_cols),
                n_features_g=int(info_full.get("n_features_g", 0)),
                n_features_total=int(len(keep_cols)),
                seed=seed,
                optuna_cfg=optuna_cfg,
                preprocess_cfg=preprocess_cfg,
                model_cfg=model_cfg,
                ablation_group=vi_name,
            )
            metrics_rows.extend(task_metrics)
            oof_rows.extend(task_oof)
            trial_rows.extend(task_trials)

    return _to_records_df(metrics_rows), _to_records_df(oof_rows), _to_records_df(trial_rows)


def _write_summary(
    *,
    output_dir: Path,
    cfg: dict,
    anchor_df: pd.DataFrame,
    feature_overview_df: pd.DataFrame,
    split_usage_df: pd.DataFrame,
    metrics_summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    best_anchor_df: pd.DataFrame,
    factor_importance_df: pd.DataFrame,
    top_positive_anchor_df: pd.DataFrame | None = None,
    index_importance_df: pd.DataFrame | None = None,
) -> None:
    lines = []
    lines.append("# Result3.3 Summary")
    lines.append("")
    lines.append("## 1. 实验定义")
    lines.append(f"- 时间轴: `{', '.join(_resolve_timeline_dirs(cfg.get('data', {})).keys())}`")
    lines.append(f"- 场景: `{', '.join(_resolve_scenarios_cfg(cfg.get('data', {})).keys())}`")
    lines.append(f"- 性状: `{', '.join(cfg.get('experiment', {}).get('targets', DEFAULT_TARGETS))}`")
    lines.append(f"- 模型: `{', '.join(cfg.get('experiment', {}).get('predictors_run', []))}`")
    lines.append(f"- 输入变体: `{', '.join(_resolve_modality_variants(cfg.get('experiment', {})))}`")
    lines.append("")
    lines.append("## 2. Anchor 预览")
    lines.append(markdown_table(anchor_df, max_rows=30))
    lines.append("")
    lines.append("## 3. 特征维度预览")
    lines.append(markdown_table(feature_overview_df, max_rows=30))
    lines.append("")
    lines.append("## 4. Split 使用概览")
    lines.append(markdown_table(split_usage_df, max_rows=30))
    lines.append("")
    lines.append("## 5. 主结果概览")
    lines.append(markdown_table(metrics_summary_df, max_rows=30))
    lines.append("")
    lines.append("## 6. 单窗口增量")
    lines.append(markdown_table(delta_df, max_rows=30))
    lines.append("")
    lines.append("## 7. 最佳 Anchor")
    lines.append(markdown_table(best_anchor_df, max_rows=30))
    lines.append("")
    lines.append("## 8. 因子组重要性")
    lines.append(markdown_table(factor_importance_df, max_rows=30))
    lines.append("")
    lines.append("## 9. Top Positive Anchors")
    lines.append(markdown_table(top_positive_anchor_df, max_rows=30) if top_positive_anchor_df is not None else "(empty)")
    lines.append("")
    lines.append("## 10. Index Importance")
    lines.append(markdown_table(index_importance_df, max_rows=30) if index_importance_df is not None else "(empty)")
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_result3_3(config_path: Path, *, dry_run: bool = False) -> Path | None:
    cfg = read_yaml(config_path)
    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})
    optuna_cfg = cfg.get("optuna", {})
    preprocess_cfg = cfg.get("preprocessing", {})
    output_cfg = cfg.get("output", {})

    seed = int(exp_cfg.get("random_seed", 42))
    _set_global_seed(seed)

    timeline_dirs = _resolve_timeline_dirs(data_cfg)
    scenarios_cfg = _resolve_scenarios_cfg(data_cfg)
    genotype_representation = str(exp_cfg.get("genotype_representation", "grm_pca"))
    targets = [str(x) for x in exp_cfg.get("targets", DEFAULT_TARGETS)]
    active_predictors = _resolve_active_predictors(exp_cfg)
    modality_variants = _resolve_modality_variants(exp_cfg)
    vi_names = _resolve_vi_names(exp_cfg)
    anchor_indices = _resolve_anchor_indices(exp_cfg) if isinstance(exp_cfg.get("anchor_indices"), list) and exp_cfg.get("anchor_indices") else None
    task_filter_set = _resolve_task_filter_set(exp_cfg)
    n_anchor_bins = int(exp_cfg.get("n_anchor_bins", 20))
    model_cfg = exp_cfg.get("model_backends", {})
    factor_ablation_enabled = bool(exp_cfg.get("factor_group_ablation", {}).get("enabled", True))
    index_ablation_cfg = exp_cfg.get("index_ablation", {})
    index_ablation_enabled = bool(index_ablation_cfg.get("enabled", False))
    top_positive_k = int(index_ablation_cfg.get("top_k", 5))

    output_dir = None if dry_run else _resolve_output_dir(output_cfg)
    progress_enabled = False if dry_run else bool(output_cfg.get("progress_log", True))
    _append_progress(output_dir, enabled=progress_enabled, event="run_start", payload={"config_path": str(config_path)})

    timeline_bundles: dict[str, MultiTargetDataBundle] = {}
    timeline_anchors: dict[str, list[AnchorDefinition]] = {}
    timeline_available_single_vi_names: dict[str, dict[int, set[str]]] = {}
    feature_cache: dict[tuple[str, str, int | None], tuple[pd.DataFrame, dict]] = {}
    anchor_rows: list[dict] = []
    feature_rows: list[dict] = []
    split_usage_rows: list[dict] = []
    prepared_split_cache: dict[tuple[str, str, str], dict] = {}

    for timeline_name, input_dir_str in timeline_dirs.items():
        bundle = load_multitarget_model_inputs(Path(str(input_dir_str)), genotype_representation=genotype_representation)
        anchors = build_anchor_definitions(bundle, n_anchor_bins=n_anchor_bins)
        timeline_bundles[timeline_name] = bundle
        timeline_anchors[timeline_name] = anchors
        timeline_available_single_vi_names[timeline_name] = _available_single_vi_names_by_anchor(
            bundle,
            anchors,
            anchor_indices=anchor_indices,
        )

        for anchor in anchors:
            anchor_rows.append(
                {
                    "timeline": timeline_name,
                    "anchor_idx": int(anchor.anchor_idx),
                    "anchor_tb": int(anchor.anchor_tb),
                    "anchor_phase": anchor.anchor_phase,
                    "anchor_tb_token": anchor.anchor_tb_token,
                    "anchor_band": _anchor_band(int(anchor.anchor_idx), len(anchors)),
                    "hyperspectral_static_col_count": int(anchor.hyperspectral_static_col_count),
                    "hyperspectral_prefix_col_count": int(anchor.hyperspectral_prefix_col_count),
                }
            )

        if not dry_run:
            cache, cache_rows = _build_feature_cache(
                timeline_name=timeline_name,
                bundle=bundle,
                anchors=anchors,
                modality_variants=modality_variants,
                vi_names=vi_names,
                anchor_indices=anchor_indices,
                available_single_vi_names_by_anchor=timeline_available_single_vi_names[timeline_name],
            )
            feature_cache.update(cache)
            feature_rows.extend(cache_rows)

        for scenario_name, scenario_cfg in scenarios_cfg.items():
            raw_split_groups = _resolve_split_groups_for_scenario(
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

    tasks = _runtime_tasks(
        timeline_dirs=timeline_dirs,
        scenarios_cfg=scenarios_cfg,
        targets=targets,
        active_predictors=active_predictors,
        modality_variants=modality_variants,
        vi_names=vi_names,
        anchor_indices=anchor_indices,
        timeline_anchors=timeline_anchors,
        prepared_split_cache=prepared_split_cache,
        task_filter_set=task_filter_set,
        timeline_available_single_vi_names=timeline_available_single_vi_names,
    )

    if dry_run:
        summary = {
            "config_path": str(config_path),
            "timeline_dirs": timeline_dirs,
            "scenarios": list(scenarios_cfg.keys()),
            "targets": targets,
            "predictors": active_predictors,
            "modality_variants": modality_variants,
            "vi_names": vi_names if any(v.endswith("_VI") for v in modality_variants) else [],
            "effective_anchor_count": {k: len(v) for k, v in timeline_anchors.items()},
            "n_tasks": len(tasks),
            "n_tasks_by_variant": {
                variant: int(sum(1 for t in tasks if t["modality_variant"] == variant))
                for variant in modality_variants
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return None

    if not tasks:
        raise RuntimeError("没有可执行的有效任务，请检查 split / target / timeline 配置。")

    metrics_rows: list[dict] = []
    oof_rows: list[dict] = []
    trial_rows: list[dict] = []

    total_tasks = int(len(tasks))
    for task_idx, task in enumerate(tasks, start=1):
        scenario_name = str(task["scenario"])
        timeline_name = str(task["timeline"])
        target_col = str(task["target"])
        predictor = str(task["predictor"])
        modality_variant = str(task["modality_variant"])
        anchor_idx = task.get("anchor_idx")
        vi_name = task.get("vi_name")

        split_info = prepared_split_cache[(scenario_name, timeline_name, target_col)]
        bundle = timeline_bundles[timeline_name]
        anchors = timeline_anchors[timeline_name]
        cache_key = (
            (timeline_name, modality_variant, anchor_idx, vi_name)
            if vi_name is not None
            else (timeline_name, modality_variant, anchor_idx)
        )
        x, info = feature_cache[cache_key]
        anchor = anchors[int(anchor_idx)] if anchor_idx is not None else (anchors[-1] if modality_variant == "GH_FULL" and anchors else None)
        anchor_tb = int(anchor.anchor_tb) if anchor is not None else None
        anchor_phase = anchor.anchor_phase if anchor is not None else None
        anchor_band = _anchor_band(int(anchor_idx), len(anchors)) if anchor_idx is not None else ("full_season" if modality_variant == "GH_FULL" else None)

        print(
            f"[TASK {task_idx}/{total_tasks}] scenario={scenario_name} timeline={timeline_name} target={target_col} predictor={predictor} variant={modality_variant} anchor={anchor_idx if anchor_idx is not None else 'NA'} vi={vi_name if vi_name is not None else 'NA'} features={x.shape[1]}",
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
            anchor_idx=int(anchor_idx) if anchor_idx is not None else None,
            anchor_tb=anchor_tb,
            anchor_phase=anchor_phase,
            anchor_band=anchor_band,
            vi_name=str(info.get("vi_name")) if info.get("vi_name") else None,
            vi_names=list(info.get("vi_names", [])) if info.get("vi_names") else None,
            n_features_h=int(info.get("n_features_h", 0)),
            n_features_g=int(info.get("n_features_g", 0)),
            n_features_total=int(info.get("n_features_total", x.shape[1])),
            seed=seed,
            optuna_cfg=optuna_cfg,
            preprocess_cfg=preprocess_cfg,
            model_cfg=model_cfg,
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

    anchor_df = _to_records_df(anchor_rows)
    feature_overview_df = _to_records_df(feature_rows)
    split_usage_df = _to_records_df(split_usage_rows)
    metrics_fold_df = _to_records_df(metrics_rows)
    oof_df = _to_records_df(oof_rows)
    trial_df = _to_records_df(trial_rows)
    metrics_summary_df = _build_metrics_summary(metrics_fold_df)
    metrics_by_year_df = _build_metrics_by_year(oof_df)
    single_anchor_delta_df, best_anchor_df = _build_best_anchor_summary(metrics_summary_df)
    top_positive_anchor_df = _build_top_positive_anchor_summary(metrics_summary_df, top_k=top_positive_k)
    single_vi_transferability_df = _build_single_vi_transferability_summary(metrics_summary_df)
    g_conditioned_single_vi_df = _build_g_conditioned_single_vi_summary(metrics_summary_df)

    factor_fold_df = pd.DataFrame()
    factor_oof_df = pd.DataFrame()
    factor_trial_df = pd.DataFrame()
    factor_metrics_summary_df = pd.DataFrame()
    factor_importance_df = pd.DataFrame()
    index_fold_df = pd.DataFrame()
    index_oof_df = pd.DataFrame()
    index_trial_df = pd.DataFrame()
    index_metrics_summary_df = pd.DataFrame()
    index_importance_df = pd.DataFrame()
    if factor_ablation_enabled and not best_anchor_df.empty:
        factor_fold_df, factor_oof_df, factor_trial_df = _run_factor_group_ablation(
            best_anchor_df=best_anchor_df,
            timeline_bundles=timeline_bundles,
            timeline_anchors=timeline_anchors,
            feature_cache=feature_cache,
            prepared_split_cache=prepared_split_cache,
            seed=seed,
            optuna_cfg=optuna_cfg,
            preprocess_cfg=preprocess_cfg,
            model_cfg=model_cfg,
            progress_enabled=progress_enabled,
            output_dir=output_dir,
        )
        base_factor_fold_df = _select_best_anchor_main_fold_rows(metrics_fold_df, best_anchor_df)
        factor_fold_all_df = pd.concat([base_factor_fold_df, factor_fold_df], ignore_index=True) if not factor_fold_df.empty else base_factor_fold_df
        factor_metrics_summary_df = _build_metrics_summary(factor_fold_all_df)
        factor_importance_df = _build_factor_group_importance(factor_metrics_summary_df, best_anchor_df)
        if not factor_fold_all_df.empty:
            factor_fold_all_df.to_csv(output_dir / "factor_group_metrics_by_fold.csv", index=False)
        if not factor_metrics_summary_df.empty:
            factor_metrics_summary_df.to_csv(output_dir / "factor_group_metrics_summary.csv", index=False)
        if not factor_oof_df.empty:
            factor_oof_df.to_parquet(output_dir / "factor_group_oof_predictions.parquet", index=False)
        if not factor_trial_df.empty:
            factor_trial_df.to_csv(output_dir / "factor_group_optuna_trials.csv", index=False)
        if not factor_importance_df.empty:
            factor_importance_df.to_csv(output_dir / "factor_group_importance.csv", index=False)

    if index_ablation_enabled and not top_positive_anchor_df.empty:
        index_fold_df, index_oof_df, index_trial_df = _run_index_ablation(
            selected_anchor_df=top_positive_anchor_df,
            timeline_bundles=timeline_bundles,
            timeline_anchors=timeline_anchors,
            feature_cache=feature_cache,
            prepared_split_cache=prepared_split_cache,
            seed=seed,
            optuna_cfg=optuna_cfg,
            preprocess_cfg=preprocess_cfg,
            model_cfg=model_cfg,
            progress_enabled=progress_enabled,
            output_dir=output_dir,
        )
        base_index_fold_df = _select_top_anchor_main_fold_rows(metrics_fold_df, top_positive_anchor_df)
        index_fold_all_df = pd.concat([base_index_fold_df, index_fold_df], ignore_index=True) if not index_fold_df.empty else base_index_fold_df
        index_metrics_summary_df = _build_metrics_summary(index_fold_all_df)
        index_importance_df = _build_index_importance(index_metrics_summary_df, top_positive_anchor_df)
        if not top_positive_anchor_df.empty:
            top_positive_anchor_df.to_csv(output_dir / "top_positive_anchors.csv", index=False)
        if not index_fold_all_df.empty:
            index_fold_all_df.to_csv(output_dir / "index_ablation_metrics_by_fold.csv", index=False)
        if not index_metrics_summary_df.empty:
            index_metrics_summary_df.to_csv(output_dir / "index_ablation_metrics_summary.csv", index=False)
        if not index_oof_df.empty:
            index_oof_df.to_parquet(output_dir / "index_ablation_oof_predictions.parquet", index=False)
        if not index_trial_df.empty:
            index_trial_df.to_csv(output_dir / "index_ablation_optuna_trials.csv", index=False)
        if not index_importance_df.empty:
            index_importance_df.to_csv(output_dir / "index_importance.csv", index=False)

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
    if not oof_df.empty:
        oof_df.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    if not trial_df.empty:
        trial_df.to_csv(output_dir / "optuna_trials.csv", index=False)

    config_snapshot = {
        "config_path": str(config_path),
        "git_commit_hash": get_git_commit_hash(Path.cwd()),
        "generated_at": now_iso(),
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
            "modality_variants": modality_variants,
            "n_anchor_bins_effective": {k: len(v) for k, v in timeline_anchors.items()},
        },
        "optuna": optuna_cfg,
        "preprocessing": preprocess_cfg,
        "output_dir": output_dir.as_posix(),
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
        factor_importance_df=factor_importance_df,
        top_positive_anchor_df=top_positive_anchor_df,
        index_importance_df=index_importance_df,
    )

    _append_progress(output_dir, enabled=progress_enabled, event="run_end", payload={"output_dir": output_dir.as_posix()})
    print(f"Result3.3 completed: {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Result3.3 single-anchor multitrait experiments.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_result3_3(Path(args.config), dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
