from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from models.common.io_utils import read_yaml, write_yaml


DEFAULT_TARGETS = ["ActualYD", "CM", "LM", "PHM", "Spike", "TKW"]
DEFAULT_PREDICTORS_5 = ["ridge", "lasso", "elasticnet", "lightgbm", "random_forest"]
DEFAULT_VI_NAMES = [
    "vi_evi2",
    "vi_gndvi",
    "vi_grvi",
    "vi_mgrvi",
    "vi_msavi",
    "vi_msr",
    "vi_ndre",
    "vi_ndvi",
    "vi_osavi",
    "vi_rdvi",
    "vi_savi",
    "vi_vari",
]


def default_four_scenario_cfg() -> dict[str, dict[str, str]]:
    fold_map = "outputs/reports/loso_genotype_nested_genotype_fold_map.csv"
    return {
        "reference": {
            "custom_split_strategy": "reference",
            "genotype_fold_map_path": fold_map,
        },
        "within_season": {
            "custom_split_strategy": "within_season_known_year_unknown_genotype",
            "genotype_fold_map_path": fold_map,
        },
        "loso": {
            "custom_split_strategy": "loso_known_genotype",
            "genotype_fold_map_path": fold_map,
        },
        "loso_genotype": {
            "custom_split_strategy": "loso_genotype_unknown",
            "genotype_fold_map_path": fold_map,
        },
    }


def _default_timeline_dir(data_cfg: dict[str, Any]) -> str:
    timeline_dirs = data_cfg.get("timeline_dirs", {})
    if isinstance(timeline_dirs, dict) and timeline_dirs.get("gdd_rel_heading"):
        return str(timeline_dirs["gdd_rel_heading"])
    if data_cfg.get("input_dir"):
        return str(data_cfg["input_dir"])
    return "data/processed/model_inputs_engineered/gdd_rel_heading"


def _normalize_base_cfg(base_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg.setdefault("data", {})
    cfg.setdefault("experiment", {})
    cfg.setdefault("optuna", {})
    cfg.setdefault("preprocessing", {})
    cfg.setdefault("output", {})
    cfg["data"]["timeline_dirs"] = {"gdd_rel_heading": _default_timeline_dir(cfg["data"])}
    cfg["data"]["scenarios"] = default_four_scenario_cfg()
    cfg["experiment"]["targets"] = [str(x) for x in cfg["experiment"].get("targets", DEFAULT_TARGETS)]
    cfg["experiment"]["genotype_representation"] = str(cfg["experiment"].get("genotype_representation", "grm_pca"))
    cfg["output"].setdefault("append_timestamp", True)
    cfg["output"].setdefault("progress_log", True)
    return cfg


def build_result31_section_cfg(
    *,
    base_cfg: dict[str, Any],
    modality_combo: str,
    output_dir: str = "",
) -> dict[str, Any]:
    cfg = _normalize_base_cfg(base_cfg)
    exp_cfg = cfg["experiment"]
    exp_cfg["modality_combo"] = str(modality_combo)
    exp_cfg.pop("modality_combos", None)
    exp_cfg["predictors_supported"] = list(DEFAULT_PREDICTORS_5)
    exp_cfg["predictors_run"] = list(DEFAULT_PREDICTORS_5)
    if output_dir:
        cfg["output"]["output_dir_base"] = str(output_dir)
        cfg["output"]["append_timestamp"] = False
        cfg["output"]["allow_overwrite"] = True
    return cfg


def build_result32a_section_cfg(
    *,
    base_cfg: dict[str, Any],
    output_dir: str = "",
) -> dict[str, Any]:
    cfg = _normalize_base_cfg(base_cfg)
    exp_cfg = cfg["experiment"]
    exp_cfg["predictors_supported"] = list(DEFAULT_PREDICTORS_5)
    exp_cfg["predictors_run"] = list(DEFAULT_PREDICTORS_5)
    exp_cfg["modality_variants"] = ["G", "GH_SINGLE"]
    exp_cfg["n_anchor_bins"] = int(exp_cfg.get("n_anchor_bins", 20))
    exp_cfg["factor_group_ablation"] = {"enabled": True}
    exp_cfg["index_ablation"] = {"enabled": False}
    if output_dir:
        cfg["output"]["output_dir_base"] = str(output_dir)
        cfg["output"]["append_timestamp"] = False
        cfg["output"]["allow_overwrite"] = True
    return cfg


def build_result32bc_section_cfg(
    *,
    base_cfg: dict[str, Any],
    best_anchor_csv: Path,
    output_dir: str = "",
) -> dict[str, Any]:
    cfg = _normalize_base_cfg(base_cfg)
    exp_cfg = cfg["experiment"]
    requested_predictors = [str(x).lower() for x in exp_cfg.get("predictors_run", [])]
    exp_cfg["predictors_supported"] = list(DEFAULT_PREDICTORS_5)
    exp_cfg["predictors_run"] = requested_predictors or list(DEFAULT_PREDICTORS_5)
    exp_cfg["vi_names"] = [str(x) for x in exp_cfg.get("vi_names", DEFAULT_VI_NAMES)]
    exp_cfg["best_anchor_csv"] = str(best_anchor_csv)
    exp_cfg["n_anchor_bins"] = int(exp_cfg.get("n_anchor_bins", 20))
    if output_dir:
        cfg["output"]["output_dir_base"] = str(output_dir)
        cfg["output"]["append_timestamp"] = False
        cfg["output"]["allow_overwrite"] = True
    return cfg


def read_base_config(config_path: Path) -> dict[str, Any]:
    return read_yaml(config_path)


def write_resolved_config(
    *,
    cfg: dict[str, Any],
    section_code: str,
    repo_root: Path,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = repo_root / "outputs" / "tmp_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{section_code.lower().replace('.', '_')}_resolved_{stamp}.yaml"
    write_yaml(cfg, out_path)
    return out_path
