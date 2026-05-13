from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from crophg.common.report_utils import write_json


@dataclass(frozen=True)
class WithinSeasonSplit:
    split_name: str
    target_year: int
    outer_fold: int
    inner_fold: int
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]
    random_seed: int | None


SCENARIO_ORDER = ["reference", "within_season", "loso", "loso_genotype"]
SCENARIO_STRATEGY_ALIASES = {
    "reference": {"reference", "known_year_known_genotype", "reference_known_year_known_genotype"},
    "within_season": {"within_season_known_year_unknown_genotype", "known_year_unknown_genotype", "within_season_grouped_genotype"},
    "loso": {"loso_known_genotype", "unknown_year_known_genotype", "loso_cellwise"},
    "loso_genotype": {"loso_genotype_unknown", "unknown_year_unknown_genotype", "loso_genotype_cellwise"},
}


def _read_table(path: Path, file_type: str | None = None) -> pd.DataFrame:
    kind = (file_type or path.suffix.lstrip(".")).lower()
    if kind == "parquet":
        return pd.read_parquet(path)
    if kind == "csv":
        return pd.read_csv(path)
    if kind in {"xlsx", "xls"}:
        return pd.read_excel(path)
    raise ValueError(f"不支持的文件类型: {path}")


def _normalize_text(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _ensure_disjoint(*, train_ids: list[str], val_ids: list[str], test_ids: list[str], split_name: str) -> None:
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    if train_set & val_set:
        raise ValueError(f"{split_name}: train_ids 与 val_ids 存在交集。")
    if train_set & test_set:
        raise ValueError(f"{split_name}: train_ids 与 test_ids 存在交集。")
    if val_set & test_set:
        raise ValueError(f"{split_name}: val_ids 与 test_ids 存在交集。")


def _load_genotype_fold_map(path: Path) -> dict[str, int]:
    df = pd.read_csv(path)
    missing = {"genotype_id", "genotype_fold"} - set(df.columns)
    if missing:
        raise KeyError(f"genotype fold map 缺少列: {sorted(missing)}")
    out = {str(row["genotype_id"]): int(row["genotype_fold"]) for _, row in df.iterrows()}
    if not out:
        raise ValueError(f"genotype fold map 为空: {path}")
    return out


def _prepare_meta_df(
    sample_df: pd.DataFrame,
    *,
    required_columns: list[str],
    sample_id_column: str,
) -> pd.DataFrame:
    missing = [c for c in required_columns if c not in sample_df.columns]
    if missing:
        raise KeyError(f"输入样本索引缺少必需列: {missing}")
    if sample_id_column not in sample_df.columns:
        raise KeyError(f"输入样本索引缺少 sample_id_column: {sample_id_column}")

    keep_cols = list(dict.fromkeys(required_columns + [sample_id_column]))
    out = sample_df.loc[:, keep_cols].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["sample_id"] = out[sample_id_column].map(_normalize_text)
    if "plot_id" not in out.columns:
        out["plot_id"] = out["sample_id"]
    else:
        out["plot_id"] = out["plot_id"].map(_normalize_text)
    out["genotype_id"] = out["genotype_id"].map(_normalize_text)
    out = out.dropna(subset=["year", "plot_id", "genotype_id"]).copy()
    if not out["sample_id"].is_unique:
        dup_cnt = int(out["sample_id"].duplicated().sum())
        raise ValueError(f"sample_id 非唯一，重复数量={dup_cnt}")
    return out.reset_index(drop=True)


def _meta_with_genotype_fold(meta_df: pd.DataFrame, genotype_fold_map: dict[str, int]) -> pd.DataFrame:
    work = meta_df.copy()
    if "sample_id" not in work.columns:
        work["sample_id"] = work["plot_id"].astype(str)
    else:
        work["sample_id"] = work["sample_id"].astype(str)
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work["genotype_id"] = work["genotype_id"].astype(str)
    work["genotype_fold"] = work["genotype_id"].map(genotype_fold_map).astype("Int64")
    if work["genotype_fold"].isna().any():
        missing = sorted(work.loc[work["genotype_fold"].isna(), "genotype_id"].astype(str).unique().tolist())[:10]
        raise ValueError(f"存在无法映射到 genotype_fold 的 genotype_id，例如: {missing}")
    return work


def _build_reference_split_groups(
    meta_df: pd.DataFrame,
    *,
    genotype_fold_map: dict[str, int],
    random_seed: int,
) -> dict[tuple[int, int], list[WithinSeasonSplit]]:
    work = _meta_with_genotype_fold(meta_df, genotype_fold_map)
    years = sorted(work["year"].dropna().astype(int).unique().tolist())
    folds = sorted(work["genotype_fold"].dropna().astype(int).unique().tolist())
    groups: dict[tuple[int, int], list[WithinSeasonSplit]] = {}

    for year in years:
        year_df = work.loc[work["year"] == int(year)].copy()
        for outer_fold in folds:
            test_ids = year_df.loc[year_df["genotype_fold"] == int(outer_fold), "sample_id"].astype(str).tolist()
            if not test_ids:
                continue
            if int(year) == int(years[0]) and int(outer_fold) == int(folds[0]):
                val_folds = [int(f) for f in folds if int(f) != int(outer_fold)]
            else:
                val_folds = [int(folds[(folds.index(int(outer_fold)) + 1) % len(folds)])]

            rows: list[WithinSeasonSplit] = []
            for inner_fold, val_fold in enumerate(val_folds):
                val_ids = year_df.loc[year_df["genotype_fold"] == int(val_fold), "sample_id"].astype(str).tolist()
                train_mask = ~(
                    ((work["year"] == int(year)) & (work["genotype_fold"] == int(outer_fold)))
                    | ((work["year"] == int(year)) & (work["genotype_fold"] == int(val_fold)))
                )
                train_ids = work.loc[train_mask, "sample_id"].astype(str).tolist()
                split_name = f"reference_{year}_outer{outer_fold}_inner{inner_fold}"
                rows.append(WithinSeasonSplit(
                    split_name=split_name,
                    target_year=int(year),
                    outer_fold=int(outer_fold),
                    inner_fold=int(inner_fold),
                    train_ids=train_ids,
                    val_ids=val_ids,
                    test_ids=test_ids,
                    random_seed=int(random_seed),
                ))
            groups[(int(year), int(outer_fold))] = rows
    return groups


def _build_known_year_unknown_genotype_split_groups(
    meta_df: pd.DataFrame,
    *,
    genotype_fold_map: dict[str, int],
    random_seed: int,
) -> dict[tuple[int, int], list[WithinSeasonSplit]]:
    work = _meta_with_genotype_fold(meta_df, genotype_fold_map)
    years = sorted(work["year"].dropna().astype(int).unique().tolist())
    folds = sorted(work["genotype_fold"].dropna().astype(int).unique().tolist())
    groups: dict[tuple[int, int], list[WithinSeasonSplit]] = {}

    for year in years:
        year_df = work.loc[work["year"] == int(year)].copy()
        for outer_fold in folds:
            test_ids = year_df.loc[year_df["genotype_fold"] == int(outer_fold), "sample_id"].astype(str).tolist()
            if not test_ids:
                continue
            if int(year) == int(years[0]) and int(outer_fold) == int(folds[0]):
                val_folds = [int(f) for f in folds if int(f) != int(outer_fold)]
            else:
                val_folds = [int(folds[(folds.index(int(outer_fold)) + 1) % len(folds)])]

            rows: list[WithinSeasonSplit] = []
            for inner_fold, val_fold in enumerate(val_folds):
                val_ids = work.loc[work["genotype_fold"] == int(val_fold), "sample_id"].astype(str).tolist()
                train_mask = (
                    (work["genotype_fold"] != int(outer_fold))
                    & (work["genotype_fold"] != int(val_fold))
                )
                train_ids = work.loc[train_mask, "sample_id"].astype(str).tolist()
                split_name = f"within_season_known_year_{year}_outer{outer_fold}_inner{inner_fold}"
                rows.append(WithinSeasonSplit(
                    split_name=split_name,
                    target_year=int(year),
                    outer_fold=int(outer_fold),
                    inner_fold=int(inner_fold),
                    train_ids=train_ids,
                    val_ids=val_ids,
                    test_ids=test_ids,
                    random_seed=int(random_seed),
                ))
            groups[(int(year), int(outer_fold))] = rows
    return groups


def _build_unknown_year_known_genotype_split_groups(
    meta_df: pd.DataFrame,
    *,
    genotype_fold_map: dict[str, int],
    random_seed: int,
    inner_validation_policy: str = "default",
) -> dict[tuple[int, int], list[WithinSeasonSplit]]:
    work = _meta_with_genotype_fold(meta_df, genotype_fold_map)
    years = sorted(work["year"].dropna().astype(int).unique().tolist())
    folds = sorted(work["genotype_fold"].dropna().astype(int).unique().tolist())
    groups: dict[tuple[int, int], list[WithinSeasonSplit]] = {}
    policy = str(inner_validation_policy or "default").strip().lower()

    for year in years:
        remaining_years = [y for y in years if int(y) != int(year)]
        if len(remaining_years) != 2:
            raise ValueError("当前 loso 自定义划分要求恰有 3 个年份。")
        for outer_fold in folds:
            test_ids = work.loc[
                (work["year"] == int(year)) & (work["genotype_fold"] == int(outer_fold)),
                "sample_id",
            ].astype(str).tolist()
            if not test_ids:
                continue
            is_source_outer = int(year) == int(years[0]) and int(outer_fold) == int(folds[0])
            use_ood_cell = policy in {"ood_cell", "year_cell_ood", "ood_cell_first_outer"} and is_source_outer
            if use_ood_cell:
                val_cells = [(int(val_year), int(val_fold)) for val_year in remaining_years for val_fold in folds]
            elif is_source_outer:
                val_cells = [(int(val_year), int(val_fold)) for val_year in remaining_years for val_fold in folds]
            else:
                val_cells = [(int(remaining_years[0]), int(outer_fold))]

            rows: list[WithinSeasonSplit] = []
            for inner_fold, (val_year, val_fold) in enumerate(val_cells):
                val_ids = work.loc[
                    (work["year"] == int(val_year)) & (work["genotype_fold"] == int(val_fold)),
                    "sample_id",
                ].astype(str).tolist()
                if use_ood_cell:
                    train_mask = (work["year"] != int(year)) & (work["year"] != int(val_year))
                else:
                    train_mask = (work["year"] != int(year)) & ~(
                        (work["year"] == int(val_year)) & (work["genotype_fold"] == int(val_fold))
                    )
                train_ids = work.loc[train_mask, "sample_id"].astype(str).tolist()
                split_name = f"loso_known_genotype_{year}_outer{outer_fold}_inner{inner_fold}"
                rows.append(WithinSeasonSplit(
                    split_name=split_name,
                    target_year=int(year),
                    outer_fold=int(outer_fold),
                    inner_fold=int(inner_fold),
                    train_ids=train_ids,
                    val_ids=val_ids,
                    test_ids=test_ids,
                    random_seed=int(random_seed),
                ))
            groups[(int(year), int(outer_fold))] = rows
    return groups


def _build_unknown_year_unknown_genotype_split_groups(
    meta_df: pd.DataFrame,
    *,
    genotype_fold_map: dict[str, int],
    random_seed: int,
    inner_validation_policy: str = "default",
) -> dict[tuple[int, int], list[WithinSeasonSplit]]:
    work = _meta_with_genotype_fold(meta_df, genotype_fold_map)
    years = sorted(work["year"].dropna().astype(int).unique().tolist())
    folds = sorted(work["genotype_fold"].dropna().astype(int).unique().tolist())
    groups: dict[tuple[int, int], list[WithinSeasonSplit]] = {}
    policy = str(inner_validation_policy or "default").strip().lower()

    for year in years:
        remaining_years = [y for y in years if int(y) != int(year)]
        if len(remaining_years) != 2:
            raise ValueError("当前 loso_genotype 自定义划分要求恰有 3 个年份。")
        for outer_fold in folds:
            test_ids = work.loc[
                (work["year"] == int(year)) & (work["genotype_fold"] == int(outer_fold)),
                "sample_id",
            ].astype(str).tolist()
            if not test_ids:
                continue
            is_source_outer = int(year) == int(years[0]) and int(outer_fold) == int(folds[0])
            use_ood_cell = policy in {"ood_cell", "year_genotype_cell_ood", "ood_cell_first_outer"} and is_source_outer
            if use_ood_cell:
                val_cells = [
                    (int(val_year), int(val_fold))
                    for val_year in remaining_years
                    for val_fold in folds
                    if int(val_fold) != int(outer_fold)
                ]
            elif is_source_outer:
                val_folds = [int(f) for f in folds if int(f) != int(outer_fold)]
                val_cells = [(None, int(val_fold)) for val_fold in val_folds]
            else:
                val_folds = [int(folds[(folds.index(int(outer_fold)) + 1) % len(folds)])]
                val_cells = [(None, int(val_folds[0]))]

            rows: list[WithinSeasonSplit] = []
            for inner_fold, (val_year, val_fold) in enumerate(val_cells):
                if int(val_fold) == int(outer_fold):
                    raise ValueError("val_fold 不能与 outer_fold 相同。")
                if use_ood_cell:
                    val_ids = work.loc[
                        (work["year"] == int(val_year)) & (work["genotype_fold"] == int(val_fold)),
                        "sample_id",
                    ].astype(str).tolist()
                    train_mask = (
                        (work["year"] != int(year))
                        & (work["year"] != int(val_year))
                        & (work["genotype_fold"] != int(outer_fold))
                        & (work["genotype_fold"] != int(val_fold))
                    )
                else:
                    val_ids = work.loc[
                        (work["year"].isin(remaining_years)) & (work["genotype_fold"] == int(val_fold)),
                        "sample_id",
                    ].astype(str).tolist()
                    train_mask = (
                        (work["year"] != int(year))
                        & (work["genotype_fold"] != int(outer_fold))
                        & (work["genotype_fold"] != int(val_fold))
                    )
                train_ids = work.loc[train_mask, "sample_id"].astype(str).tolist()
                split_name = f"loso_genotype_unknown_{year}_outer{outer_fold}_inner{inner_fold}"
                rows.append(WithinSeasonSplit(
                    split_name=split_name,
                    target_year=int(year),
                    outer_fold=int(outer_fold),
                    inner_fold=int(inner_fold),
                    train_ids=train_ids,
                    val_ids=val_ids,
                    test_ids=test_ids,
                    random_seed=int(random_seed),
                ))
            groups[(int(year), int(outer_fold))] = rows
    return groups


def load_split_groups(split_dir: Path, validation_scenario: str) -> dict[tuple[int, int], list[WithinSeasonSplit]]:
    scenario = str(validation_scenario).strip().lower().replace("-", "_")
    if scenario == "within_season":
        grouped: dict[tuple[int, int], list[WithinSeasonSplit]] = {}
        for path in sorted(split_dir.glob("within_season_*.json")):
            obj = json.loads(path.read_text(encoding="utf-8"))
            split = WithinSeasonSplit(
                split_name=str(obj.get("split_name", path.stem)),
                target_year=int(obj.get("target_year")),
                outer_fold=int(obj.get("outer_fold")),
                inner_fold=int(obj.get("inner_fold")),
                train_ids=[str(x) for x in obj.get("train_ids", [])],
                val_ids=[str(x) for x in obj.get("val_ids", [])],
                test_ids=[str(x) for x in obj.get("test_ids", [])],
                random_seed=obj.get("random_seed"),
            )
            grouped.setdefault((split.target_year, split.outer_fold), []).append(split)
        return {k: sorted(v, key=lambda x: x.inner_fold) for k, v in sorted(grouped.items(), key=lambda item: item[0])}
    if scenario in {"loso", "loso_genotype"}:
        grouped: dict[tuple[int, int], list[WithinSeasonSplit]] = {}
        pattern = "loso_genotype*.json" if scenario == "loso_genotype" else "loso_known_genotype*.json"
        for path in sorted(split_dir.glob(pattern)):
            obj = json.loads(path.read_text(encoding="utf-8"))
            split_name = str(obj.get("split_name", path.stem))
            target_year = int(obj.get("target_year"))
            outer_fold = int(obj.get("outer_fold", 0))
            inner_fold = int(obj.get("inner_fold", obj.get("fold", 0)))
            split = WithinSeasonSplit(
                split_name=split_name,
                target_year=target_year,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                train_ids=[str(x) for x in obj.get("train_ids", [])],
                val_ids=[str(x) for x in obj.get("val_ids", [])],
                test_ids=[str(x) for x in obj.get("test_ids", [])],
                random_seed=obj.get("random_seed"),
            )
            grouped.setdefault((split.target_year, split.outer_fold), []).append(split)
        return {k: sorted(v, key=lambda x: x.inner_fold) for k, v in sorted(grouped.items(), key=lambda item: item[0])}
    raise ValueError(f"不支持的 validation_scenario: {validation_scenario}")


def build_split_groups_for_scenario(
    *,
    scenario_name: str,
    scenario_cfg: dict,
    meta_df: pd.DataFrame,
    random_seed: int,
    repo_root: Path,
) -> dict[tuple[int, int], list[WithinSeasonSplit]]:
    strategy = str(scenario_cfg.get("custom_split_strategy", "")).strip().lower()
    if not strategy:
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
        fold_map_path = (repo_root / fold_map_path).resolve()
    genotype_fold_map = _load_genotype_fold_map(fold_map_path)

    if strategy in SCENARIO_STRATEGY_ALIASES["reference"]:
        return _build_reference_split_groups(meta_df, genotype_fold_map=genotype_fold_map, random_seed=random_seed)
    if strategy in SCENARIO_STRATEGY_ALIASES["within_season"]:
        return _build_known_year_unknown_genotype_split_groups(
            meta_df,
            genotype_fold_map=genotype_fold_map,
            random_seed=random_seed,
        )
    if strategy in SCENARIO_STRATEGY_ALIASES["loso"]:
        return _build_unknown_year_known_genotype_split_groups(
            meta_df,
            genotype_fold_map=genotype_fold_map,
            random_seed=random_seed,
            inner_validation_policy=str(scenario_cfg.get("inner_validation_policy", "default")),
        )
    if strategy in SCENARIO_STRATEGY_ALIASES["loso_genotype"]:
        return _build_unknown_year_unknown_genotype_split_groups(
            meta_df,
            genotype_fold_map=genotype_fold_map,
            random_seed=random_seed,
            inner_validation_policy=str(scenario_cfg.get("inner_validation_policy", "default")),
        )
    raise ValueError(f"不支持的 custom_split_strategy: {strategy}")


def _year_distribution(sample_df: pd.DataFrame) -> dict[str, int]:
    counts = (
        pd.to_numeric(sample_df["year"], errors="coerce")
        .astype("Int64")
        .value_counts(dropna=True)
        .sort_index()
        .to_dict()
    )
    return {str(int(k)): int(v) for k, v in counts.items()}


def _genotype_overlap_counts(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, int]:
    train_g = set(train_df["genotype_id"].dropna().astype(str).tolist())
    val_g = set(val_df["genotype_id"].dropna().astype(str).tolist())
    test_g = set(test_df["genotype_id"].dropna().astype(str).tolist())
    return {
        "train_vs_val": int(len(train_g & val_g)),
        "train_vs_test": int(len(train_g & test_g)),
        "val_vs_test": int(len(val_g & test_g)),
    }


def _subset_by_ids(df: pd.DataFrame, sample_ids: list[str]) -> pd.DataFrame:
    return df[df["sample_id"].isin(sample_ids)].copy()


def _build_stats_entry(
    *,
    split_name: str,
    split_type: str,
    target_year: int,
    outer_fold: int | None,
    inner_fold: int | None,
    fold: int | None,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    all_df: pd.DataFrame,
) -> dict[str, object]:
    train_df = _subset_by_ids(all_df, train_ids)
    val_df = _subset_by_ids(all_df, val_ids)
    test_df = _subset_by_ids(all_df, test_ids)
    return {
        "split_name": split_name,
        "split_type": split_type,
        "target_year": int(target_year),
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "fold": fold,
        "n_train": int(len(train_ids)),
        "n_val": int(len(val_ids)),
        "n_test": int(len(test_ids)),
        "year_distribution": {
            "train": _year_distribution(train_df),
            "val": _year_distribution(val_df),
            "test": _year_distribution(test_df),
        },
        "genotype_overlap": _genotype_overlap_counts(train_df, val_df, test_df),
    }


def _render_summary_markdown(
    *,
    generated_at: str,
    config_path: Path,
    split_dir: Path,
    total_split_count: int,
    scenario_counts: dict[str, int],
    stats_rows: list[dict[str, object]],
    warnings: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# 四场景 Train/Val/Test 划分汇总")
    lines.append("")
    lines.append(f"- 生成时间: {generated_at}")
    lines.append(f"- 配置文件: {config_path}")
    lines.append(f"- 划分输出目录: {split_dir}")
    lines.append(f"- 总划分数: {total_split_count}")
    for scenario in SCENARIO_ORDER:
        lines.append(f"- {scenario} 划分数: {scenario_counts.get(scenario, 0)}")
    lines.append("")
    lines.append("## 划分明细")
    lines.append("")
    lines.append(
        "| split_name | scenario | target_year | outer | inner/fold | n_train | n_val | n_test | year_dist(train/val/test) | geno_overlap(train-val/train-test/val-test) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    for row in stats_rows:
        inner_or_fold = row["inner_fold"] if row["inner_fold"] is not None else row["fold"]
        year_dist = row["year_distribution"]
        dist_text = f"{year_dist['train']} / {year_dist['val']} / {year_dist['test']}"
        overlap = row["genotype_overlap"]
        overlap_text = f"{overlap['train_vs_val']} / {overlap['train_vs_test']} / {overlap['val_vs_test']}"
        lines.append(
            f"| {row['split_name']} | {row['split_type']} | {row['target_year']} | "
            f"{'' if row['outer_fold'] is None else row['outer_fold']} | "
            f"{'' if inner_or_fold is None else inner_or_fold} | "
            f"{row['n_train']} | {row['n_val']} | {row['n_test']} | "
            f"{dist_text} | {overlap_text} |"
        )
    if warnings:
        lines.append("")
        lines.append("## 警告")
        for warning in warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines)


def build_four_scenario_splits(*, config_path: Path, repo_root: Path | None = None) -> dict[str, object]:
    repo = repo_root or Path.cwd()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_cfg = dict(cfg.get("input", {}))
    split_cfg = dict(cfg.get("splits", {}))
    output_cfg = dict(cfg.get("output", {}))

    input_path = repo / str(input_cfg.get("path", "data/processed/master_plot_metadata.parquet"))
    if not input_path.exists():
        raise FileNotFoundError(f"输入样本索引不存在: {input_path}")

    sample_raw = _read_table(input_path, file_type=input_cfg.get("file_type"))
    required_columns = [str(c) for c in input_cfg.get("required_columns", [])]
    sample_id_column = str(input_cfg.get("sample_id_column", "plot_id"))
    meta_df = _prepare_meta_df(sample_raw, required_columns=required_columns, sample_id_column=sample_id_column)

    random_seed = int(split_cfg.get("random_seed", 42))
    scenarios_cfg = dict(split_cfg.get("scenarios", {}))
    if not scenarios_cfg:
        raise ValueError("splits.scenarios 不能为空。")

    split_dir = repo / str(output_cfg.get("split_dir", "data/processed/splits_4scenarios"))
    split_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    if bool(output_cfg.get("clean_previous_generated_files", True)):
        for pattern in ["reference_*.json", "within_season_*.json", "loso_*.json", "loso_genotype_*.json"]:
            for path in split_dir.glob(pattern):
                if path.is_file():
                    path.unlink()

    stats_rows: list[dict[str, object]] = []
    scenario_counts: dict[str, int] = {key: 0 for key in SCENARIO_ORDER}

    for scenario_name in SCENARIO_ORDER:
        scenario_cfg = dict(scenarios_cfg.get(scenario_name, {}))
        if not scenario_cfg:
            warnings.append(f"场景 `{scenario_name}` 缺少配置，已跳过。")
            continue
        split_groups = build_split_groups_for_scenario(
            scenario_name=scenario_name,
            scenario_cfg=scenario_cfg,
            meta_df=meta_df,
            random_seed=random_seed,
            repo_root=repo,
        )
        for key, splits in split_groups.items():
            for sp in splits:
                train_df = _subset_by_ids(meta_df, sp.train_ids)
                val_df = _subset_by_ids(meta_df, sp.val_ids)
                test_df = _subset_by_ids(meta_df, sp.test_ids)
                _ensure_disjoint(
                    train_ids=sp.train_ids,
                    val_ids=sp.val_ids,
                    test_ids=sp.test_ids,
                    split_name=sp.split_name,
                )
                payload = {
                    "split_name": sp.split_name,
                    "split_type": scenario_name,
                    "target_year": int(sp.target_year),
                    "outer_fold": int(sp.outer_fold),
                    "inner_fold": int(sp.inner_fold),
                    "random_seed": int(sp.random_seed) if sp.random_seed is not None else None,
                    "train_ids": sp.train_ids,
                    "val_ids": sp.val_ids,
                    "test_ids": sp.test_ids,
                }
                write_json(split_dir / f"{sp.split_name}.json", payload)
                stats_rows.append(
                    _build_stats_entry(
                        split_name=sp.split_name,
                        split_type=scenario_name,
                        target_year=sp.target_year,
                        outer_fold=sp.outer_fold,
                        inner_fold=sp.inner_fold,
                        fold=None if sp.inner_fold is None else sp.inner_fold,
                        train_ids=sp.train_ids,
                        val_ids=sp.val_ids,
                        test_ids=sp.test_ids,
                        all_df=meta_df,
                    )
                )
                scenario_counts[scenario_name] += 1

    stats_rows = sorted(stats_rows, key=lambda row: row["split_name"])
    report_path = repo / str(output_cfg.get("summary_report", "outputs/reports/data_split_summary_4scenarios.md"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = _render_summary_markdown(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        config_path=config_path if config_path.is_absolute() else (repo / config_path),
        split_dir=split_dir,
        total_split_count=len(stats_rows),
        scenario_counts=scenario_counts,
        stats_rows=stats_rows,
        warnings=warnings,
    )
    report_path.write_text(report_text, encoding="utf-8")
    write_json(
        report_path.with_suffix(".json"),
        {
            "config_path": str(config_path if config_path.is_absolute() else (repo / config_path)),
            "split_dir": str(split_dir),
            "total_splits": int(len(stats_rows)),
            "scenario_counts": scenario_counts,
            "warnings": warnings,
        },
    )

    return {
        "split_dir": str(split_dir),
        "report_path": str(report_path),
        "total_splits": int(len(stats_rows)),
        "scenario_counts": scenario_counts,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build four-scenario train/val/test splits.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_splits_4scenarios_example.yaml"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    result = build_four_scenario_splits(config_path=config_path, repo_root=repo_root)
    print("build_four_scenario_splits_done")
    print(f"split_dir: {result['split_dir']}")
    print(f"report: {result['report_path']}")
    print(f"total={result['total_splits']}")
    for scenario in SCENARIO_ORDER:
        print(f"{scenario}={result['scenario_counts'].get(scenario, 0)}")
    if result["warnings"]:
        print("warnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
