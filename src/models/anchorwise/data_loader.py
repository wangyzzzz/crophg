from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PHASE_TB_PATTERN = re.compile(r"(?:^|__)ph_([A-Za-z0-9_]+)__tb_([pm])(\d+)$")
TB_PATTERN = re.compile(r"(?:^|__)tb_([pm])(\d+)$")
TimeKey = Tuple[Optional[str], int]


@dataclass
class DayDataBundle:
    meta: pd.DataFrame
    y: pd.Series
    x_climate: pd.DataFrame
    x_hyperspectral: pd.DataFrame
    x_genotype: pd.DataFrame
    climate_tb_map: Dict[TimeKey, List[str]]
    hyperspectral_tb_map: Dict[TimeKey, List[str]]
    common_tbs: List[TimeKey]


@dataclass
class MultiTargetDataBundle:
    meta: pd.DataFrame
    y_df: pd.DataFrame
    x_climate: pd.DataFrame
    x_hyperspectral: pd.DataFrame
    x_genotype: pd.DataFrame
    climate_tb_map: Dict[TimeKey, List[str]]
    hyperspectral_tb_map: Dict[TimeKey, List[str]]
    common_tbs: List[TimeKey]


def parse_tb_from_col(col_name: str) -> TimeKey | None:
    phase_match = PHASE_TB_PATTERN.search(col_name)
    if phase_match:
        phase = phase_match.group(1)
        sign = 1 if phase_match.group(2) == "p" else -1
        return (phase, sign * int(phase_match.group(3)))
    match = TB_PATTERN.search(col_name)
    if not match:
        return None
    sign = 1 if match.group(1) == "p" else -1
    return (None, sign * int(match.group(2)))


def tb_to_token(tb: int) -> str:
    sign = "p" if tb >= 0 else "m"
    return f"{sign}{abs(tb):05d}"


def time_key_sort_key(time_key: TimeKey) -> tuple[int, int]:
    phase, tb = time_key
    phase_token = str(phase or "sow").lower()
    tb_int = int(tb)
    if phase_token == "head":
        # `head` is not an independent timeline starting from zero.
        # It continues after the full `sow` phase on the same biological growth axis.
        return (1, tb_int)
    return (0, tb_int)


def time_key_to_token(time_key: TimeKey) -> str:
    phase, tb = time_key
    if phase:
        return f"ph_{phase}__tb_{tb_to_token(int(tb))}"
    return f"tb_{tb_to_token(int(tb))}"


def build_tb_map(columns: List[str]) -> Dict[TimeKey, List[str]]:
    tb_map: Dict[TimeKey, List[str]] = {}
    for col in columns:
        tb = parse_tb_from_col(col)
        if tb is None:
            continue
        tb_map.setdefault(tb, []).append(col)
    return {k: sorted(v) for k, v in sorted(tb_map.items(), key=lambda x: time_key_sort_key(x[0]))}


def _unify_y_genotype_id(y_df: pd.DataFrame) -> pd.DataFrame:
    if "genotype_id" in y_df.columns:
        return y_df

    gx = "genotype_id_x" in y_df.columns
    gy = "genotype_id_y" in y_df.columns
    if gx and gy:
        left = y_df["genotype_id_x"].astype(str)
        right = y_df["genotype_id_y"].astype(str)
        if not (left == right).all():
            raise ValueError("y.parquet 中 genotype_id_x 与 genotype_id_y 不一致。")
        y_df = y_df.copy()
        y_df["genotype_id"] = left
    elif gx:
        y_df = y_df.copy()
        y_df["genotype_id"] = y_df["genotype_id_x"].astype(str)
    elif gy:
        y_df = y_df.copy()
        y_df["genotype_id"] = y_df["genotype_id_y"].astype(str)
    return y_df


def load_day_model_inputs(
    input_dir: Path,
    target_col: str = "ActualYD",
    genotype_representation: str = "all",
) -> DayDataBundle:
    multi = load_multitarget_model_inputs(input_dir=input_dir, genotype_representation=genotype_representation)
    if target_col not in multi.y_df.columns:
        raise ValueError(f"y.parquet 缺少目标列: {target_col}")

    y_series = multi.y_df[target_col].dropna().copy()
    common_ids = list(
        multi.meta.index.intersection(y_series.index)
        .intersection(multi.x_climate.index)
        .intersection(multi.x_hyperspectral.index)
        .intersection(multi.x_genotype.index)
    )

    if not common_ids:
        raise ValueError("最终对齐后无共同样本。")

    return DayDataBundle(
        meta=multi.meta.loc[common_ids].copy(),
        y=y_series.loc[common_ids].copy(),
        x_climate=multi.x_climate.loc[common_ids].copy(),
        x_hyperspectral=multi.x_hyperspectral.loc[common_ids].copy(),
        x_genotype=multi.x_genotype.loc[common_ids].copy(),
        climate_tb_map=multi.climate_tb_map,
        hyperspectral_tb_map=multi.hyperspectral_tb_map,
        common_tbs=multi.common_tbs,
    )


def load_multitarget_model_inputs(
    input_dir: Path,
    genotype_representation: str = "all",
) -> MultiTargetDataBundle:
    plot_index = pd.read_parquet(input_dir / "plot_index.parquet")
    x_climate_raw = pd.read_parquet(input_dir / "X_climate.parquet")
    x_genotype_raw = pd.read_parquet(input_dir / "X_genotype.parquet")
    x_hyperspectral_raw = pd.read_parquet(input_dir / "X_hyperspectral.parquet")
    y_raw = pd.read_parquet(input_dir / "y.parquet")

    y_raw = _unify_y_genotype_id(y_raw)

    for df in [plot_index, x_climate_raw, x_genotype_raw, x_hyperspectral_raw, y_raw]:
        if "plot_id" not in df.columns:
            raise ValueError("输入文件缺少 plot_id 列。")
        df["plot_id"] = df["plot_id"].astype(str)

    required_y_cols = {"plot_id", "year"}
    missing_y = required_y_cols - set(y_raw.columns)
    if missing_y:
        raise ValueError(f"y.parquet 缺少列: {sorted(missing_y)}")

    ids = set(plot_index["plot_id"]) & set(x_climate_raw["plot_id"]) & set(x_genotype_raw["plot_id"]) & set(
        x_hyperspectral_raw["plot_id"]
    ) & set(y_raw["plot_id"])
    if not ids:
        raise ValueError("对齐后没有可用样本。")

    base = plot_index[["plot_id", "year", "genotype_id", "plot_index", "plot"]].copy()
    base = base[base["plot_id"].isin(ids)].drop_duplicates(subset=["plot_id"])

    y_keep_cols = ["plot_id", "year"] + [c for c in y_raw.columns if c not in {"plot_id", "year"}]
    y_df = y_raw.loc[:, y_keep_cols].copy()
    y_df = y_df[y_df["plot_id"].isin(ids)].drop_duplicates(subset=["plot_id"])

    base = base.merge(y_df[["plot_id", "year"]], on="plot_id", suffixes=("", "_y"), how="left")
    if "year_y" in base.columns:
        base["year"] = base["year_y"].fillna(base["year"])
        base = base.drop(columns=["year_y"])

    base["year"] = base["year"].astype(int)
    base["genotype_id"] = base["genotype_id"].astype(str)

    def _slice_by_ids(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["plot_id"].isin(ids)].drop_duplicates(subset=["plot_id"]).set_index("plot_id")

    x_climate_df = _slice_by_ids(x_climate_raw)
    x_genotype_df = _slice_by_ids(x_genotype_raw)
    x_hyperspectral_df = _slice_by_ids(x_hyperspectral_raw)

    meta = base.set_index("plot_id")

    climate_tb_cols = [c for c in x_climate_df.columns if parse_tb_from_col(c) is not None]
    hypers_tb_cols = [c for c in x_hyperspectral_df.columns if parse_tb_from_col(c) is not None]

    climate_tb_map = build_tb_map(climate_tb_cols)
    hypers_tb_map = build_tb_map(hypers_tb_cols)

    geno_mode = genotype_representation.lower().strip()
    if geno_mode == "pca":
        genotype_feature_prefixes = ("geno_pc_",)
    elif geno_mode == "grm_pca":
        genotype_feature_prefixes = ("grm_pc_",)
    elif geno_mode == "grm":
        genotype_feature_prefixes = ("grm_val_",)
    elif geno_mode == "all":
        genotype_feature_prefixes = ("geno_pc_", "grm_pc_", "grm_val_")
    else:
        raise ValueError(
            f"不支持的 genotype_representation: {genotype_representation}. "
            "可选值: all/pca/grm/grm_pca"
        )

    genotype_feature_cols = [c for c in x_genotype_df.columns if c.startswith(genotype_feature_prefixes)]

    if not genotype_feature_cols:
        id_like_cols = {"plot", "plot_index", "year", "genotype_id"}
        genotype_feature_cols = [
            c
            for c in x_genotype_df.columns
            if c not in id_like_cols and pd.api.types.is_numeric_dtype(x_genotype_df[c])
        ]

    if not genotype_feature_cols:
        raise ValueError("X_genotype.parquet 未识别到可用基因型特征列。")

    x_climate = x_climate_df[climate_tb_cols].copy()
    x_hyperspectral = x_hyperspectral_df[hypers_tb_cols].copy()
    x_genotype = x_genotype_df[genotype_feature_cols].copy()

    common_ids = list(
        meta.index.intersection(x_climate.index)
        .intersection(x_hyperspectral.index)
        .intersection(x_genotype.index)
        .intersection(y_df.set_index("plot_id").index)
    )

    if not common_ids:
        raise ValueError("最终对齐后无共同样本。")

    meta = meta.loc[common_ids].copy()
    x_climate = x_climate.loc[common_ids].copy()
    x_hyperspectral = x_hyperspectral.loc[common_ids].copy()
    x_genotype = x_genotype.loc[common_ids].copy()
    y_df = y_df.drop(columns=[c for c in ["genotype_id_x", "genotype_id_y", "genotype_id"] if c in y_df.columns], errors="ignore")
    y_df = y_df.drop(columns=[c for c in ["plot", "plot_index"] if c in y_df.columns], errors="ignore")
    y_df = y_df.drop_duplicates(subset=["plot_id"]).set_index("plot_id").loc[common_ids].copy()

    common_tbs = sorted(
        set(climate_tb_map.keys()).intersection(hypers_tb_map.keys()),
        key=time_key_sort_key,
    )
    if not common_tbs:
        raise ValueError("climate 与 hyperspectral 没有共同 time bin。")

    return MultiTargetDataBundle(
        meta=meta,
        y_df=y_df,
        x_climate=x_climate,
        x_hyperspectral=x_hyperspectral,
        x_genotype=x_genotype,
        climate_tb_map=climate_tb_map,
        hyperspectral_tb_map=hypers_tb_map,
        common_tbs=common_tbs,
    )
