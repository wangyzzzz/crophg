from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SCENARIO_ORDER = ["reference", "within_season", "loso", "loso_genotype"]
SCENARIO_LABELS = {
    "reference": "Reference",
    "within_season": "Genotype-Novel",
    "loso": "Year-Novel",
    "loso_genotype": "Joint-Novel",
}


def ordered_category(series: pd.Series, order: list[str]) -> pd.Series:
    return pd.Categorical(series.astype(str), categories=order, ordered=True)


def map_scenario_labels(series: pd.Series) -> pd.Series:
    as_str = series.astype(str)
    return as_str.map(lambda x: SCENARIO_LABELS.get(x, x))


def read_required_csv(base: Path, name: str) -> pd.DataFrame:
    path = base / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def to_text_table(df: pd.DataFrame, float_fmt: str = "{:.6f}") -> str:
    if df.empty:
        return "```text\n(empty)\n```"
    fmt_df = df.copy()
    float_cols = fmt_df.select_dtypes(include=["float64", "float32"]).columns.tolist()
    for col in float_cols:
        fmt_df[col] = fmt_df[col].map(lambda x: float_fmt.format(x) if pd.notna(x) else "")
    return "```text\n" + fmt_df.to_string(index=False) + "\n```"


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "(empty)"
    part = df.head(max_rows).copy()
    try:
        return part.to_markdown(index=False)
    except Exception:
        return to_text_table(part, float_fmt="{:.6f}")

