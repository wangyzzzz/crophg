from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from crophg.common.report_utils import (
    SCENARIO_ORDER,
    map_scenario_labels,
    ordered_category,
    read_required_csv,
    to_text_table,
    write_json,
)

SECTION_CODE = "3.4B"
SECTION_SCOPE = "public"
SECTION_SLUG = "anchor_vi_evolution"
VARIANT_ORDER = ["H_ANCHOR_AUTO", "G+H_ANCHOR_AUTO"]
TRAIT_ORDER = ["ActualYD", "CM", "LM", "PHM", "Spike", "TKW"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="crophg 3.4B formal analysis entry")
    parser.add_argument("--input-dir", type=str, required=True, help="Existing result directory with metrics_by_fold.csv")
    parser.add_argument("--output-dir", type=str, default="", help="Optional output report directory")
    parser.add_argument("--print-spec", action="store_true", help="Print section scaffold metadata and exit")
    return parser


def _parse_json_list(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, list):
        return []
    return [str(x) for x in obj]


def _ordered_category_safe(series: pd.Series, order: list[str]) -> pd.Series:
    return ordered_category(series, order)


def _to_text_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    if df.empty:
        return "```text\n(empty)\n```"
    fmt_df = df.copy()
    for col in fmt_df.select_dtypes(include=["float64", "float32"]).columns.tolist():
        fmt_df[col] = fmt_df[col].map(lambda x: float_fmt.format(x) if pd.notna(x) else "")
    return "```text\n" + fmt_df.to_string(index=False) + "\n```"


def build_kept_group_evolution(metrics_by_fold: pd.DataFrame) -> pd.DataFrame:
    df = metrics_by_fold.loc[
        metrics_by_fold["input_variant"].isin(VARIANT_ORDER),
        [
            "scenario",
            "target",
            "input_variant",
            "anchor_idx",
            "anchor_order",
            "anchor_tb",
            "anchor_phase",
            "n_group_kept",
            "n_anchor_kept",
            "n_vi_kept",
            "growth_aware_min_groups",
            "selected_window_radius",
            "kept_group_ids_json",
            "kept_anchor_tokens_json",
            "kept_vi_names_json",
        ],
    ].copy()
    if df.empty:
        return pd.DataFrame()

    out_rows: list[dict[str, object]] = []
    grouped = df.groupby(
        ["scenario", "target", "input_variant", "anchor_idx", "anchor_order", "anchor_tb", "anchor_phase"],
        as_index=False,
    )
    for _, sub in grouped:
        group_lists = [_parse_json_list(x) for x in sub["kept_group_ids_json"].tolist()]
        anchor_lists = [_parse_json_list(x) for x in sub["kept_anchor_tokens_json"].tolist()]
        vi_lists = [_parse_json_list(x) for x in sub["kept_vi_names_json"].tolist()]
        union_group_ids = sorted({item for row in group_lists for item in row})
        union_anchor_tokens = sorted({item for row in anchor_lists for item in row})
        union_vi_names = sorted({item for row in vi_lists for item in row})
        row0 = sub.iloc[0]
        out_rows.append(
            {
                "scenario": str(row0["scenario"]),
                "target": str(row0["target"]),
                "input_variant": str(row0["input_variant"]),
                "anchor_idx": int(row0["anchor_idx"]),
                "anchor_order": int(row0["anchor_order"]),
                "anchor_tb": int(row0["anchor_tb"]),
                "anchor_phase": str(row0["anchor_phase"]),
                "mean_n_group_kept": float(pd.to_numeric(sub["n_group_kept"], errors="coerce").mean()),
                "mean_n_anchor_kept": float(pd.to_numeric(sub["n_anchor_kept"], errors="coerce").mean()),
                "mean_n_vi_kept": float(pd.to_numeric(sub["n_vi_kept"], errors="coerce").mean()),
                "mean_growth_aware_min_groups": float(pd.to_numeric(sub["growth_aware_min_groups"], errors="coerce").mean()),
                "mean_selected_window_radius": float(pd.to_numeric(sub["selected_window_radius"], errors="coerce").mean()),
                "union_group_count": int(len(union_group_ids)),
                "union_anchor_count": int(len(union_anchor_tokens)),
                "union_vi_count": int(len(union_vi_names)),
                "union_anchor_tokens_json": json.dumps(union_anchor_tokens, ensure_ascii=False),
                "union_vi_names_json": json.dumps(union_vi_names, ensure_ascii=False),
                "union_group_ids_json": json.dumps(union_group_ids, ensure_ascii=False),
                "union_anchor_tokens_text": ", ".join(union_anchor_tokens),
                "union_vi_names_text": ", ".join(union_vi_names),
            }
        )

    out = pd.DataFrame(out_rows)
    out["scenario"] = _ordered_category_safe(out["scenario"], SCENARIO_ORDER)
    out["target"] = _ordered_category_safe(out["target"], TRAIT_ORDER)
    out["input_variant"] = _ordered_category_safe(out["input_variant"], VARIANT_ORDER)
    out = out.sort_values(["scenario", "target", "input_variant", "anchor_idx"]).reset_index(drop=True)
    out["scenario"] = map_scenario_labels(out["scenario"])
    out["target"] = out["target"].astype(str)
    out["input_variant"] = out["input_variant"].astype(str)
    return out


def build_vi_frequency_table(evolution_df: pd.DataFrame) -> pd.DataFrame:
    if evolution_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, row in evolution_df.iterrows():
        vi_names = _parse_json_list(row["union_vi_names_json"])
        for vi_name in vi_names:
            rows.append(
                {
                    "scenario": row["scenario"],
                    "target": row["target"],
                    "input_variant": row["input_variant"],
                    "anchor_idx": int(row["anchor_idx"]),
                    "anchor_order": int(row["anchor_order"]),
                    "anchor_tb": int(row["anchor_tb"]),
                    "anchor_phase": row["anchor_phase"],
                    "vi_name": vi_name,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["scenario"] = _ordered_category_safe(out["scenario"], SCENARIO_ORDER)
    out["target"] = _ordered_category_safe(out["target"], TRAIT_ORDER)
    out["input_variant"] = _ordered_category_safe(out["input_variant"], VARIANT_ORDER)
    out = out.sort_values(["scenario", "target", "input_variant", "anchor_idx", "vi_name"]).reset_index(drop=True)
    out["scenario"] = map_scenario_labels(out["scenario"])
    out["target"] = out["target"].astype(str)
    out["input_variant"] = out["input_variant"].astype(str)
    return out


def build_anchor_frequency_table(evolution_df: pd.DataFrame) -> pd.DataFrame:
    if evolution_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, row in evolution_df.iterrows():
        anchor_tokens = _parse_json_list(row["union_anchor_tokens_json"])
        for anchor_token in anchor_tokens:
            rows.append(
                {
                    "scenario": row["scenario"],
                    "target": row["target"],
                    "input_variant": row["input_variant"],
                    "anchor_idx": int(row["anchor_idx"]),
                    "anchor_order": int(row["anchor_order"]),
                    "anchor_tb": int(row["anchor_tb"]),
                    "anchor_phase": row["anchor_phase"],
                    "kept_anchor_token": anchor_token,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["scenario"] = _ordered_category_safe(out["scenario"], SCENARIO_ORDER)
    out["target"] = _ordered_category_safe(out["target"], TRAIT_ORDER)
    out["input_variant"] = _ordered_category_safe(out["input_variant"], VARIANT_ORDER)
    out = out.sort_values(["scenario", "target", "input_variant", "anchor_idx", "kept_anchor_token"]).reset_index(drop=True)
    out["scenario"] = map_scenario_labels(out["scenario"])
    out["target"] = out["target"].astype(str)
    out["input_variant"] = out["input_variant"].astype(str)
    return out


def write_report(
    *,
    out_dir: Path,
    input_dir: Path,
    evolution_df: pd.DataFrame,
    vi_frequency_df: pd.DataFrame,
    anchor_frequency_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Result 3.4B Formal Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- input_dir: `{input_dir.as_posix()}`")
    lines.append("- section meaning: evolution of selected anchor×VI units")
    lines.append("- fixed variants: `H_ANCHOR_AUTO`, `G+H_ANCHOR_AUTO`")
    lines.append("- focus: how selected anchor×VI units change as more growth prefixes become observable")
    lines.append("")
    lines.append("## Prefix-level Evolution Preview")
    preview_cols = [
        "scenario",
        "target",
        "input_variant",
        "anchor_order",
        "anchor_tb",
        "anchor_phase",
        "mean_growth_aware_min_groups",
        "mean_n_group_kept",
        "union_anchor_count",
        "union_vi_count",
        "union_anchor_tokens_text",
        "union_vi_names_text",
    ]
    lines.append(_to_text_table(evolution_df.loc[:, [c for c in preview_cols if c in evolution_df.columns]].head(80)))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- 如果 `union_anchor_tokens_text` 随 prefix 扩展而向后延伸，说明表示层的时间支持集在扩张。")
    lines.append("- 如果 `union_vi_names_text` 在不同 trait 间明显不同，说明被保留的光谱维度具有 trait-specific 结构。")
    lines.append("- 如果 `G+H_ANCHOR_AUTO` 与 `H_ANCHOR_AUTO` 在同一 prefix 下保留的组合不同，说明 GH 联合筛 H 已经改变了有效表示。")
    lines.append("")
    lines.append("## Key Files")
    lines.append("- `kept_group_evolution.csv`")
    lines.append("- `kept_vi_frequency.csv`")
    lines.append("- `kept_anchor_frequency.csv`")
    (out_dir / "result_3_4b_formal_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.print_spec:
        print(f"section={SECTION_CODE}")
        print(f"scope={SECTION_SCOPE}")
        print(f"slug={SECTION_SLUG}")
        return 0

    input_dir = Path(args.input_dir).resolve()
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d")
        out_dir = Path.cwd() / "outputs" / "reports" / f"result_3_4b_formal_analysis_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_fold = read_required_csv(input_dir, "metrics_by_fold.csv")
    evolution_df = build_kept_group_evolution(metrics_by_fold)
    vi_frequency_df = build_vi_frequency_table(evolution_df)
    anchor_frequency_df = build_anchor_frequency_table(evolution_df)

    evolution_df.to_csv(out_dir / "kept_group_evolution.csv", index=False)
    vi_frequency_df.to_csv(out_dir / "kept_vi_frequency.csv", index=False)
    anchor_frequency_df.to_csv(out_dir / "kept_anchor_frequency.csv", index=False)
    write_json(out_dir / "run_notes.json", {"input_dir": str(input_dir)})

    write_report(
        out_dir=out_dir,
        input_dir=input_dir,
        evolution_df=evolution_df,
        vi_frequency_df=vi_frequency_df,
        anchor_frequency_df=anchor_frequency_df,
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
