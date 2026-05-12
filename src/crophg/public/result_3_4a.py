from __future__ import annotations

import argparse
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

SECTION_CODE = "3.4A"
SECTION_SCOPE = "public"
SECTION_SLUG = "growth_prefix_performance_curve"
VARIANT_ORDER = ["G", "H_FULL", "G+FULLH", "H_ANCHOR_AUTO", "G+H_ANCHOR_AUTO"]
ANCHOR_ORDER_COLS = ["scenario", "target", "input_variant", "anchor_order", "anchor_tb", "anchor_phase"]
PROPOSED_VARIANTS = ["G+FULLH", "H_ANCHOR_AUTO", "G+H_ANCHOR_AUTO"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="crophg 3.4A formal analysis entry")
    parser.add_argument("--input-dir", type=str, required=True, help="Existing result directory with metrics_summary.csv and metrics_by_fold.csv")
    parser.add_argument("--output-dir", type=str, default="", help="Optional output report directory")
    parser.add_argument("--print-spec", action="store_true", help="Print section scaffold metadata and exit")
    return parser


def _ensure_variants(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["input_variant"].isin(VARIANT_ORDER)].copy()


def _to_text_table(df: pd.DataFrame, float_fmt: str = "{:.6f}") -> str:
    if df.empty:
        return "```text\n(empty)\n```"
    fmt_df = df.copy()
    for col in fmt_df.select_dtypes(include=["float64", "float32"]).columns.tolist():
        fmt_df[col] = fmt_df[col].map(lambda x: float_fmt.format(x) if pd.notna(x) else "")
    return "```text\n" + fmt_df.to_string(index=False) + "\n```"


def build_trait_level_compare(metrics_summary: pd.DataFrame) -> pd.DataFrame:
    table = (
        metrics_summary.pivot_table(
            index=["scenario", "target"],
            columns="input_variant",
            values="mean_pearson",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for variant in VARIANT_ORDER:
        if variant not in table.columns:
            table[variant] = float("nan")
    table["scenario"] = ordered_category(table["scenario"], SCENARIO_ORDER)
    table = table.sort_values(["scenario", "target"]).reset_index(drop=True)
    table["scenario"] = map_scenario_labels(table["scenario"])
    return table.loc[:, ["scenario", "target", *VARIANT_ORDER]].copy()


def build_scenario_summary(metrics_summary: pd.DataFrame) -> pd.DataFrame:
    table = metrics_summary.groupby(["scenario", "input_variant"], as_index=False)["mean_pearson"].mean().copy()
    table["scenario"] = ordered_category(table["scenario"], SCENARIO_ORDER)
    table["input_variant"] = ordered_category(table["input_variant"], VARIANT_ORDER)
    table = table.sort_values(["scenario", "input_variant"]).reset_index(drop=True)
    table["scenario"] = map_scenario_labels(table["scenario"])
    table["input_variant"] = table["input_variant"].astype(str)
    return table


def build_overall_summary(metrics_summary: pd.DataFrame) -> pd.DataFrame:
    table = (
        metrics_summary.groupby("input_variant", as_index=False)["mean_pearson"]
        .mean()
        .rename(columns={"mean_pearson": "overall_mean_pearson"})
        .copy()
    )
    table["input_variant"] = ordered_category(table["input_variant"], VARIANT_ORDER)
    table = table.sort_values("input_variant").reset_index(drop=True)
    table["input_variant"] = table["input_variant"].astype(str)
    return table


def build_prefix_curve_summary(metrics_summary: pd.DataFrame) -> pd.DataFrame:
    df = metrics_summary.loc[metrics_summary["input_variant"].isin(VARIANT_ORDER), :].copy()
    if df.empty:
        return pd.DataFrame(columns=["scenario", "target", "input_variant", "anchor_order", "anchor_tb", "anchor_phase", "mean_pearson"])
    group_cols = ["scenario", "target", "input_variant", "anchor_order", "anchor_tb", "anchor_phase"]
    out = df.groupby(group_cols, as_index=False)["mean_pearson"].mean()
    out["scenario"] = ordered_category(out["scenario"], SCENARIO_ORDER)
    out["input_variant"] = ordered_category(out["input_variant"], VARIANT_ORDER)
    out = out.sort_values(["scenario", "target", "input_variant", "anchor_order"]).reset_index(drop=True)
    out["scenario"] = map_scenario_labels(out["scenario"])
    out["input_variant"] = out["input_variant"].astype(str)
    out["target"] = out["target"].astype(str)
    out["anchor_phase"] = out["anchor_phase"].astype(str)
    return out


def build_anchor_delta_summary(prefix_curve: pd.DataFrame) -> pd.DataFrame:
    if prefix_curve.empty:
        return pd.DataFrame()
    pivot = (
        prefix_curve.pivot_table(
            index=["scenario", "target", "anchor_order", "anchor_tb", "anchor_phase"],
            columns="input_variant",
            values="mean_pearson",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for variant in VARIANT_ORDER:
        if variant not in pivot.columns:
            pivot[variant] = float("nan")
    pivot["delta_g_vs_h_full"] = pivot["G"] - pivot["H_FULL"]
    pivot["delta_gh_full_vs_h_full"] = pivot["G+FULLH"] - pivot["H_FULL"]
    pivot["delta_h_auto_vs_h_full"] = pivot["H_ANCHOR_AUTO"] - pivot["H_FULL"]
    pivot["delta_gh_auto_vs_h_full"] = pivot["G+H_ANCHOR_AUTO"] - pivot["H_FULL"]
    pivot["delta_gh_auto_vs_gh_full"] = pivot["G+H_ANCHOR_AUTO"] - pivot["G+FULLH"]
    pivot["delta_gh_auto_vs_g"] = pivot["G+H_ANCHOR_AUTO"] - pivot["G"]
    pivot["scenario"] = ordered_category(pivot["scenario"], SCENARIO_ORDER)
    pivot = pivot.sort_values(["scenario", "target", "anchor_order"]).reset_index(drop=True)
    pivot["scenario"] = map_scenario_labels(pivot["scenario"])
    pivot["target"] = pivot["target"].astype(str)
    pivot["anchor_phase"] = pivot["anchor_phase"].astype(str)
    return pivot.loc[
        :,
        [
            "scenario",
            "target",
            "anchor_order",
            "anchor_tb",
            "anchor_phase",
            *VARIANT_ORDER,
            "delta_g_vs_h_full",
            "delta_gh_full_vs_h_full",
            "delta_h_auto_vs_h_full",
            "delta_gh_auto_vs_h_full",
            "delta_gh_auto_vs_gh_full",
            "delta_gh_auto_vs_g",
        ],
    ].copy()


def build_saturation_summary(prefix_curve: pd.DataFrame) -> pd.DataFrame:
    if prefix_curve.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (scenario, target, input_variant), sub in prefix_curve.groupby(["scenario", "target", "input_variant"]):
        sub = sub.sort_values("anchor_order")
        first = float(sub.iloc[0]["mean_pearson"])
        last = float(sub.iloc[-1]["mean_pearson"])
        peak = float(sub["mean_pearson"].max())
        peak_anchor = int(sub.loc[sub["mean_pearson"].idxmax(), "anchor_order"])
        rows.append(
            {
                "scenario": scenario,
                "target": target,
                "input_variant": input_variant,
                "first_anchor_pearson": first,
                "last_anchor_pearson": last,
                "absolute_gain": last - first,
                "peak_anchor_order": peak_anchor,
                "peak_pearson": peak,
            }
        )
    out = pd.DataFrame(rows)
    out["scenario"] = ordered_category(out["scenario"], SCENARIO_ORDER)
    out["input_variant"] = ordered_category(out["input_variant"], VARIANT_ORDER)
    out = out.sort_values(["scenario", "target", "input_variant"]).reset_index(drop=True)
    out["scenario"] = map_scenario_labels(out["scenario"])
    out["target"] = out["target"].astype(str)
    out["input_variant"] = out["input_variant"].astype(str)
    return out


def build_early_advantage_summary(prefix_curve: pd.DataFrame) -> pd.DataFrame:
    if prefix_curve.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (scenario, target), sub in prefix_curve.groupby(["scenario", "target"]):
        sub = sub.sort_values("anchor_order")
        last_by_variant: dict[str, float] = {}
        first_by_variant: dict[str, float] = {}
        reach90_by_variant: dict[str, float] = {}

        for variant, variant_df in sub.groupby("input_variant"):
            variant_df = variant_df.sort_values("anchor_order")
            last_val = float(variant_df.iloc[-1]["mean_pearson"])
            first_val = float(variant_df.iloc[0]["mean_pearson"])
            threshold = 0.9 * last_val
            hit = variant_df.loc[variant_df["mean_pearson"] >= threshold]
            reach90 = float(hit.iloc[0]["anchor_order"]) if not hit.empty else float("nan")
            last_by_variant[str(variant)] = last_val
            first_by_variant[str(variant)] = first_val
            reach90_by_variant[str(variant)] = reach90

        row: dict[str, object] = {"scenario": scenario, "target": target}
        h_full_last = last_by_variant.get("H_FULL", float("nan"))
        h_full_first = first_by_variant.get("H_FULL", float("nan"))
        h_full_reach90 = reach90_by_variant.get("H_FULL", float("nan"))
        row["h_full_first_anchor"] = h_full_first
        row["h_full_last_anchor"] = h_full_last
        row["h_full_reach90_anchor"] = h_full_reach90
        for variant in PROPOSED_VARIANTS:
            row[f"{variant}_first_anchor"] = first_by_variant.get(variant, float("nan"))
            row[f"{variant}_last_anchor"] = last_by_variant.get(variant, float("nan"))
            row[f"{variant}_reach90_anchor"] = reach90_by_variant.get(variant, float("nan"))
            row[f"{variant}_first_minus_h_full"] = first_by_variant.get(variant, float("nan")) - h_full_first
            row[f"{variant}_last_minus_h_full"] = last_by_variant.get(variant, float("nan")) - h_full_last
            row[f"{variant}_reach90_minus_h_full"] = reach90_by_variant.get(variant, float("nan")) - h_full_reach90
        rows.append(row)

    out = pd.DataFrame(rows)
    out["scenario"] = ordered_category(out["scenario"], SCENARIO_ORDER)
    out = out.sort_values(["scenario", "target"]).reset_index(drop=True)
    out["scenario"] = map_scenario_labels(out["scenario"])
    out["target"] = out["target"].astype(str)
    return out


def write_report(
    *,
    out_dir: Path,
    input_dir: Path,
    trait_level_compare: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    prefix_curve: pd.DataFrame,
    anchor_delta_summary: pd.DataFrame,
    saturation_summary: pd.DataFrame,
    early_advantage_summary: pd.DataFrame,
    notes: dict[str, object],
) -> None:
    best_row = overall_summary.sort_values("overall_mean_pearson", ascending=False).iloc[0]
    best_variant = str(best_row["input_variant"])
    best_score = float(best_row["overall_mean_pearson"])

    lines: list[str] = []
    lines.append("# Result 3.4A Formal Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- input_dir: `{input_dir.as_posix()}`")
    lines.append("- section meaning: whether `G+FULLH / H_ANCHOR_AUTO / G+H_ANCHOR_AUTO` establish effective prediction earlier than `H_FULL`")
    lines.append("- baseline: `H_FULL`")
    lines.append("- proposed models: `G+FULLH`, `H_ANCHOR_AUTO`, `G+H_ANCHOR_AUTO`")
    lines.append("- auxiliary reference: `G`")
    lines.append("- fixed variants: `G`, `H_FULL`, `G+FULLH`, `H_ANCHOR_AUTO`, `G+H_ANCHOR_AUTO`")
    lines.append("- report unit: `scenario × trait × input_variant × anchor_order`")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Best overall variant: `{best_variant}` with mean Pearson `{best_score:.4f}`.")
    if notes.get("anchor_orders"):
        lines.append(f"- Observed anchor orders: `{notes['anchor_orders']}`")
    lines.append("")
    lines.append("## Trait-level Compare")
    lines.append(_to_text_table(trait_level_compare))
    lines.append("")
    lines.append("## Scenario Mean")
    lines.append(_to_text_table(scenario_summary))
    lines.append("")
    lines.append("## Overall Mean")
    lines.append(_to_text_table(overall_summary))
    lines.append("")
    lines.append("## Prefix Curve Preview")
    preview_cols = ["scenario", "target", "input_variant", "anchor_order", "anchor_tb", "anchor_phase", "mean_pearson"]
    lines.append(_to_text_table(prefix_curve.loc[:, preview_cols].head(80)))
    lines.append("")
    lines.append("## Anchor Delta Summary")
    lines.append(_to_text_table(anchor_delta_summary))
    lines.append("")
    lines.append("## Early Advantage Summary")
    lines.append(_to_text_table(early_advantage_summary))
    lines.append("")
    lines.append("## Saturation Summary")
    lines.append(_to_text_table(saturation_summary))
    (out_dir / "result_3_4a_formal_analysis.md").write_text("\n".join(lines), encoding="utf-8")


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
        out_dir = Path.cwd() / "outputs" / "reports" / f"result_3_4a_formal_analysis_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_summary = _ensure_variants(read_required_csv(input_dir, "metrics_summary.csv"))
    prefix_curve = build_prefix_curve_summary(metrics_summary)
    trait_level_compare = build_trait_level_compare(metrics_summary)
    scenario_summary = build_scenario_summary(metrics_summary)
    overall_summary = build_overall_summary(metrics_summary)
    anchor_delta_summary = build_anchor_delta_summary(prefix_curve)
    early_advantage_summary = build_early_advantage_summary(prefix_curve)
    saturation_summary = build_saturation_summary(prefix_curve)

    anchor_orders = sorted(pd.to_numeric(prefix_curve["anchor_order"], errors="coerce").dropna().astype(int).unique().tolist())
    notes = {"input_dir": str(input_dir), "anchor_orders": anchor_orders}

    prefix_curve.to_csv(out_dir / "prefix_curve_summary.csv", index=False)
    trait_level_compare.to_csv(out_dir / "trait_level_compare.csv", index=False)
    scenario_summary.to_csv(out_dir / "scenario_summary.csv", index=False)
    overall_summary.to_csv(out_dir / "overall_summary.csv", index=False)
    anchor_delta_summary.to_csv(out_dir / "anchor_delta_summary.csv", index=False)
    early_advantage_summary.to_csv(out_dir / "early_advantage_summary.csv", index=False)
    saturation_summary.to_csv(out_dir / "saturation_summary.csv", index=False)
    write_json(out_dir / "run_notes.json", notes)

    write_report(
        out_dir=out_dir,
        input_dir=input_dir,
        trait_level_compare=trait_level_compare,
        scenario_summary=scenario_summary,
        overall_summary=overall_summary,
        prefix_curve=prefix_curve,
        anchor_delta_summary=anchor_delta_summary,
        early_advantage_summary=early_advantage_summary,
        saturation_summary=saturation_summary,
        notes=notes,
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
