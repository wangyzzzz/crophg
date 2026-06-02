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

SECTION_CODE = "3.3B"
SECTION_SCOPE = "public"
SECTION_SLUG = "predictive_performance_and_compression_gain"
VARIANT_ORDER = ["G", "H_FULL", "G+FULLH", "H_ANCHOR_AUTO", "G+H_ANCHOR_AUTO"]
AUTO_VARIANTS = ["H_ANCHOR_AUTO", "G+H_ANCHOR_AUTO"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="cropvig_2 formal analysis entry")
    parser.add_argument("--input-dir", type=str, required=True, help="Existing result directory with metrics_summary.csv and metrics_by_fold.csv")
    parser.add_argument("--output-dir", type=str, default="", help="Optional output report directory")
    parser.add_argument("--print-spec", action="store_true", help="Print section scaffold metadata and exit")
    return parser


def _ensure_variants(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["input_variant"].isin(VARIANT_ORDER)].copy()


def _select_final_prefix_rows(metrics_summary: pd.DataFrame) -> pd.DataFrame:
    if metrics_summary.empty or "anchor_order" not in metrics_summary.columns:
        return metrics_summary.copy()
    df = metrics_summary.copy()
    df["_anchor_order_num"] = pd.to_numeric(df["anchor_order"], errors="coerce")
    if df["_anchor_order_num"].notna().any():
        max_anchor = df.groupby(["scenario", "target", "input_variant"], dropna=False)["_anchor_order_num"].transform("max")
        df = df.loc[df["_anchor_order_num"] == max_anchor].copy()
    return df.drop(columns=["_anchor_order_num"], errors="ignore")


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


def build_trait_overall_summary(metrics_summary: pd.DataFrame) -> pd.DataFrame:
    table = metrics_summary.groupby(["target", "input_variant"], as_index=False)["mean_pearson"].mean().copy()
    table["input_variant"] = ordered_category(table["input_variant"], VARIANT_ORDER)
    table = (
        table.pivot(index="target", columns="input_variant", values="mean_pearson")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for variant in VARIANT_ORDER:
        if variant not in table.columns:
            table[variant] = float("nan")
    return table.loc[:, ["target", *VARIANT_ORDER]].copy()


def build_gain_summary(metrics_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        metrics_summary.groupby(["scenario", "target", "input_variant"], as_index=False)["mean_pearson"]
        .mean()
        .pivot(index=["scenario", "target"], columns="input_variant", values="mean_pearson")
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
    pivot = pivot.sort_values(["scenario", "target"]).reset_index(drop=True)
    pivot["scenario"] = map_scenario_labels(pivot["scenario"])
    return pivot.loc[
        :,
        [
            "scenario",
            "target",
            "delta_g_vs_h_full",
            "delta_gh_full_vs_h_full",
            "delta_h_auto_vs_h_full",
            "delta_gh_auto_vs_h_full",
            "delta_gh_auto_vs_gh_full",
            "delta_gh_auto_vs_g",
        ],
    ].copy()


def build_feature_summary(metrics_by_fold: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant in ["H_FULL", "G+FULLH", "H_ANCHOR_AUTO", "G+H_ANCHOR_AUTO"]:
        sub = metrics_by_fold.loc[metrics_by_fold["input_variant"] == variant].copy()
        if sub.empty:
            continue
        row: dict[str, object] = {"input_variant": variant}
        if variant in ["H_FULL", "G+FULLH"]:
            row["mean_total_features"] = float(pd.to_numeric(sub["n_features_total"], errors="coerce").mean())
            row["mean_h_features"] = float(pd.to_numeric(sub["n_features_h_full"], errors="coerce").mean())
            row["mean_g_features"] = float(pd.to_numeric(sub["n_features_g"], errors="coerce").mean())
        else:
            row["mean_total_features"] = float(pd.to_numeric(sub["n_features_after_pruning"], errors="coerce").mean())
            row["mean_h_features"] = float(pd.to_numeric(sub["n_h_features_after_pruning"], errors="coerce").mean())
            row["mean_g_features"] = float(pd.to_numeric(sub["n_features_g"], errors="coerce").mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["input_variant"] = ordered_category(out["input_variant"], ["H_FULL", "G+FULLH", "H_ANCHOR_AUTO", "G+H_ANCHOR_AUTO"])
    out = out.sort_values("input_variant").reset_index(drop=True)
    out["input_variant"] = out["input_variant"].astype(str)
    return out


def build_compression_summary(feature_summary: pd.DataFrame) -> pd.DataFrame:
    if feature_summary.empty:
        return pd.DataFrame(columns=["comparison", "baseline_features", "reduced_features", "retained_ratio", "compression_ratio"])

    def _lookup(variant: str, col: str) -> float:
        sub = feature_summary.loc[feature_summary["input_variant"] == variant, col]
        if sub.empty:
            return float("nan")
        return float(sub.iloc[0])

    rows = []
    for baseline, reduced, label in [
        ("H_FULL", "H_ANCHOR_AUTO", "H_FULL -> H_ANCHOR_AUTO"),
        ("G+FULLH", "G+H_ANCHOR_AUTO", "G+FULLH -> G+H_ANCHOR_AUTO"),
    ]:
        baseline_features = _lookup(baseline, "mean_total_features")
        reduced_features = _lookup(reduced, "mean_total_features")
        retained_ratio = reduced_features / baseline_features if baseline_features else float("nan")
        compression_ratio = baseline_features / reduced_features if reduced_features else float("nan")
        rows.append(
            {
                "comparison": label,
                "baseline_features": baseline_features,
                "reduced_features": reduced_features,
                "retained_ratio": retained_ratio,
                "compression_ratio": compression_ratio,
            }
        )
    return pd.DataFrame(rows)


def build_auto_window_selection_counts(metrics_by_fold: pd.DataFrame) -> pd.DataFrame:
    auto_df = metrics_by_fold.loc[
        metrics_by_fold["input_variant"].isin(AUTO_VARIANTS) & metrics_by_fold["selected_window_radius"].notna(),
        ["scenario", "input_variant", "selected_window_radius"],
    ].copy()
    if auto_df.empty:
        return pd.DataFrame(columns=["scenario", "input_variant", "selected_window_radius", "n_folds"])
    auto_df["selected_window_radius"] = auto_df["selected_window_radius"].astype(int)
    out = auto_df.groupby(["scenario", "input_variant", "selected_window_radius"], as_index=False).size()
    out = out.rename(columns={"size": "n_folds"})
    out["scenario"] = ordered_category(out["scenario"], SCENARIO_ORDER)
    out["input_variant"] = ordered_category(out["input_variant"], AUTO_VARIANTS)
    out = out.sort_values(["scenario", "input_variant", "selected_window_radius"]).reset_index(drop=True)
    out["scenario"] = map_scenario_labels(out["scenario"])
    out["input_variant"] = out["input_variant"].astype(str)
    return out


def infer_run_notes(metrics_by_fold: pd.DataFrame) -> dict[str, object]:
    notes: dict[str, object] = {}
    auto_df = metrics_by_fold.loc[metrics_by_fold["input_variant"].isin(AUTO_VARIANTS)].copy()
    if auto_df.empty:
        return notes
    k_col = "selected_k_mode" if "selected_k_mode" in auto_df.columns else ("n_group_kept" if "n_group_kept" in auto_df.columns else "")
    if k_col:
        k_vals = pd.to_numeric(auto_df[k_col], errors="coerce")
        if k_vals.notna().any():
            notes["selected_k_min"] = int(k_vals.min())
            notes["selected_k_max"] = int(k_vals.max())
            notes["selected_k_zero_count"] = int((k_vals == 0).sum())
    if "selected_window_radius" in auto_df.columns:
        radius_vals = pd.to_numeric(auto_df["selected_window_radius"], errors="coerce")
        notes["window_radius_used"] = sorted({int(x) for x in radius_vals.dropna().tolist()})
    return notes


def write_report(
    *,
    out_dir: Path,
    input_dir: Path,
    trait_level_compare: pd.DataFrame,
    trait_overall_summary: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    gain_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    compression_summary: pd.DataFrame,
    auto_window_counts: pd.DataFrame,
    notes: dict[str, object],
) -> None:
    overall_idx = overall_summary.set_index("input_variant")["overall_mean_pearson"]
    g_delta = float(overall_idx.get("G", float("nan")) - overall_idx.get("H_FULL", float("nan")))
    gh_full_delta = float(overall_idx.get("G+FULLH", float("nan")) - overall_idx.get("H_FULL", float("nan")))
    h_delta = float(overall_idx.get("H_ANCHOR_AUTO", float("nan")) - overall_idx.get("H_FULL", float("nan")))
    gh_auto_delta = float(overall_idx.get("G+H_ANCHOR_AUTO", float("nan")) - overall_idx.get("H_FULL", float("nan")))
    gh_delta = float(overall_idx.get("G+H_ANCHOR_AUTO", float("nan")) - overall_idx.get("G+FULLH", float("nan")))
    best_row = overall_summary.sort_values("overall_mean_pearson", ascending=False).iloc[0]

    lines: list[str] = []
    lines.append("# CropVIG-2 Formal Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- input_dir: `{input_dir.as_posix()}`")
    lines.append("- section meaning: predictive performance and compression gain")
    lines.append("- baseline: `H_FULL`")
    lines.append("- proposed models: `G+FULLH`, `H_ANCHOR_AUTO`, `G+H_ANCHOR_AUTO`")
    lines.append("- auxiliary reference: `G`")
    lines.append("- fixed variants: `G`, `H_FULL`, `G+FULLH`, `H_ANCHOR_AUTO`, `G+H_ANCHOR_AUTO`")
    if notes.get("selected_k_min") is not None and notes.get("selected_k_max") is not None:
        lines.append(f"- observed K range: `{notes['selected_k_min']}-{notes['selected_k_max']}`")
    if notes.get("window_radius_used"):
        lines.append(f"- observed AUTO radii: `{notes['window_radius_used']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(
        f"- Best overall variant: `{best_row['input_variant']}` with mean Pearson `{float(best_row['overall_mean_pearson']):.4f}`."
    )
    lines.append(f"- `G - H_FULL` overall delta: `{g_delta:.4f}`.")
    lines.append(f"- `G+FULLH - H_FULL` overall delta: `{gh_full_delta:.4f}`.")
    lines.append(f"- `H_ANCHOR_AUTO - H_FULL` overall delta: `{h_delta:.4f}`.")
    lines.append(f"- `G+H_ANCHOR_AUTO - H_FULL` overall delta: `{gh_auto_delta:.4f}`.")
    lines.append(f"- `G+H_ANCHOR_AUTO - G+FULLH` overall delta: `{gh_delta:.4f}`.")
    lines.append("")
    lines.append("## Trait-level Compare")
    lines.append(to_text_table(trait_level_compare))
    lines.append("")
    lines.append("## Trait Overall Mean")
    lines.append(to_text_table(trait_overall_summary))
    lines.append("")
    lines.append("## Scenario Mean")
    lines.append(to_text_table(scenario_summary))
    lines.append("")
    lines.append("## Overall Mean")
    lines.append(to_text_table(overall_summary))
    lines.append("")
    lines.append("## Gain Summary")
    lines.append(to_text_table(gain_summary))
    lines.append("")
    lines.append("## Feature Summary")
    lines.append(to_text_table(feature_summary, float_fmt="{:.1f}"))
    lines.append("")
    lines.append("## Compression Summary")
    lines.append(to_text_table(compression_summary))
    lines.append("")
    lines.append("## AUTO Window Selection")
    lines.append(to_text_table(auto_window_counts, float_fmt="{:.1f}"))
    (out_dir / "cropvig_2_formal_analysis.md").write_text("\n".join(lines), encoding="utf-8")


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
        out_dir = Path.cwd() / "outputs" / "reports" / f"cropvig_2_formal_analysis_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_summary = _select_final_prefix_rows(_ensure_variants(read_required_csv(input_dir, "metrics_summary.csv")))
    metrics_by_fold = _ensure_variants(read_required_csv(input_dir, "metrics_by_fold.csv"))

    trait_level_compare = build_trait_level_compare(metrics_summary)
    trait_overall_summary = build_trait_overall_summary(metrics_summary)
    scenario_summary = build_scenario_summary(metrics_summary)
    overall_summary = build_overall_summary(metrics_summary)
    gain_summary = build_gain_summary(metrics_summary)
    feature_summary = build_feature_summary(metrics_by_fold)
    compression_summary = build_compression_summary(feature_summary)
    auto_window_counts = build_auto_window_selection_counts(metrics_by_fold)
    notes = infer_run_notes(metrics_by_fold)

    trait_level_compare.to_csv(out_dir / "trait_level_compare.csv", index=False)
    trait_overall_summary.to_csv(out_dir / "trait_overall_summary.csv", index=False)
    scenario_summary.to_csv(out_dir / "scenario_summary.csv", index=False)
    overall_summary.to_csv(out_dir / "overall_summary.csv", index=False)
    gain_summary.to_csv(out_dir / "gain_summary.csv", index=False)
    feature_summary.to_csv(out_dir / "feature_summary.csv", index=False)
    compression_summary.to_csv(out_dir / "compression_summary.csv", index=False)
    auto_window_counts.to_csv(out_dir / "auto_window_selection_counts.csv", index=False)
    write_json(out_dir / "run_notes.json", {"input_dir": str(input_dir), **notes})

    write_report(
        out_dir=out_dir,
        input_dir=input_dir,
        trait_level_compare=trait_level_compare,
        trait_overall_summary=trait_overall_summary,
        scenario_summary=scenario_summary,
        overall_summary=overall_summary,
        gain_summary=gain_summary,
        feature_summary=feature_summary,
        compression_summary=compression_summary,
        auto_window_counts=auto_window_counts,
        notes=notes,
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
