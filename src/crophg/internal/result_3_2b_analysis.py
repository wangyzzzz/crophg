from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from crophg.common.report_utils import read_required_csv, to_text_table, write_json
from crophg.internal.result_3_2bc_analysis_common import (
    build_cross_scenario_tables,
    build_same_vi_by_predictor,
    build_same_vi_summary,
    build_scenario_trait_overview,
    build_shared_anchor_report_table,
    build_top_h_vi,
    plot_scenario_heatmaps,
)

SECTION_CODE = "3.2B"
SECTION_SCOPE = "internal"
SECTION_SLUG = "single_vi_change_under_h_only_formal_analysis"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="crophg 3.2B formal analysis entry")
    parser.add_argument("--input-dir", type=str, required=True, help="Existing 3.2B result directory")
    parser.add_argument("--anchor-source-dir", type=str, required=True, help="Existing 3.2A result directory used to reconstruct formal shared-anchor")
    parser.add_argument("--output-dir", type=str, default="", help="Optional output report directory")
    parser.add_argument("--print-spec", action="store_true", help="Print section scaffold metadata and exit")
    return parser


def write_report(
    *,
    out_dir: Path,
    input_dir: Path,
    anchor_source_dir: Path,
    shared_anchor_df,
    scenario_trait_df,
    top_h_df,
    severe_drop_df,
    stable_df,
    heatmap_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Result 3.2B Formal Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- input_dir: `{input_dir.as_posix()}`")
    lines.append(f"- anchor_source_dir: `{anchor_source_dir.as_posix()}`")
    lines.append("- section meaning: H-only single-VI change under the formal shared-anchor rule")
    lines.append("- compared modality: `H_SINGLE_VI`")
    lines.append("")
    lines.append("## Shared Anchor")
    lines.append(to_text_table(shared_anchor_df))
    lines.append("")
    lines.append("## Scenario-Trait Overview")
    lines.append(to_text_table(scenario_trait_df))
    lines.append("")
    lines.append("## Top H-only VI by Scenario and Trait")
    lines.append(
        to_text_table(
            top_h_df[
                [
                    "scenario",
                    "target",
                    "rank",
                    "vi_label",
                    "h_mean_pearson",
                    "gh_minus_g",
                    "gh_minus_h",
                ]
            ]
        )
    )
    lines.append("")
    lines.append("## Most Severe Cross-Scenario Drop")
    lines.append(
        to_text_table(
            severe_drop_df[
                [
                    "target",
                    "vi_label",
                    "reference_h",
                    "within_season_h",
                    "loso_h",
                    "loso_genotype_h",
                    "mean_drop_from_reference",
                    "max_drop_from_reference",
                    "gh_minus_g_mean",
                ]
            ].head(24)
        )
    )
    lines.append("")
    lines.append("## Stable H-only VI")
    lines.append(
        to_text_table(
            stable_df[
                [
                    "target",
                    "vi_label",
                    "reference_h",
                    "within_season_h",
                    "loso_h",
                    "loso_genotype_h",
                    "h_mean_across_scenarios",
                    "h_range",
                    "gh_minus_g_mean",
                ]
            ].head(24)
        )
    )
    lines.append("")
    lines.append("## Figure")
    lines.append(f"- only-H VI heatmap: `{heatmap_path.as_posix()}`")
    (out_dir / "result_3_2b_formal_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.print_spec:
        print(f"section={SECTION_CODE}")
        print(f"scope={SECTION_SCOPE}")
        print(f"slug={SECTION_SLUG}")
        return 0

    input_dir = Path(args.input_dir).resolve()
    anchor_source_dir = Path(args.anchor_source_dir).resolve()
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d")
        out_dir = Path.cwd() / "outputs" / "reports" / f"result_3_2b_formal_analysis_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = read_required_csv(input_dir, "metrics_summary.csv")
    delta_df = read_required_csv(anchor_source_dir, "single_anchor_delta.csv")

    shared_anchor_df = build_shared_anchor_report_table(delta_df)
    same_vi_by_predictor_df = build_same_vi_by_predictor(metrics_df, delta_df)
    same_vi_summary_df = build_same_vi_summary(same_vi_by_predictor_df)
    top_h_df = build_top_h_vi(same_vi_summary_df)
    top_gh_df = build_top_gh_vi(same_vi_summary_df)
    scenario_trait_df = build_scenario_trait_overview(same_vi_summary_df, top_h_df, top_gh_df)
    cross_df, severe_drop_df, stable_df = build_cross_scenario_tables(same_vi_summary_df)

    shared_anchor_df.to_csv(out_dir / "shared_anchor_table.csv", index=False)
    same_vi_by_predictor_df.to_csv(out_dir / "same_vi_by_predictor.csv", index=False)
    same_vi_summary_df.to_csv(out_dir / "same_vi_by_scenario_trait.csv", index=False)
    top_h_df.to_csv(out_dir / "top_h_vi_by_scenario_trait.csv", index=False)
    scenario_trait_df.to_csv(out_dir / "scenario_trait_overview.csv", index=False)
    cross_df.to_csv(out_dir / "cross_scenario_stability_all.csv", index=False)
    severe_drop_df.to_csv(out_dir / "cross_scenario_severe_drop_reference_ge_030.csv", index=False)
    stable_df.to_csv(out_dir / "cross_scenario_stable_h_vi.csv", index=False)

    heatmap_path = out_dir / "result_3_2b_only_h_vi_trait_heatmaps.png"
    plot_scenario_heatmaps(
        same_vi_summary_df,
        value_col="h_mean_pearson",
        title="Result 3.2B: only-H single-VI accuracy across deployment scenarios",
        out_path=heatmap_path,
        cmap="YlGnBu",
        center=None,
    )
    write_json(
        out_dir / "run_notes.json",
        {
            "input_dir": str(input_dir),
            "anchor_source_dir": str(anchor_source_dir),
            "shared_anchor_rule": "trait_shared_mean_delta_across_scenarios_predictors",
        },
    )
    write_report(
        out_dir=out_dir,
        input_dir=input_dir,
        anchor_source_dir=anchor_source_dir,
        shared_anchor_df=shared_anchor_df,
        scenario_trait_df=scenario_trait_df,
        top_h_df=top_h_df,
        severe_drop_df=severe_drop_df,
        stable_df=stable_df,
        heatmap_path=heatmap_path,
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
