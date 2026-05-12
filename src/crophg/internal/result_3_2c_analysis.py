from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from crophg.common.report_utils import read_required_csv, to_text_table, write_json
from crophg.internal.result_3_2bc_analysis_common import (
    build_cross_scenario_tables,
    build_g_baseline_table,
    build_role_shift_tables,
    build_same_vi_by_predictor,
    build_same_vi_summary,
    build_scenario_trait_overview,
    build_shared_anchor_report_table,
    build_top_gh_vi,
    build_top_h_vi,
    plot_g_baseline_heatmap,
    plot_scenario_heatmaps,
)

SECTION_CODE = "3.2C"
SECTION_SCOPE = "internal"
SECTION_SLUG = "single_vi_increment_under_g_formal_analysis"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="crophg 3.2C formal analysis entry")
    parser.add_argument("--input-dir", type=str, required=True, help="Existing 3.2C result directory")
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
    g_baseline_df,
    scenario_trait_df,
    top_gh_df,
    useful_h_but_not_gh_df,
    h_weak_but_g_helps_df,
    g_heatmap_path: Path,
    gh_heatmap_path: Path,
    compatibility_note: str = "",
) -> None:
    lines: list[str] = []
    lines.append("# Result 3.2C Formal Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- input_dir: `{input_dir.as_posix()}`")
    lines.append(f"- anchor_source_dir: `{anchor_source_dir.as_posix()}`")
    lines.append("- section meaning: single-VI incremental value under G background")
    lines.append("- compared modality: `GH_SINGLE_VI` relative to `G`")
    if compatibility_note:
        lines.append(f"- compatibility: {compatibility_note}")
    lines.append("")
    lines.append("## Shared Anchor")
    lines.append(to_text_table(shared_anchor_df))
    lines.append("")
    lines.append("## G Baseline")
    lines.append(to_text_table(g_baseline_df))
    lines.append("")
    lines.append("## Scenario-Trait Overview")
    lines.append(to_text_table(scenario_trait_df))
    lines.append("")
    lines.append("## Top GH Increment VI by Scenario and Trait")
    lines.append(
        to_text_table(
            top_gh_df[
                [
                    "scenario",
                    "target",
                    "rank",
                    "vi_label",
                    "gh_minus_g",
                    "h_mean_pearson",
                    "gh_minus_h",
                ]
            ]
        )
    )
    lines.append("")
    lines.append("## H Useful but GH Gain Limited")
    lines.append(
        to_text_table(
            useful_h_but_not_gh_df[
                [
                    "target",
                    "vi_label",
                    "h_mean_across_scenarios",
                    "gh_minus_g_mean",
                    "h_range",
                    "reference_h",
                    "within_season_h",
                    "loso_h",
                    "loso_genotype_h",
                ]
            ].head(24)
        )
    )
    lines.append("")
    lines.append("## H Weak but GH Helps")
    lines.append(
        to_text_table(
            h_weak_but_g_helps_df[
                [
                    "target",
                    "vi_label",
                    "h_mean_across_scenarios",
                    "gh_minus_g_mean",
                    "h_range",
                    "reference_h",
                    "within_season_h",
                    "loso_h",
                    "loso_genotype_h",
                ]
            ].head(24)
        )
    )
    lines.append("")
    lines.append("## Figures")
    lines.append(f"- G baseline heatmap: `{g_heatmap_path.as_posix()}`")
    lines.append(f"- GH minus G heatmap: `{gh_heatmap_path.as_posix()}`")
    (out_dir / "result_3_2c_formal_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def _read_metrics_with_compatibility(input_dir: Path):
    """Read 3.2C metrics, falling back to sibling 3.2B combined metrics when needed."""
    requested_input_dir = input_dir
    metrics_path = input_dir / "metrics_summary.csv"
    compatibility_note = ""

    if not metrics_path.exists():
        sibling_3_2b = input_dir.parent / "3_2b"
        sibling_metrics = sibling_3_2b / "metrics_summary.csv"
        if sibling_metrics.exists():
            input_dir = sibling_3_2b
            metrics_path = sibling_metrics
            compatibility_note = (
                "requested input did not contain metrics_summary.csv; "
                "using sibling 3_2b combined H_SINGLE_VI/GH_SINGLE_VI metrics"
            )

    metrics_df = read_required_csv(input_dir, "metrics_summary.csv")
    if "modality_variant" not in metrics_df.columns:
        raise ValueError(f"{metrics_path} 缺少 modality_variant 列，无法执行 3.2C 分析。")

    modalities = set(metrics_df["modality_variant"].astype(str))
    if "GH_SINGLE_VI" not in modalities:
        sibling_3_2b = requested_input_dir.parent / "3_2b"
        sibling_metrics = sibling_3_2b / "metrics_summary.csv"
        if input_dir != sibling_3_2b and sibling_metrics.exists():
            sibling_df = read_required_csv(sibling_3_2b, "metrics_summary.csv")
            sibling_modalities = set(sibling_df["modality_variant"].astype(str))
            if "GH_SINGLE_VI" in sibling_modalities:
                input_dir = sibling_3_2b
                metrics_df = sibling_df
                compatibility_note = (
                    "requested input did not contain GH_SINGLE_VI; "
                    "using sibling 3_2b combined H_SINGLE_VI/GH_SINGLE_VI metrics"
                )
                modalities = sibling_modalities

    if "GH_SINGLE_VI" not in modalities:
        raise ValueError(
            f"{input_dir / 'metrics_summary.csv'} 不包含 GH_SINGLE_VI，无法生成 Result 3.2C。"
        )

    if input_dir.name == "3_2b" and not compatibility_note:
        compatibility_note = (
            "input is 3_2b combined metrics; 3.2C reuses GH_SINGLE_VI rows "
            "from the shared-anchor run instead of rerunning duplicate models"
        )

    return input_dir, metrics_df, compatibility_note


def main() -> int:
    args = build_parser().parse_args()
    if args.print_spec:
        print(f"section={SECTION_CODE}")
        print(f"scope={SECTION_SCOPE}")
        print(f"slug={SECTION_SLUG}")
        return 0

    requested_input_dir = Path(args.input_dir).resolve()
    input_dir = requested_input_dir
    anchor_source_dir = Path(args.anchor_source_dir).resolve()
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d")
        out_dir = Path.cwd() / "outputs" / "reports" / f"result_3_2c_formal_analysis_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_dir, metrics_df, compatibility_note = _read_metrics_with_compatibility(input_dir)
    delta_df = read_required_csv(anchor_source_dir, "single_anchor_delta.csv")

    shared_anchor_df = build_shared_anchor_report_table(delta_df)
    g_baseline_df = build_g_baseline_table(delta_df)
    same_vi_by_predictor_df = build_same_vi_by_predictor(metrics_df, delta_df)
    same_vi_summary_df = build_same_vi_summary(same_vi_by_predictor_df)
    top_h_df = build_top_h_vi(same_vi_summary_df)
    top_gh_df = build_top_gh_vi(same_vi_summary_df)
    scenario_trait_df = build_scenario_trait_overview(same_vi_summary_df, top_h_df, top_gh_df)
    cross_df, severe_drop_df, stable_df = build_cross_scenario_tables(same_vi_summary_df)
    useful_h_but_not_gh_df, h_weak_but_g_helps_df = build_role_shift_tables(cross_df)

    shared_anchor_df.to_csv(out_dir / "shared_anchor_table.csv", index=False)
    g_baseline_df.to_csv(out_dir / "g_baseline_by_scenario_trait.csv", index=False)
    same_vi_by_predictor_df.to_csv(out_dir / "same_vi_by_predictor.csv", index=False)
    same_vi_summary_df.to_csv(out_dir / "same_vi_by_scenario_trait.csv", index=False)
    top_gh_df.to_csv(out_dir / "top_gh_delta_vi_by_scenario_trait.csv", index=False)
    scenario_trait_df.to_csv(out_dir / "scenario_trait_overview.csv", index=False)
    cross_df.to_csv(out_dir / "cross_scenario_stability_all.csv", index=False)
    severe_drop_df.to_csv(out_dir / "cross_scenario_severe_drop_reference_ge_030.csv", index=False)
    stable_df.to_csv(out_dir / "cross_scenario_stable_h_vi.csv", index=False)
    useful_h_but_not_gh_df.to_csv(out_dir / "useful_h_but_not_gh.csv", index=False)
    h_weak_but_g_helps_df.to_csv(out_dir / "h_weak_but_g_helps.csv", index=False)

    g_heatmap_path = out_dir / "result_3_2c_g_baseline_heatmap.png"
    gh_heatmap_path = out_dir / "result_3_2c_gh_minus_g_vi_trait_heatmaps.png"
    plot_g_baseline_heatmap(g_baseline_df, g_heatmap_path)
    plot_scenario_heatmaps(
        same_vi_summary_df,
        value_col="gh_minus_g",
        title="Result 3.2C: single-VI G+H increment over G across deployment scenarios",
        out_path=gh_heatmap_path,
        cmap="RdBu_r",
        center=0,
    )
    write_json(
        out_dir / "run_notes.json",
        {
            "requested_input_dir": str(requested_input_dir),
            "input_dir": str(input_dir),
            "anchor_source_dir": str(anchor_source_dir),
            "compatibility_note": compatibility_note,
            "shared_anchor_rule": "trait_shared_mean_delta_across_scenarios_predictors",
        },
    )
    write_report(
        out_dir=out_dir,
        input_dir=input_dir,
        anchor_source_dir=anchor_source_dir,
        shared_anchor_df=shared_anchor_df,
        g_baseline_df=g_baseline_df,
        scenario_trait_df=scenario_trait_df,
        top_gh_df=top_gh_df,
        useful_h_but_not_gh_df=useful_h_but_not_gh_df,
        h_weak_but_g_helps_df=h_weak_but_g_helps_df,
        g_heatmap_path=g_heatmap_path,
        gh_heatmap_path=gh_heatmap_path,
        compatibility_note=compatibility_note,
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
