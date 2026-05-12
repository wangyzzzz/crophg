from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from crophg.common.report_utils import read_required_csv, to_text_table, write_json

SECTION_CODE = "3.1A"
SECTION_SCOPE = "internal"
SECTION_SLUG = "h_only_deployment_loss_formal_analysis"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="crophg 3.1A formal analysis entry")
    parser.add_argument("--input-dir", type=str, required=True, help="Existing 3.1A result directory")
    parser.add_argument("--output-dir", type=str, default="", help="Optional output report directory")
    parser.add_argument("--print-spec", action="store_true", help="Print section scaffold metadata and exit")
    return parser


def build_best_accuracy_table(metrics_by_scenario_target: pd.DataFrame) -> pd.DataFrame:
    return metrics_by_scenario_target.copy().sort_values(["scenario", "target"]).reset_index(drop=True)


def build_scenario_mean_table(best_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        best_df.groupby("scenario", as_index=False)
        .agg(
            mean_best_r2=("best_r2", "mean"),
            mean_best_pearson=("best_pearson", "mean"),
            mean_avg_r2=("avg_r2", "mean"),
            mean_avg_pearson=("avg_pearson", "mean"),
        )
        .sort_values("scenario")
        .reset_index(drop=True)
    )
    return out


def build_trait_gap_table(best_df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        best_df.pivot_table(index="target", columns="scenario", values="best_pearson", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in ["reference", "within_season", "loso", "loso_genotype"]:
        if col not in pivot.columns:
            pivot[col] = float("nan")
    pivot["gap_reference_to_loso"] = pivot["reference"] - pivot["loso"]
    pivot["gap_reference_to_loso_genotype"] = pivot["reference"] - pivot["loso_genotype"]
    pivot["gap_loso_to_loso_genotype"] = pivot["loso"] - pivot["loso_genotype"]
    return pivot.sort_values("target").reset_index(drop=True)


def write_report(
    *,
    out_dir: Path,
    input_dir: Path,
    feature_overview_df: pd.DataFrame,
    scenario_mean_df: pd.DataFrame,
    best_df: pd.DataFrame,
    trait_gap_df: pd.DataFrame,
) -> None:
    worst_row = trait_gap_df.sort_values("gap_reference_to_loso_genotype", ascending=False).iloc[0]

    lines: list[str] = []
    lines.append("# Result 3.1A Formal Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- input_dir: `{input_dir.as_posix()}`")
    lines.append("- section meaning: H-only deployment loss across four scenarios")
    lines.append("- modality: `H_FULL`")
    lines.append("")
    lines.append("## Feature Overview")
    lines.append(to_text_table(feature_overview_df))
    lines.append("")
    lines.append("## Scenario Mean")
    lines.append(to_text_table(scenario_mean_df))
    lines.append("")
    lines.append("## Scenario-Trait Best Accuracy")
    lines.append(to_text_table(best_df))
    lines.append("")
    lines.append("## Trait Deployment Gaps")
    lines.append(to_text_table(trait_gap_df))
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        f"- Largest `Reference -> Joint-Novel` Pearson drop in this run: `{worst_row['target']}` with gap `{float(worst_row['gap_reference_to_loso_genotype']):.4f}`."
    )
    (out_dir / "result_3_1a_formal_analysis.md").write_text("\n".join(lines), encoding="utf-8")


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
        out_dir = Path.cwd() / "outputs" / "reports" / f"result_3_1a_formal_analysis_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_overview_df = read_required_csv(input_dir, "feature_overview.csv")
    best_df = build_best_accuracy_table(read_required_csv(input_dir, "metrics_by_scenario_target.csv"))
    scenario_mean_df = build_scenario_mean_table(best_df)
    trait_gap_df = build_trait_gap_table(best_df)

    feature_overview_df.to_csv(out_dir / "feature_overview.csv", index=False)
    best_df.to_csv(out_dir / "best_accuracy_by_scenario_target.csv", index=False)
    scenario_mean_df.to_csv(out_dir / "scenario_mean_summary.csv", index=False)
    trait_gap_df.to_csv(out_dir / "trait_deployment_gaps.csv", index=False)
    write_json(
        out_dir / "run_notes.json",
        {"input_dir": str(input_dir), "modality": "H_FULL", "section": SECTION_CODE},
    )
    write_report(
        out_dir=out_dir,
        input_dir=input_dir,
        feature_overview_df=feature_overview_df,
        scenario_mean_df=scenario_mean_df,
        best_df=best_df,
        trait_gap_df=trait_gap_df,
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
