from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from crophg.common.report_utils import read_required_csv, to_text_table, write_json

SECTION_CODE = "3.1B"
SECTION_SCOPE = "internal"
SECTION_SLUG = "g_compensation_and_fullh_complementarity_formal_analysis"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="crophg 3.1B formal analysis entry")
    parser.add_argument("--input-dir", type=str, required=True, help="Existing 3.1B result directory")
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


def build_trait_table(best_df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        best_df.pivot_table(index="target", columns="scenario", values="best_pearson", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in ["reference", "within_season", "loso", "loso_genotype"]:
        if col not in pivot.columns:
            pivot[col] = float("nan")
    pivot["mean_across_scenarios"] = pivot[["reference", "within_season", "loso", "loso_genotype"]].mean(axis=1)
    return pivot.sort_values("target").reset_index(drop=True)


def write_report(
    *,
    out_dir: Path,
    input_dir: Path,
    feature_overview_df: pd.DataFrame,
    scenario_mean_df: pd.DataFrame,
    best_df: pd.DataFrame,
    trait_df: pd.DataFrame,
) -> None:
    best_scenario = scenario_mean_df.sort_values("mean_best_pearson", ascending=False).iloc[0]

    lines: list[str] = []
    lines.append("# Result 3.1B Formal Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- input_dir: `{input_dir.as_posix()}`")
    lines.append("- section meaning: G compensation and G+FULLH complementarity")
    lines.append("- modality: `G+FULLH`")
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
    lines.append("## Trait Mean Across Scenarios")
    lines.append(to_text_table(trait_df))
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        f"- Best scenario-level mean Pearson in this run: `{best_scenario['scenario']}` with `{float(best_scenario['mean_best_pearson']):.4f}`."
    )
    (out_dir / "result_3_1b_formal_analysis.md").write_text("\n".join(lines), encoding="utf-8")


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
        out_dir = Path.cwd() / "outputs" / "reports" / f"result_3_1b_formal_analysis_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_overview_df = read_required_csv(input_dir, "feature_overview.csv")
    best_df = build_best_accuracy_table(read_required_csv(input_dir, "metrics_by_scenario_target.csv"))
    scenario_mean_df = build_scenario_mean_table(best_df)
    trait_df = build_trait_table(best_df)

    feature_overview_df.to_csv(out_dir / "feature_overview.csv", index=False)
    best_df.to_csv(out_dir / "best_accuracy_by_scenario_target.csv", index=False)
    scenario_mean_df.to_csv(out_dir / "scenario_mean_summary.csv", index=False)
    trait_df.to_csv(out_dir / "trait_mean_across_scenarios.csv", index=False)
    write_json(
        out_dir / "run_notes.json",
        {"input_dir": str(input_dir), "modality": "G+FULLH", "section": SECTION_CODE},
    )
    write_report(
        out_dir=out_dir,
        input_dir=input_dir,
        feature_overview_df=feature_overview_df,
        scenario_mean_df=scenario_mean_df,
        best_df=best_df,
        trait_df=trait_df,
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
