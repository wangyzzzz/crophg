from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from crophg.common.report_utils import read_required_csv, to_text_table, write_json

SECTION_CODE = "3.2A"
SECTION_SCOPE = "internal"
SECTION_SLUG = "effective_windows_across_scenarios_formal_analysis"
SCENARIOS = ["reference", "within_season", "loso", "loso_genotype"]
TARGETS = ["ActualYD", "CM", "LM", "PHM", "Spike", "TKW"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="crophg 3.2A formal analysis entry")
    parser.add_argument("--input-dir", type=str, required=True, help="Existing 3.2A result directory")
    parser.add_argument("--output-dir", type=str, default="", help="Optional output report directory")
    parser.add_argument("--print-spec", action="store_true", help="Print section scaffold metadata and exit")
    return parser


def anchor_label(anchor_tb: float | int, anchor_phase: str | None) -> str:
    tb_int = int(anchor_tb)
    return f"H+{tb_int}" if str(anchor_phase) == "head" else f"S+{tb_int}"


def build_anchor_delta_grouped(delta_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        delta_df.groupby(["scenario", "target", "anchor_idx", "anchor_tb", "anchor_phase"], as_index=False)
        .agg(
            mean_delta_pearson=("delta_pearson", "mean"),
            mean_delta_r2=("delta_r2", "mean"),
        )
        .sort_values(["scenario", "target", "anchor_idx"])
        .reset_index(drop=True)
    )
    return grouped


def build_anchor_delta_overview(grouped_df: pd.DataFrame) -> pd.DataFrame:
    overview = (
        grouped_df.groupby(["scenario", "target"], as_index=False)
        .agg(max_mean_delta_pearson=("mean_delta_pearson", "max"))
        .sort_values(["scenario", "max_mean_delta_pearson"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return overview


def build_top_anchor_table(delta_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    top = (
        delta_df.groupby(
            ["scenario", "target", "anchor_idx", "anchor_tb", "anchor_phase", "anchor_band"],
            as_index=False,
        )
        .agg(
            mean_delta_pearson=("delta_pearson", "mean"),
            mean_delta_r2=("delta_r2", "mean"),
            mean_pearson=("mean_pearson", "mean"),
            mean_g_pearson=("g_mean_pearson", "mean"),
        )
        .sort_values(["scenario", "target", "mean_delta_pearson"], ascending=[True, True, False])
        .groupby(["scenario", "target"], as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    top["rank"] = top.groupby(["scenario", "target"]).cumcount() + 1
    return top


def build_best_anchor_consistency(best_df: pd.DataFrame) -> pd.DataFrame:
    consistency = (
        best_df.groupby(["scenario", "target", "best_anchor_phase", "best_anchor_band"], as_index=False)
        .size()
        .rename(columns={"size": "n_predictors"})
        .sort_values(["scenario", "target", "n_predictors"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return consistency


def build_factor_importance_summary(factors_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        factors_df.groupby(["scenario", "target", "factor_group"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            mean_importance_r2=("importance_r2", "mean"),
        )
        .sort_values(["scenario", "target", "mean_importance"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return summary


def plot_anchor_delta_heatmap(grouped_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    anchor_meta = (
        grouped_df[["anchor_idx", "anchor_tb", "anchor_phase"]]
        .drop_duplicates()
        .sort_values("anchor_idx")
        .reset_index(drop=True)
    )
    if anchor_meta.empty:
        raise RuntimeError("3.2A grouped delta table is empty; cannot plot heatmap.")

    vmin = float(grouped_df["mean_delta_pearson"].min())
    vmax = float(grouped_df["mean_delta_pearson"].max())
    x_tick_positions = [i + 0.5 for i in anchor_meta["anchor_idx"].tolist()]
    x_tick_labels = [
        anchor_label(tb, phase)
        for tb, phase in zip(anchor_meta["anchor_tb"], anchor_meta["anchor_phase"])
    ]

    head_boundary_x: float | None = None
    head_rows = anchor_meta.loc[anchor_meta["anchor_phase"].astype(str) == "head", "anchor_idx"]
    if not head_rows.empty:
        head_boundary_x = float(head_rows.min())

    for ax, scenario in zip(axes.flat, SCENARIOS):
        sub = grouped_df.loc[grouped_df["scenario"].astype(str) == scenario].copy()
        pivot = (
            sub.pivot(index="target", columns="anchor_idx", values="mean_delta_pearson")
            .reindex(index=TARGETS, columns=anchor_meta["anchor_idx"].tolist())
        )
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=vmin,
            vmax=vmax,
            cbar=ax is axes.flat[0],
            linewidths=0.35,
            linecolor="white",
        )
        ax.set_title(scenario)
        ax.set_xlabel("Anchor")
        ax.set_ylabel("Trait")
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
        if head_boundary_x is not None:
            ax.axvline(head_boundary_x, color="black", linestyle="--", linewidth=1.2, alpha=0.9)

    fig.suptitle("Result 3.2A: mean GH-G delta across single anchors", fontsize=16)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    *,
    out_dir: Path,
    input_dir: Path,
    overview_df: pd.DataFrame,
    top_anchor_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    heatmap_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Result 3.2A Formal Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- input_dir: `{input_dir.as_posix()}`")
    lines.append("- section meaning: effective G+H windows across deployment scenarios")
    lines.append("- comparison unit: `G` vs `GH_SINGLE` under single-anchor evaluation")
    lines.append("")
    lines.append("## Scenario-Trait Overview")
    lines.append(to_text_table(overview_df))
    lines.append("")
    lines.append("## Top Anchors by Scenario and Trait")
    lines.append(to_text_table(top_anchor_df))
    lines.append("")
    lines.append("## Best-Anchor Consistency Across Predictors")
    lines.append(to_text_table(consistency_df, float_fmt="{:.1f}"))
    lines.append("")
    lines.append("## Factor Group Importance")
    lines.append(to_text_table(factor_df))
    lines.append("")
    lines.append("## Figure")
    lines.append(f"- anchor delta heatmap: `{heatmap_path.as_posix()}`")
    (out_dir / "result_3_2a_formal_analysis.md").write_text("\n".join(lines), encoding="utf-8")


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
        out_dir = Path.cwd() / "outputs" / "reports" / f"result_3_2a_formal_analysis_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    delta_df = read_required_csv(input_dir, "single_anchor_delta.csv")
    best_df = read_required_csv(input_dir, "best_anchor_by_trait_scenario.csv")
    factor_df = read_required_csv(input_dir, "factor_group_importance.csv")

    grouped_df = build_anchor_delta_grouped(delta_df)
    overview_df = build_anchor_delta_overview(grouped_df)
    top_anchor_df = build_top_anchor_table(delta_df)
    consistency_df = build_best_anchor_consistency(best_df)
    factor_summary_df = build_factor_importance_summary(factor_df)

    grouped_df.to_csv(out_dir / "anchor_delta_grouped.csv", index=False)
    overview_df.to_csv(out_dir / "anchor_delta_overview.csv", index=False)
    top_anchor_df.to_csv(out_dir / "top_anchor_by_scenario_trait.csv", index=False)
    consistency_df.to_csv(out_dir / "best_anchor_consistency.csv", index=False)
    factor_summary_df.to_csv(out_dir / "factor_group_importance_summary.csv", index=False)

    heatmap_path = out_dir / "result_3_2a_anchor_delta_heatmap.png"
    plot_anchor_delta_heatmap(grouped_df, heatmap_path)
    write_json(
        out_dir / "run_notes.json",
        {
            "input_dir": str(input_dir),
            "n_grouped_rows": int(len(grouped_df)),
            "n_top_rows": int(len(top_anchor_df)),
        },
    )
    write_report(
        out_dir=out_dir,
        input_dir=input_dir,
        overview_df=overview_df,
        top_anchor_df=top_anchor_df,
        consistency_df=consistency_df,
        factor_df=factor_summary_df,
        heatmap_path=heatmap_path,
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
