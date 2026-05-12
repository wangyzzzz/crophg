from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from crophg.internal.shared_anchor import build_shared_anchor_table as select_shared_anchor_table

SCENARIOS = ["reference", "within_season", "loso", "loso_genotype"]
TARGETS = ["ActualYD", "CM", "LM", "PHM", "Spike", "TKW"]
PREDICTORS = ["ridge", "lasso", "elasticnet", "lightgbm", "random_forest"]
VI_ORDER = [
    "vi_evi2",
    "vi_gndvi",
    "vi_grvi",
    "vi_mgrvi",
    "vi_msavi",
    "vi_msr",
    "vi_ndre",
    "vi_ndvi",
    "vi_osavi",
    "vi_rdvi",
    "vi_savi",
    "vi_vari",
]


def vi_label(name: str) -> str:
    return str(name).replace("vi_", "").upper()


def anchor_label(anchor_tb: float | int, anchor_phase: str) -> str:
    tb_int = int(anchor_tb)
    return f"H+{tb_int}" if str(anchor_phase) == "head" else f"S+{tb_int}"


def build_shared_anchor_report_table(delta_df: pd.DataFrame) -> pd.DataFrame:
    _, best = select_shared_anchor_table(delta_df)
    table = best.copy().sort_values("target").reset_index(drop=True)
    table["anchor_label"] = [
        anchor_label(tb, phase)
        for tb, phase in zip(table["best_anchor_tb"], table["best_anchor_phase"])
    ]
    return table[
        [
            "target",
            "timeline",
            "anchor_label",
            "best_anchor",
            "best_anchor_tb",
            "best_anchor_phase",
            "best_anchor_band",
            "best_delta",
            "best_mean_pearson",
            "g_mean_pearson",
            "coverage_ratio",
            "n_units",
        ]
    ].copy()


def build_g_baseline_table(delta_df: pd.DataFrame) -> pd.DataFrame:
    work = delta_df.copy()
    work["predictor"] = work["predictor"].astype(str).str.lower()
    g_df = (
        work.loc[work["predictor"].isin(PREDICTORS), ["scenario", "target", "predictor", "g_mean_pearson"]]
        .drop_duplicates()
        .groupby(["scenario", "target"], as_index=False)
        .agg(
            n_predictors=("predictor", "nunique"),
            g_mean_pearson=("g_mean_pearson", "mean"),
        )
        .sort_values(["scenario", "target"])
        .reset_index(drop=True)
    )
    return g_df


def build_same_vi_by_predictor(metrics_df: pd.DataFrame, delta_df: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics_df.copy()
    metrics["predictor"] = metrics["predictor"].astype(str).str.lower()
    wide = (
        metrics.pivot_table(
            index=[
                "scenario",
                "target",
                "predictor",
                "anchor_idx",
                "anchor_tb",
                "anchor_phase",
                "anchor_band",
                "vi_name",
            ],
            columns="modality_variant",
            values="mean_pearson",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["scenario", "target", "predictor", "vi_name"])
        .reset_index(drop=True)
    )
    for col in ["H_SINGLE_VI", "GH_SINGLE_VI"]:
        if col not in wide.columns:
            wide[col] = float("nan")
    wide["vi_label"] = wide["vi_name"].map(vi_label)

    g_lookup = (
        delta_df.copy()[["scenario", "target", "predictor", "g_mean_pearson"]]
        .assign(predictor=lambda d: d["predictor"].astype(str).str.lower())
        .drop_duplicates()
        .loc[lambda d: d["predictor"].isin(PREDICTORS)]
    )
    wide = wide.merge(g_lookup, on=["scenario", "target", "predictor"], how="left")
    wide["gh_minus_g"] = wide["GH_SINGLE_VI"] - wide["g_mean_pearson"]
    wide["h_minus_g"] = wide["H_SINGLE_VI"] - wide["g_mean_pearson"]
    wide["gh_minus_h"] = wide["GH_SINGLE_VI"] - wide["H_SINGLE_VI"]
    return wide


def build_same_vi_summary(wide_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        wide_df.groupby(
            ["scenario", "target", "anchor_idx", "anchor_tb", "anchor_phase", "anchor_band", "vi_name", "vi_label"],
            as_index=False,
        )
        .agg(
            n_predictors=("predictor", "nunique"),
            g_mean_pearson=("g_mean_pearson", "mean"),
            h_mean_pearson=("H_SINGLE_VI", "mean"),
            gh_mean_pearson=("GH_SINGLE_VI", "mean"),
            gh_minus_g=("gh_minus_g", "mean"),
            h_minus_g=("h_minus_g", "mean"),
            gh_minus_h=("gh_minus_h", "mean"),
        )
        .sort_values(["scenario", "target", "vi_name"])
        .reset_index(drop=True)
    )
    return summary


def build_top_h_vi(summary_df: pd.DataFrame) -> pd.DataFrame:
    top_h = (
        summary_df.sort_values(
            ["scenario", "target", "h_mean_pearson", "gh_minus_g"],
            ascending=[True, True, False, False],
        )
        .groupby(["scenario", "target"], as_index=False)
        .head(3)
        .copy()
    )
    top_h["rank"] = top_h.groupby(["scenario", "target"]).cumcount() + 1
    return top_h


def build_top_gh_vi(summary_df: pd.DataFrame) -> pd.DataFrame:
    top_gh = (
        summary_df.sort_values(
            ["scenario", "target", "gh_minus_g", "h_mean_pearson"],
            ascending=[True, True, False, False],
        )
        .groupby(["scenario", "target"], as_index=False)
        .head(3)
        .copy()
    )
    top_gh["rank"] = top_gh.groupby(["scenario", "target"]).cumcount() + 1
    return top_gh


def build_scenario_trait_overview(summary_df: pd.DataFrame, top_h_df: pd.DataFrame, top_gh_df: pd.DataFrame) -> pd.DataFrame:
    scenario_trait = (
        summary_df.groupby(["scenario", "target"], as_index=False)
        .agg(
            g_mean_pearson=("g_mean_pearson", "mean"),
            h_mean_pearson=("h_mean_pearson", "mean"),
            gh_mean_pearson=("gh_mean_pearson", "mean"),
            gh_minus_g=("gh_minus_g", "mean"),
            gh_minus_h=("gh_minus_h", "mean"),
        )
        .sort_values(["scenario", "target"])
        .reset_index(drop=True)
    )
    top_h_best = top_h_df.loc[top_h_df["rank"] == 1, ["scenario", "target", "vi_label", "h_mean_pearson"]].rename(
        columns={"vi_label": "best_h_vi", "h_mean_pearson": "best_h_value"}
    )
    top_gh_best = top_gh_df.loc[top_gh_df["rank"] == 1, ["scenario", "target", "vi_label", "gh_minus_g"]].rename(
        columns={"vi_label": "best_gh_delta_vi", "gh_minus_g": "best_gh_delta_value"}
    )
    return (
        scenario_trait.merge(top_h_best, on=["scenario", "target"], how="left")
        .merge(top_gh_best, on=["scenario", "target"], how="left")
        .sort_values(["scenario", "target"])
        .reset_index(drop=True)
    )


def build_cross_scenario_tables(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pivot_h = (
        summary_df.pivot_table(
            index=["target", "vi_name", "vi_label"],
            columns="scenario",
            values="h_mean_pearson",
            aggfunc="mean",
        )
        .reset_index()
    )
    pivot_gh = (
        summary_df.pivot_table(
            index=["target", "vi_name", "vi_label"],
            columns="scenario",
            values="gh_minus_g",
            aggfunc="mean",
        )
        .reset_index()
    )
    pivot_h = pivot_h.rename(columns={scenario: f"{scenario}_h" for scenario in SCENARIOS})
    pivot_gh = pivot_gh.rename(columns={scenario: f"{scenario}_ghg" for scenario in SCENARIOS})
    merged = pivot_h.merge(pivot_gh, on=["target", "vi_name", "vi_label"], how="left")

    for scenario in SCENARIOS:
        h_col = f"{scenario}_h"
        gh_col = f"{scenario}_ghg"
        if h_col not in merged.columns:
            merged[h_col] = float("nan")
        if gh_col not in merged.columns:
            merged[gh_col] = float("nan")

    merged["h_mean_across_scenarios"] = merged[[f"{s}_h" for s in SCENARIOS]].mean(axis=1)
    merged["h_min"] = merged[[f"{s}_h" for s in SCENARIOS]].min(axis=1)
    merged["h_max"] = merged[[f"{s}_h" for s in SCENARIOS]].max(axis=1)
    merged["h_range"] = merged["h_max"] - merged["h_min"]
    merged["h_std"] = merged[[f"{s}_h" for s in SCENARIOS]].std(axis=1, ddof=0)
    merged["gh_minus_g_mean"] = merged[[f"{s}_ghg" for s in SCENARIOS]].mean(axis=1)
    merged["gh_minus_g_min"] = merged[[f"{s}_ghg" for s in SCENARIOS]].min(axis=1)
    merged["gh_minus_g_max"] = merged[[f"{s}_ghg" for s in SCENARIOS]].max(axis=1)
    merged["reference_h"] = merged["reference_h"]
    merged["mean_drop_from_reference"] = merged["reference_h"] - merged[
        ["within_season_h", "loso_h", "loso_genotype_h"]
    ].mean(axis=1)
    merged["max_drop_from_reference"] = merged["reference_h"] - merged[
        ["within_season_h", "loso_h", "loso_genotype_h"]
    ].min(axis=1)

    severe_drop = (
        merged.loc[merged["reference_h"] >= 0.30]
        .sort_values(
            ["mean_drop_from_reference", "max_drop_from_reference", "reference_h"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    stable = (
        merged.loc[merged["h_mean_across_scenarios"] >= 0.20]
        .sort_values(["h_range", "h_std", "h_mean_across_scenarios"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return merged, severe_drop, stable


def build_role_shift_tables(merged_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    useful_h_but_not_gh = (
        merged_df.loc[(merged_df["h_mean_across_scenarios"] >= 0.30) & (merged_df["gh_minus_g_mean"] <= 0.03)]
        .sort_values(["h_mean_across_scenarios", "gh_minus_g_mean"], ascending=[False, True])
        .reset_index(drop=True)
    )
    h_weak_but_g_helps = (
        merged_df.loc[(merged_df["h_mean_across_scenarios"] <= 0.20) & (merged_df["gh_minus_g_mean"] >= 0.05)]
        .sort_values(["gh_minus_g_mean", "h_mean_across_scenarios"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return useful_h_but_not_gh, h_weak_but_g_helps


def plot_scenario_heatmaps(
    summary_df: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    out_path: Path,
    cmap: str,
    center: float | None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    vi_labels = [vi_label(v) for v in VI_ORDER]

    if center is None:
        vmin = float(summary_df[value_col].min())
        vmax = float(summary_df[value_col].max())
    else:
        vmax = max(abs(float(summary_df[value_col].min())), abs(float(summary_df[value_col].max())), 0.02)
        vmin = -vmax

    for ax, scenario in zip(axes.flat, SCENARIOS):
        sub = summary_df.loc[summary_df["scenario"].astype(str) == scenario].copy()
        pivot = (
            sub.pivot(index="target", columns="vi_label", values=value_col)
            .reindex(index=TARGETS, columns=vi_labels)
        )
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.35,
            linecolor="white",
            cbar=ax is axes.flat[0],
        )
        ax.set_title(scenario)
        ax.set_xlabel("Vegetation index")
        ax.set_ylabel("Trait")
        ax.tick_params(axis="x", labelrotation=45)
        ax.tick_params(axis="y", labelrotation=0)

    fig.suptitle(title, fontsize=16)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_g_baseline_heatmap(g_df: pd.DataFrame, out_path: Path) -> None:
    pivot = g_df.pivot(index="target", columns="scenario", values="g_mean_pearson").reindex(index=TARGETS, columns=SCENARIOS)
    fig, ax = plt.subplots(figsize=(8.4, 5.3), constrained_layout=True)
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlGnBu",
        annot=True,
        fmt=".3f",
        linewidths=0.45,
        linecolor="white",
        cbar=True,
    )
    ax.set_title("G baseline Pearson across deployment scenarios")
    ax.set_xlabel("Deployment scenario")
    ax.set_ylabel("Trait")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
