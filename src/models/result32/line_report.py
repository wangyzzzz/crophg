from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _safe_filename_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_").lower()
    return token or "item"


def _format_anchor_tb(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _short_predictor(name: str) -> str:
    mapping = {
        "lightgbm": "lgbm",
        "random_forest": "rf",
        "elasticnet": "enet",
        "lasso": "lasso",
        "ridge": "ridge",
    }
    return mapping.get(str(name), str(name))


def build_anchor_axis_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=["anchor_idx", "anchor_tb", "anchor_tb_label", "x_pos"])

    axis_df = (
        summary_df[["anchor_idx", "anchor_tb"]]
        .drop_duplicates()
        .sort_values("anchor_idx")
        .reset_index(drop=True)
    )
    axis_df["anchor_tb_label"] = axis_df["anchor_tb"].map(_format_anchor_tb)
    axis_df["x_pos"] = range(len(axis_df))
    return axis_df


def select_best_predictor_per_anchor(
    summary_df: pd.DataFrame,
    *,
    metric_col: str,
    tie_breaker_col: str | None = None,
) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()

    if metric_col not in summary_df.columns:
        raise KeyError(f"缺少指标列: {metric_col}")

    tie_breaker_col = tie_breaker_col or ("mean_pearson" if metric_col == "mean_r2" else "mean_r2")
    sort_cols = ["scenario", "target", "modality_combo", "anchor_idx", metric_col]
    ascending = [True, True, True, True, False]
    if tie_breaker_col in summary_df.columns:
        sort_cols.append(tie_breaker_col)
        ascending.append(False)

    ranked = summary_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    best = (
        ranked.groupby(["scenario", "target", "modality_combo", "anchor_idx"], as_index=False, sort=False)
        .first()
        .sort_values(["scenario", "target", "anchor_idx", "modality_combo"])
        .reset_index(drop=True)
    )
    return best


def generate_anchor_line_plots(
    *,
    summary_df: pd.DataFrame,
    output_dir: Path,
    scenario_order: list[str],
    target_order: list[str],
    modality_order: list[str],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    metric_specs = [
        ("mean_r2", "R2", "figure_r2_anchor_lineplot.png"),
        ("mean_pearson", "Pearson", "figure_pearson_anchor_lineplot.png"),
    ]
    palette = dict(zip(modality_order, sns.color_palette("tab10", n_colors=len(modality_order))))

    generated: list[Path] = []
    for metric_col, metric_label, file_name in metric_specs:
        best_df = select_best_predictor_per_anchor(summary_df, metric_col=metric_col)
        for scenario in scenario_order:
            for target in target_order:
                sub = best_df[(best_df["scenario"] == scenario) & (best_df["target"] == target)].copy()
                if sub.empty:
                    continue

                sub = sub.sort_values(["anchor_idx", "modality_combo"]).reset_index(drop=True)
                x_ticks = build_anchor_axis_table(sub)
                x_lookup = dict(zip(x_ticks["anchor_idx"], x_ticks["x_pos"]))

                fig, ax = plt.subplots(figsize=(10.8, 5.8), constrained_layout=True)
                for modality in modality_order:
                    line_df = sub[sub["modality_combo"] == modality].sort_values("anchor_idx")
                    if line_df.empty:
                        continue
                    ax.plot(
                        line_df["anchor_idx"].map(x_lookup).astype(float),
                        line_df[metric_col].astype(float),
                        marker="o",
                        linewidth=2.0,
                        markersize=4.5,
                        label=modality,
                        color=palette[modality],
                    )

                ax.set_title(f"{scenario} | {target} | {metric_label}")
                ax.set_xlabel("Anchor time (anchor_tb)")
                ax.set_ylabel(metric_label)
                ax.set_xticks(x_ticks["x_pos"].astype(float))
                ax.set_xticklabels(x_ticks["anchor_tb_label"].tolist(), rotation=0)
                ax.grid(True, axis="y", alpha=0.35)

                top_ax = ax.twiny()
                top_ax.set_xlim(ax.get_xlim())
                top_ax.set_xticks(x_ticks["x_pos"].astype(float))
                top_ax.set_xticklabels([f"A{int(x)}" for x in x_ticks["anchor_idx"]], fontsize=9)
                top_ax.set_xlabel("Anchor index")

                ax.legend(
                    title="Modality",
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=False,
                )

                # Explain which predictor won at the last anchor for each modality without cluttering the curve body.
                predictor_rows = (
                    sub.sort_values(["modality_combo", "anchor_idx"])
                    .groupby("modality_combo", as_index=False)
                    .tail(1)[["modality_combo", "predictor"]]
                )
                note = " | ".join(
                    f"{row['modality_combo']}={_short_predictor(row['predictor'])}"
                    for _, row in predictor_rows.sort_values("modality_combo").iterrows()
                )
                ax.text(
                    0.0,
                    -0.28,
                    f"Best predictor at latest anchor: {note}",
                    transform=ax.transAxes,
                    fontsize=8.5,
                    va="top",
                )

                out_path = output_dir / (
                    f"figure_{_safe_filename_token(scenario)}_{_safe_filename_token(target)}_{metric_label.lower()}_anchor_lineplot.png"
                )
                fig.savefig(out_path, dpi=220, bbox_inches="tight")
                plt.close(fig)
                generated.append(out_path)

    return generated
