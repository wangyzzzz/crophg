from __future__ import annotations

from pathlib import Path

import pandas as pd

from crophg.common.report_utils import markdown_table


DEFAULT_SCENARIOS = ["reference", "within_season", "loso", "loso_genotype"]
DEFAULT_PREDICTORS = ["ridge", "lasso", "elasticnet", "lightgbm", "random_forest"]


def build_shared_anchor_table(
    delta_df: pd.DataFrame,
    *,
    scenarios: list[str] | None = None,
    predictors: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = [str(x) for x in (scenarios or DEFAULT_SCENARIOS)]
    predictors = [str(x).lower() for x in (predictors or DEFAULT_PREDICTORS)]

    work = delta_df.copy()
    work["scenario"] = work["scenario"].astype(str)
    work["predictor"] = work["predictor"].astype(str).str.lower()
    work["target"] = work["target"].astype(str)
    work["timeline"] = work["timeline"].astype(str)
    work = work[work["scenario"].isin(scenarios)]
    work = work[work["predictor"].isin(predictors)]
    work = work[work["modality_variant"].astype(str) == "GH_SINGLE"]
    if work.empty:
        raise RuntimeError("筛选后 delta 数据为空，无法构建 shared anchor。")

    expected_units = len(scenarios) * len(predictors)
    summary = (
        work.groupby(
            ["target", "timeline", "anchor_idx", "anchor_tb", "anchor_phase", "anchor_band"],
            as_index=False,
        )
        .agg(
            mean_delta_pearson=("delta_pearson", "mean"),
            mean_gh_pearson=("mean_pearson", "mean"),
            mean_g_pearson=("g_mean_pearson", "mean"),
            n_units=("delta_pearson", "size"),
            n_scenarios=("scenario", "nunique"),
            n_predictors=("predictor", "nunique"),
        )
        .sort_values(
            ["target", "timeline", "mean_delta_pearson", "mean_gh_pearson", "anchor_idx"],
            ascending=[True, True, False, False, True],
        )
        .reset_index(drop=True)
    )
    summary["coverage_ratio"] = summary["n_units"] / float(expected_units)

    best = summary.groupby(["target", "timeline"], as_index=False).head(1).copy()
    best = best.rename(
        columns={
            "anchor_idx": "best_anchor",
            "anchor_tb": "best_anchor_tb",
            "anchor_phase": "best_anchor_phase",
            "anchor_band": "best_anchor_band",
            "mean_delta_pearson": "best_delta",
            "mean_gh_pearson": "best_mean_pearson",
            "mean_g_pearson": "g_mean_pearson",
        }
    )
    best["anchor_selection_rule"] = "trait_shared_mean_delta_across_scenarios_predictors"

    expanded_rows: list[dict[str, object]] = []
    for _, row in best.iterrows():
        for scenario in scenarios:
            for predictor in predictors:
                expanded_rows.append(
                    {
                        "scenario": scenario,
                        "target": row["target"],
                        "timeline": row["timeline"],
                        "predictor": predictor,
                        "best_anchor": int(row["best_anchor"]),
                        "best_anchor_tb": row["best_anchor_tb"],
                        "best_anchor_phase": row["best_anchor_phase"],
                        "best_anchor_band": row["best_anchor_band"],
                        "best_delta": row["best_delta"],
                        "best_mean_pearson": row["best_mean_pearson"],
                        "g_mean_pearson": row["g_mean_pearson"],
                        "anchor_selection_rule": row["anchor_selection_rule"],
                        "coverage_ratio": row["coverage_ratio"],
                        "n_units": int(row["n_units"]),
                        "n_scenarios": int(row["n_scenarios"]),
                        "n_predictors": int(row["n_predictors"]),
                    }
                )

    expanded = pd.DataFrame(expanded_rows).sort_values(
        ["scenario", "target", "timeline", "predictor"]
    ).reset_index(drop=True)
    return expanded, best.reset_index(drop=True)


def write_shared_anchor_artifacts(
    *,
    delta_csv: Path,
    output_csv: Path,
    summary_md: Path,
    scenarios: list[str] | None = None,
    predictors: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    delta_df = pd.read_csv(delta_csv)
    expanded, best = build_shared_anchor_table(
        delta_df,
        scenarios=scenarios,
        predictors=predictors,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_csv(output_csv, index=False)

    best_show = best.copy()
    for col in ["best_delta", "best_mean_pearson", "g_mean_pearson", "coverage_ratio"]:
        best_show[col] = pd.to_numeric(best_show[col], errors="coerce").round(4)

    lines: list[str] = []
    lines.append("# Shared Anchor Selection")
    lines.append("")
    lines.append("## Rule")
    lines.append("")
    lines.append(
        "对每个 trait，在 `single_anchor_delta.csv` 中，选择 `GH-G` 的 `delta_pearson` 在指定场景与模型上平均值最大的单个 anchor。"
    )
    lines.append("")
    lines.append(f"- scenarios: `{', '.join(scenarios or DEFAULT_SCENARIOS)}`")
    lines.append(f"- predictors: `{', '.join([str(x).lower() for x in (predictors or DEFAULT_PREDICTORS)])}`")
    lines.append("")
    lines.append("## Selected Shared Anchors")
    lines.append("")
    lines.append(
        markdown_table(
            best_show[
                [
                    "target",
                    "timeline",
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
            ],
            max_rows=30,
        )
    )
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    return expanded, best
