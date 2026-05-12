from __future__ import annotations

import pandas as pd

from crophg.internal.result_3_2a_analysis import (
    build_anchor_delta_grouped,
    build_anchor_delta_overview,
    build_best_anchor_consistency,
    build_factor_importance_summary,
    build_top_anchor_table,
)
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
)
from crophg.internal.result_3_2c_analysis import _read_metrics_with_compatibility


def _mock_delta_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in ["reference", "within_season", "loso", "loso_genotype"]:
        for predictor in ["ridge", "lasso", "elasticnet", "lightgbm", "random_forest"]:
            rows.append(
                {
                    "scenario": scenario,
                    "target": "ActualYD",
                    "timeline": "gdd_rel_heading",
                    "modality_variant": "GH_SINGLE",
                    "predictor": predictor,
                    "anchor_idx": 1,
                    "anchor_tb": 150,
                    "anchor_phase": "sow",
                    "anchor_band": "early",
                    "mean_pearson": 0.55,
                    "g_mean_pearson": 0.40,
                    "delta_pearson": 0.15,
                    "delta_r2": 0.10,
                }
            )
            rows.append(
                {
                    "scenario": scenario,
                    "target": "ActualYD",
                    "timeline": "gdd_rel_heading",
                    "modality_variant": "GH_SINGLE",
                    "predictor": predictor,
                    "anchor_idx": 2,
                    "anchor_tb": 70,
                    "anchor_phase": "head",
                    "anchor_band": "late",
                    "mean_pearson": 0.62,
                    "g_mean_pearson": 0.40,
                    "delta_pearson": 0.22,
                    "delta_r2": 0.20,
                }
            )
    return pd.DataFrame(rows)


def _mock_best_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": "reference",
                "target": "ActualYD",
                "best_anchor_phase": "head",
                "best_anchor_band": "late",
            },
            {
                "scenario": "reference",
                "target": "ActualYD",
                "best_anchor_phase": "head",
                "best_anchor_band": "late",
            },
        ]
    )


def _mock_factor_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": "reference",
                "target": "ActualYD",
                "factor_group": "VI",
                "importance": 0.16,
                "importance_r2": 0.50,
            },
            {
                "scenario": "reference",
                "target": "ActualYD",
                "factor_group": "time",
                "importance": 0.02,
                "importance_r2": 0.01,
            },
        ]
    )


def _mock_metrics_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    val_map = {
        ("reference", "vi_ndre"): (0.67, 0.64),
        ("within_season", "vi_ndre"): (0.63, 0.60),
        ("loso", "vi_ndre"): (0.59, 0.57),
        ("loso_genotype", "vi_ndre"): (0.61, 0.58),
        ("reference", "vi_gndvi"): (0.58, 0.53),
        ("within_season", "vi_gndvi"): (0.56, 0.52),
        ("loso", "vi_gndvi"): (0.51, 0.49),
        ("loso_genotype", "vi_gndvi"): (0.54, 0.50),
    }
    for scenario in ["reference", "within_season", "loso", "loso_genotype"]:
        for predictor in ["ridge", "lasso", "elasticnet"]:
            for vi_name in ["vi_ndre", "vi_gndvi"]:
                h_val, gh_val = val_map[(scenario, vi_name)]
                rows.append(
                    {
                        "scenario": scenario,
                        "target": "ActualYD",
                        "predictor": predictor,
                        "anchor_idx": 2,
                        "anchor_tb": 70,
                        "anchor_phase": "head",
                        "anchor_band": "late",
                        "vi_name": vi_name,
                        "modality_variant": "H_SINGLE_VI",
                        "mean_pearson": h_val,
                    }
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "target": "ActualYD",
                        "predictor": predictor,
                        "anchor_idx": 2,
                        "anchor_tb": 70,
                        "anchor_phase": "head",
                        "anchor_band": "late",
                        "vi_name": vi_name,
                        "modality_variant": "GH_SINGLE_VI",
                        "mean_pearson": gh_val,
                    }
                )
    return pd.DataFrame(rows)


def test_result_3_2a_analysis_builders() -> None:
    delta_df = _mock_delta_df()
    grouped = build_anchor_delta_grouped(delta_df)
    overview = build_anchor_delta_overview(grouped)
    top = build_top_anchor_table(delta_df)
    consistency = build_best_anchor_consistency(_mock_best_df())
    factor = build_factor_importance_summary(_mock_factor_df())

    assert not grouped.empty
    assert float(overview.iloc[0]["max_mean_delta_pearson"]) > 0
    assert int(top.iloc[0]["anchor_idx"]) == 2
    assert int(consistency.iloc[0]["n_predictors"]) == 2
    assert set(factor["factor_group"]) == {"VI", "time"}


def test_result_3_2bc_common_builders() -> None:
    delta_df = _mock_delta_df()
    metrics_df = _mock_metrics_df()

    shared = build_shared_anchor_report_table(delta_df)
    g_df = build_g_baseline_table(delta_df)
    same_vi_by_predictor = build_same_vi_by_predictor(metrics_df, delta_df)
    same_vi_summary = build_same_vi_summary(same_vi_by_predictor)
    top_h = build_top_h_vi(same_vi_summary)
    top_gh = build_top_gh_vi(same_vi_summary)
    scenario_trait = build_scenario_trait_overview(same_vi_summary, top_h, top_gh)
    merged, severe_drop, stable = build_cross_scenario_tables(same_vi_summary)
    useful_h_but_not_gh, h_weak_but_g_helps = build_role_shift_tables(merged)

    assert int(shared.iloc[0]["best_anchor"]) == 2
    assert float(g_df.iloc[0]["g_mean_pearson"]) == 0.40
    assert set(same_vi_summary["vi_label"]) == {"NDRE", "GNDVI"}
    assert top_h.iloc[0]["vi_label"] == "NDRE"
    assert top_gh.iloc[0]["vi_label"] == "NDRE"
    assert "best_h_vi" in scenario_trait.columns
    assert not severe_drop.empty
    assert not stable.empty
    assert isinstance(useful_h_but_not_gh, pd.DataFrame)
    assert isinstance(h_weak_but_g_helps, pd.DataFrame)


def test_result_3_2c_analysis_falls_back_to_sibling_3_2b(tmp_path) -> None:
    root = tmp_path / "outputs" / "experiments" / "two_traits_full_pipeline_gpu2"
    input_3_2c = root / "3_2c"
    input_3_2b = root / "3_2b"
    input_3_2c.mkdir(parents=True)
    input_3_2b.mkdir(parents=True)
    _mock_metrics_df().to_csv(input_3_2b / "metrics_summary.csv", index=False)

    resolved_dir, metrics_df, note = _read_metrics_with_compatibility(input_3_2c)

    assert resolved_dir == input_3_2b
    assert "GH_SINGLE_VI" in set(metrics_df["modality_variant"])
    assert "sibling 3_2b" in note
