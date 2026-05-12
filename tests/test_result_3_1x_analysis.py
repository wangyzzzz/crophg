from __future__ import annotations

import pandas as pd

from crophg.internal.result_3_1a_analysis import (
    build_best_accuracy_table as build_31a_best,
    build_scenario_mean_table as build_31a_scenario_mean,
    build_trait_gap_table,
)
from crophg.internal.result_3_1b_analysis import (
    build_best_accuracy_table as build_31b_best,
    build_scenario_mean_table as build_31b_scenario_mean,
    build_trait_table,
)


def _mock_best_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"scenario": "reference", "target": "ActualYD", "best_r2": 0.40, "best_pearson": 0.70, "avg_r2": 0.35, "avg_pearson": 0.65},
            {"scenario": "within_season", "target": "ActualYD", "best_r2": 0.39, "best_pearson": 0.69, "avg_r2": 0.34, "avg_pearson": 0.64},
            {"scenario": "loso", "target": "ActualYD", "best_r2": 0.25, "best_pearson": 0.55, "avg_r2": 0.20, "avg_pearson": 0.50},
            {"scenario": "loso_genotype", "target": "ActualYD", "best_r2": 0.20, "best_pearson": 0.50, "avg_r2": 0.18, "avg_pearson": 0.46},
        ]
    )


def test_result_3_1a_analysis_builders() -> None:
    best_df = build_31a_best(_mock_best_df())
    scenario_mean = build_31a_scenario_mean(best_df)
    trait_gap = build_trait_gap_table(best_df)

    assert not best_df.empty
    assert scenario_mean.shape[0] == 4
    row = trait_gap.iloc[0]
    assert abs(float(row["gap_reference_to_loso_genotype"]) - 0.20) < 1e-9


def test_result_3_1b_analysis_builders() -> None:
    best_df = build_31b_best(_mock_best_df())
    scenario_mean = build_31b_scenario_mean(best_df)
    trait_df = build_trait_table(best_df)

    assert not best_df.empty
    assert scenario_mean.shape[0] == 4
    assert "mean_across_scenarios" in trait_df.columns
