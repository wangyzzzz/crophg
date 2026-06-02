from pathlib import Path

import pandas as pd

from crophg.public.cropvig_1 import (
    build_auto_window_selection_counts,
    build_overall_summary,
    build_scenario_summary,
    build_trait_level_compare,
)


def _mock_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"scenario": "reference", "target": "ActualYD", "input_variant": "G", "mean_pearson": 0.50},
            {"scenario": "reference", "target": "ActualYD", "input_variant": "H_FULL", "mean_pearson": 0.40},
            {"scenario": "reference", "target": "ActualYD", "input_variant": "G+FULLH", "mean_pearson": 0.60},
            {"scenario": "reference", "target": "ActualYD", "input_variant": "H_ANCHOR_AUTO", "mean_pearson": 0.45},
            {"scenario": "reference", "target": "ActualYD", "input_variant": "G+H_ANCHOR_AUTO", "mean_pearson": 0.65},
        ]
    )


def _mock_fold() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"scenario": "reference", "input_variant": "H_ANCHOR_AUTO", "selected_window_radius": 2},
            {"scenario": "reference", "input_variant": "G+H_ANCHOR_AUTO", "selected_window_radius": 0},
        ]
    )


def test_trait_level_compare_contains_all_variants() -> None:
    table = build_trait_level_compare(_mock_summary())
    assert list(table.columns) == ["scenario", "target", "G", "H_FULL", "G+FULLH", "H_ANCHOR_AUTO", "G+H_ANCHOR_AUTO"]


def test_scenario_summary_nonempty() -> None:
    table = build_scenario_summary(_mock_summary())
    assert not table.empty
    assert set(table["input_variant"]) == {"G", "H_FULL", "G+FULLH", "H_ANCHOR_AUTO", "G+H_ANCHOR_AUTO"}


def test_overall_summary_best_variant() -> None:
    table = build_overall_summary(_mock_summary())
    best = table.sort_values("overall_mean_pearson", ascending=False).iloc[0]
    assert best["input_variant"] == "G+H_ANCHOR_AUTO"


def test_auto_window_counts() -> None:
    table = build_auto_window_selection_counts(_mock_fold())
    assert len(table) == 2
    assert sorted(table["selected_window_radius"].tolist()) == [0, 2]
