import pandas as pd

from crophg.public.result_3_3b import (
    build_compression_summary,
    build_feature_summary,
    build_gain_summary,
    build_overall_summary,
    build_trait_overall_summary,
)


def _mock_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"scenario": "reference", "target": "ActualYD", "input_variant": "G", "mean_pearson": 0.50},
            {"scenario": "reference", "target": "ActualYD", "input_variant": "H_FULL", "mean_pearson": 0.40},
            {"scenario": "reference", "target": "ActualYD", "input_variant": "G+FULLH", "mean_pearson": 0.60},
            {"scenario": "reference", "target": "ActualYD", "input_variant": "H_ANCHOR_AUTO", "mean_pearson": 0.45},
            {"scenario": "reference", "target": "ActualYD", "input_variant": "G+H_ANCHOR_AUTO", "mean_pearson": 0.65},
            {"scenario": "loso", "target": "ActualYD", "input_variant": "G", "mean_pearson": 0.45},
            {"scenario": "loso", "target": "ActualYD", "input_variant": "H_FULL", "mean_pearson": 0.30},
            {"scenario": "loso", "target": "ActualYD", "input_variant": "G+FULLH", "mean_pearson": 0.50},
            {"scenario": "loso", "target": "ActualYD", "input_variant": "H_ANCHOR_AUTO", "mean_pearson": 0.42},
            {"scenario": "loso", "target": "ActualYD", "input_variant": "G+H_ANCHOR_AUTO", "mean_pearson": 0.61},
        ]
    )


def _mock_fold() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"input_variant": "H_FULL", "n_features_total": 6900, "n_features_h_full": 6900, "n_features_g": 0},
            {"input_variant": "G+FULLH", "n_features_total": 7312, "n_features_h_full": 6900, "n_features_g": 412},
            {
                "input_variant": "H_ANCHOR_AUTO",
                "n_features_after_pruning": 60,
                "n_h_features_after_pruning": 60,
                "n_features_g": 0,
                "selected_window_radius": 2,
            },
            {
                "input_variant": "G+H_ANCHOR_AUTO",
                "n_features_after_pruning": 450,
                "n_h_features_after_pruning": 38,
                "n_features_g": 412,
                "selected_window_radius": 0,
            },
        ]
    )


def test_overall_summary_best_variant() -> None:
    table = build_overall_summary(_mock_summary())
    best = table.sort_values("overall_mean_pearson", ascending=False).iloc[0]
    assert best["input_variant"] == "G+H_ANCHOR_AUTO"


def test_trait_overall_summary_contains_variants() -> None:
    table = build_trait_overall_summary(_mock_summary())
    assert list(table.columns) == ["target", "G", "H_FULL", "G+FULLH", "H_ANCHOR_AUTO", "G+H_ANCHOR_AUTO"]


def test_gain_summary_has_expected_deltas() -> None:
    table = build_gain_summary(_mock_summary())
    ref = table.loc[table["scenario"] == "Reference"].iloc[0]
    assert abs(ref["delta_g_vs_h_full"] - 0.10) < 1e-9
    assert abs(ref["delta_gh_full_vs_h_full"] - 0.20) < 1e-9
    assert abs(ref["delta_h_auto_vs_h_full"] - 0.05) < 1e-9
    assert abs(ref["delta_gh_auto_vs_h_full"] - 0.25) < 1e-9
    assert abs(ref["delta_gh_auto_vs_gh_full"] - 0.05) < 1e-9


def test_feature_and_compression_summary() -> None:
    feature = build_feature_summary(_mock_fold())
    compression = build_compression_summary(feature)
    h_row = compression.loc[compression["comparison"] == "H_FULL -> H_ANCHOR_AUTO"].iloc[0]
    gh_row = compression.loc[compression["comparison"] == "G+FULLH -> G+H_ANCHOR_AUTO"].iloc[0]
    assert abs(h_row["retained_ratio"] - (60 / 6900)) < 1e-9
    assert abs(gh_row["reduced_features"] - 450) < 1e-9
