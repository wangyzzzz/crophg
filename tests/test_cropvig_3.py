from pathlib import Path
import tempfile

import pandas as pd

from crophg.public.cropvig_3 import (
    build_anchor_delta_summary,
    build_early_advantage_summary,
    build_overall_summary,
    build_prefix_curve_summary,
    build_scenario_summary,
    build_saturation_summary,
    build_trait_level_compare,
)


def _mock_summary() -> pd.DataFrame:
    rows = []
    for scenario in ["reference", "within_season"]:
        for target in ["ActualYD", "CM"]:
            for variant in ["G", "H_FULL", "G+FULLH", "H_ANCHOR_AUTO", "G+H_ANCHOR_AUTO"]:
                for anchor_order in [1, 2, 3]:
                    rows.append(
                        {
                            "scenario": scenario,
                            "target": target,
                            "input_variant": variant,
                            "anchor_order": anchor_order,
                            "anchor_tb": (anchor_order - 1) * 150,
                            "anchor_phase": "sow",
                            "mean_pearson": 0.1 * anchor_order,
                        }
                    )
    return pd.DataFrame(rows)


def test_3_4a_builders_work() -> None:
    df = _mock_summary()
    prefix = build_prefix_curve_summary(df)
    trait = build_trait_level_compare(df)
    scenario = build_scenario_summary(df)
    overall = build_overall_summary(df)
    delta = build_anchor_delta_summary(prefix)
    early = build_early_advantage_summary(prefix)
    saturation = build_saturation_summary(prefix)

    assert not prefix.empty
    assert list(trait.columns) == ["scenario", "target", "G", "H_FULL", "G+FULLH", "H_ANCHOR_AUTO", "G+H_ANCHOR_AUTO"]
    assert not scenario.empty
    assert not overall.empty
    assert "delta_h_auto_vs_h_full" in delta.columns
    assert "G+FULLH_reach90_minus_h_full" in early.columns
    assert "absolute_gain" in saturation.columns
