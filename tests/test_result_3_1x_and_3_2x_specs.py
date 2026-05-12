from crophg.internal.result_3_1a import SECTION_CODE as C31A, SECTION_SCOPE as S31A
from crophg.internal.result_3_1b import SECTION_CODE as C31B, SECTION_SCOPE as S31B
from crophg.internal.result_3_2a import SECTION_CODE as C32A, SECTION_SCOPE as S32A
from crophg.internal.result_3_2b import SECTION_CODE as C32B, SECTION_SCOPE as S32B
from crophg.internal.result_3_2c import SECTION_CODE as C32C, SECTION_SCOPE as S32C
from crophg.internal.shared_anchor import build_shared_anchor_table

import pandas as pd


def test_internal_section_specs_are_uppercase() -> None:
    assert C31A == "3.1A"
    assert C31B == "3.1B"
    assert C32A == "3.2A"
    assert C32B == "3.2B"
    assert C32C == "3.2C"
    assert {S31A, S31B, S32A, S32B, S32C} == {"internal"}


def test_shared_anchor_builder_prefers_highest_mean_delta() -> None:
    df = pd.DataFrame(
        [
            {"scenario": s, "predictor": p, "timeline": "gdd_rel_heading", "target": "ActualYD", "modality_variant": "GH_SINGLE", "anchor_idx": 1, "anchor_tb": 150, "anchor_phase": "sow", "anchor_band": "early", "delta_pearson": 0.1, "mean_pearson": 0.5, "g_mean_pearson": 0.4}
            for s in ["reference", "within_season", "loso", "loso_genotype"]
            for p in ["ridge", "lasso", "elasticnet", "lightgbm", "random_forest"]
        ]
        + [
            {"scenario": s, "predictor": p, "timeline": "gdd_rel_heading", "target": "ActualYD", "modality_variant": "GH_SINGLE", "anchor_idx": 2, "anchor_tb": 70, "anchor_phase": "head", "anchor_band": "late", "delta_pearson": 0.2, "mean_pearson": 0.6, "g_mean_pearson": 0.4}
            for s in ["reference", "within_season", "loso", "loso_genotype"]
            for p in ["ridge", "lasso", "elasticnet", "lightgbm", "random_forest"]
        ]
    )
    expanded, best = build_shared_anchor_table(df)
    assert len(expanded) == 16
    assert int(best.iloc[0]["best_anchor"]) == 2
