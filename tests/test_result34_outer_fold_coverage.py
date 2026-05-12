import pandas as pd
import pytest

from models.result34.runner import _validate_result34_outer_fold_coverage


def test_result34_outer_fold_coverage_accepts_complete_task() -> None:
    split_usage_df = pd.DataFrame(
        [
            {
                "scenario": "reference",
                "timeline": "gdd_rel_heading",
                "target": "ActualYD",
                "status": "ok",
                "n_groups_used": 3,
            }
        ]
    )
    metrics_fold_df = pd.DataFrame(
        [
            {
                "scenario": "reference",
                "timeline": "gdd_rel_heading",
                "target": "ActualYD",
                "input_variant": "H_ANCHOR_AUTO",
                "predictor": "ridge",
                "target_year": 2022,
                "outer_fold": 0,
            },
            {
                "scenario": "reference",
                "timeline": "gdd_rel_heading",
                "target": "ActualYD",
                "input_variant": "H_ANCHOR_AUTO",
                "predictor": "ridge",
                "target_year": 2022,
                "outer_fold": 1,
            },
            {
                "scenario": "reference",
                "timeline": "gdd_rel_heading",
                "target": "ActualYD",
                "input_variant": "H_ANCHOR_AUTO",
                "predictor": "ridge",
                "target_year": 2022,
                "outer_fold": 2,
            },
        ]
    )

    _validate_result34_outer_fold_coverage(
        metrics_fold_df=metrics_fold_df,
        split_usage_df=split_usage_df,
    )


def test_result34_outer_fold_coverage_rejects_missing_folds() -> None:
    split_usage_df = pd.DataFrame(
        [
            {
                "scenario": "loso_genotype",
                "timeline": "gdd_rel_heading",
                "target": "PHM",
                "status": "ok",
                "n_groups_used": 15,
            }
        ]
    )
    metrics_fold_df = pd.DataFrame(
        [
            {
                "scenario": "loso_genotype",
                "timeline": "gdd_rel_heading",
                "target": "PHM",
                "input_variant": "G+H_ANCHOR_AUTO",
                "predictor": "ridge",
                "target_year": 2025,
                "outer_fold": 4,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="outer-fold coverage"):
        _validate_result34_outer_fold_coverage(
            metrics_fold_df=metrics_fold_df,
            split_usage_df=split_usage_df,
        )
