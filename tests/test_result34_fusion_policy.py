from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import models.result34.runner as result34_runner


def test_anchor_score_target_requires_raw_y_blend_one() -> None:
    y = pd.Series([1.0, 2.0, 3.0], index=["p1", "p2", "p3"])
    x = pd.DataFrame({"g1": [0.1, 0.2, 0.3]}, index=y.index)

    score_target, mode = result34_runner._resolve_anchor_local_score_target(
        x_train=x,
        y_train=y,
        g_cols=["g1"],
        input_variant="G+H_ANCHOR_VI",
        pruning_cfg={"g_aware_score_blend": 1.0},
        g_pred_train=np.array([0.9, 1.1, 1.4]),
    )

    assert mode == "y_direct_g_aware_blend_1.00"
    pd.testing.assert_series_equal(score_target, y)

    with pytest.raises(ValueError, match="g_aware_score_blend=1.0"):
        result34_runner._resolve_anchor_local_score_target(
            x_train=x,
            y_train=y,
            g_cols=["g1"],
            input_variant="G+H_ANCHOR_VI",
            pruning_cfg={"g_aware_score_blend": 0.0},
        )


def test_pruning_stats_use_raw_y_score_target_label() -> None:
    text = Path(result34_runner.__file__).read_text(encoding="utf-8")
    assert '"score_target": "y_direct_g_aware_blend_1.00"' in text
    assert "inner_stability_gaware" not in text
    assert "single_inner_gaware" not in text


def test_safe_fusion_selects_raw_y_least_squares_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.Index([f"p{i}" for i in range(8)], name="plot_id")
    y_train = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx[:4])
    y_val = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx[4:])
    y_test = pd.Series([1.5, 3.5], index=pd.Index(["t1", "t2"], name="plot_id"))

    inner = {
        "inner_fold": 0,
        "x_train": pd.DataFrame(
            {
                "h1": [1.0, 2.0, 3.0, 4.0],
                "g1": [0.0, 0.0, 0.0, 0.0],
            },
            index=idx[:4],
        ),
        "x_val": pd.DataFrame(
            {
                "h1": [1.0, 2.0, 3.0, 4.0],
                "g1": [1.0, 2.0, 3.0, 4.0],
            },
            index=idx[4:],
        ),
        "y_train": y_train,
        "y_val": y_val,
        "meta_train": pd.DataFrame({"genotype_id": ["g1"] * 4}, index=idx[:4]),
        "meta_val": pd.DataFrame({"genotype_id": ["g1"] * 4}, index=idx[4:]),
    }
    x_test = pd.DataFrame({"h1": [1.5, 3.5], "g1": [1.5, 3.5]}, index=y_test.index)
    meta_test = pd.DataFrame({"genotype_id": ["g1", "g1"]}, index=y_test.index)
    feature_spec = pd.DataFrame({"feature": ["h1"], "anchor_token": ["bin00"], "vi_name": ["vi_ndvi"]})

    def fake_h_branch(*, x_apply: pd.DataFrame, **_: object) -> np.ndarray:
        return x_apply["h1"].to_numpy(dtype=float)

    def fake_g_branch(**kwargs: object) -> dict[str, np.ndarray]:
        return {
            "pred_train": kwargs["x_train_g"]["g1"].to_numpy(dtype=float),
            "pred_val": kwargs["x_val_g"]["g1"].to_numpy(dtype=float),
            "pred_test": kwargs["x_test_g"]["g1"].to_numpy(dtype=float),
        }

    monkeypatch.setattr(result34_runner, "_fit_predict_single_branch", fake_h_branch)
    monkeypatch.setattr(result34_runner, "_get_or_fit_cached_g_branch", fake_g_branch)

    inner["x_train"]["g1"] = [1.0, 2.0, 3.0, 4.0]
    inner["y_val"] = pd.Series([2.0, 4.0, 6.0, 8.0], index=idx[4:])
    y_test = pd.Series([3.0, 7.0], index=pd.Index(["t1", "t2"], name="plot_id"))

    result = result34_runner._fit_predict_safe_fusion(
        inner_payload=[inner],
        x_test=x_test,
        y_test=y_test,
        meta_test=meta_test,
        predictor="ridge",
        params={"alpha": 1.0},
        preprocess_cfg={"imputer": {"strategy": "median", "add_indicator": False}},
        model_cfg={},
        feature_spec_df=feature_spec,
        seed=42,
        fusion_cfg={"selection_metric": "rmse"},
        repo_root=Path.cwd(),
        use_gblup_for_g=False,
    )

    assert result["candidate_selected_list"] == ["ols_h_g"]
    assert result["weight_g_mean"] > 0.5
    assert result["test_metrics"]["rmse"] < 1.0e-8
