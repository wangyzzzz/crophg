from __future__ import annotations

import unittest

import pandas as pd

from models.anchorwise.data_loader import MultiTargetDataBundle
from models.result34 import runner as result34_runner
from models.result34.feature_engineering import (
    AnchorLocalConfig,
    attach_anchor_bin_prefix_to_full_h_spec,
    build_h_anchor_local_features,
    build_h_full_features,
)
from models.result34b.h_data import build_group_design_matrix, build_h_group_representation, build_temporal_tensor


def _make_bundle() -> MultiTargetDataBundle:
    index = pd.Index(["p1", "p2"], name="plot_id")
    meta = pd.DataFrame({"year": [2024, 2025], "genotype_id": ["g1", "g2"]}, index=index)
    y_df = pd.DataFrame({"TraitA": [1.0, 2.0]}, index=index)
    x_g = pd.DataFrame({"grm_pc_1": [0.1, 0.2], "grm_pc_2": [0.3, 0.4]}, index=index)
    x_h = pd.DataFrame(
        {
            "vi_ndre__tb_m00010": [1.0, 2.0],
            "vi_ndre__tb_p00000": [1.5, 2.5],
            "vi_ndvi__tb_p00010": [3.0, 4.0],
            "vi_ndvi__tb_p00020": [3.5, 4.5],
        },
        index=index,
    )
    return MultiTargetDataBundle(
        meta=meta,
        y_df=y_df,
        x_climate=pd.DataFrame(index=index),
        x_hyperspectral=x_h,
        x_genotype=x_g,
        climate_tb_map={},
        hyperspectral_tb_map={
            (None, -10): ["vi_ndre__tb_m00010"],
            (None, 0): ["vi_ndre__tb_p00000"],
            (None, 10): ["vi_ndvi__tb_p00010"],
            (None, 20): ["vi_ndvi__tb_p00020"],
        },
        common_tbs=[(None, -10), (None, 0), (None, 10), (None, 20)],
    )


class TestResult34AnchorBinFeatures(unittest.TestCase):
    def test_anchor_grouping_uses_midpoint_points_at_radius_zero(self) -> None:
        bundle = _make_bundle()
        x_h, info = build_h_anchor_local_features(bundle, config=AnchorLocalConfig(n_anchor_bins=2, radius=0))

        self.assertEqual(int(info["n_anchor_bins_effective"]), 2)
        self.assertEqual(int(info["n_group_candidates"]), 2)
        self.assertEqual(list(x_h.columns), [
            "anchor00__tb_p00000||vi_ndre||rel_+0||vi_ndre__tb_p00000",
            "anchor01__tb_p00020||vi_ndvi||rel_+0||vi_ndvi__tb_p00020",
        ])
        spec_df = info["feature_spec_df"]
        self.assertEqual(int(spec_df["anchor_vi_group"].nunique()), 2)
        self.assertTrue((spec_df["relative_side"] == "center").all())
        self.assertEqual(spec_df["feature_type"].unique().tolist(), ["anchor_point_raw_value"])

    def test_anchor_grouping_expands_neighbor_midpoints_by_radius(self) -> None:
        bundle = _make_bundle()
        x_h, info = build_h_anchor_local_features(bundle, config=AnchorLocalConfig(n_anchor_bins=2, radius=1))

        self.assertEqual(int(info["n_anchor_bins_effective"]), 2)
        self.assertEqual(int(info["n_group_candidates"]), 4)
        self.assertEqual(int(info["anchor_window_radius"]), 1)
        spec_df = info["feature_spec_df"]
        self.assertEqual(spec_df["feature_type"].unique().tolist(), ["anchor_point_local_value"])
        self.assertEqual(
            sorted(spec_df.loc[spec_df["anchor_token"] == "anchor00__tb_p00000", "relative_offset"].unique().tolist()),
            [0, 1],
        )
        self.assertEqual(
            sorted(spec_df.loc[spec_df["anchor_token"] == "anchor01__tb_p00020", "relative_offset"].unique().tolist()),
            [-1, 0],
        )
        self.assertEqual(x_h.shape[1], 4)

    def test_growth_prefix_drops_future_source_anchors_inside_radius(self) -> None:
        bundle = _make_bundle()
        x_h, info = build_h_anchor_local_features(bundle, config=AnchorLocalConfig(n_anchor_bins=2, radius=1))

        x_prefix, info_prefix, spec_prefix = result34_runner._limit_variant_to_anchor_prefix(
            x=x_h,
            info=info,
            feature_spec_df=info["feature_spec_df"],
            input_variant="H_ANCHOR_AUTO",
            anchor_order=1,
        )

        self.assertEqual(int(info_prefix["anchor_order"]), 1)
        self.assertEqual(int(info_prefix["n_features_total"]), 1)
        self.assertEqual(list(x_prefix.columns), [
            "anchor00__tb_p00000||vi_ndre||rel_+0||vi_ndre__tb_p00000",
        ])
        self.assertTrue((spec_prefix["anchor_idx"] <= 0).all())
        self.assertTrue((spec_prefix["source_anchor_idx"] <= 0).all())

    def test_full_h_prefix_keeps_all_raw_features_within_elapsed_bins(self) -> None:
        bundle = _make_bundle()
        x_full, info = build_h_full_features(bundle)
        spec_with_prefix = attach_anchor_bin_prefix_to_full_h_spec(
            bundle,
            info["feature_spec_df"],
            n_anchor_bins=2,
        )

        self.assertEqual(int(spec_with_prefix["anchor_idx"].isna().sum()), 0)
        x_prefix, info_prefix, spec_prefix = result34_runner._limit_variant_to_anchor_prefix(
            x=x_full,
            info=info,
            feature_spec_df=spec_with_prefix,
            input_variant="H_FULL",
            anchor_order=1,
        )

        self.assertEqual(int(info_prefix["anchor_order"]), 1)
        self.assertEqual(list(x_prefix.columns), [
            "vi_ndre__tb_m00010",
            "vi_ndre__tb_p00000",
        ])
        self.assertEqual(int(info_prefix["n_features_total"]), 2)
        self.assertEqual(int(info_prefix["n_features_h_full"]), 2)
        self.assertTrue((spec_prefix["anchor_idx"] <= 0).all())

    def test_group_design_matrix_keeps_bin_order(self) -> None:
        bundle = _make_bundle()
        rep = build_h_group_representation(bundle, n_anchor_bins=2, input_variant="H_ANCHOR_AUTO")

        self.assertEqual(rep.input_variant, "H_ANCHOR_AUTO")
        self.assertEqual(rep.n_groups, 2)
        self.assertEqual(rep.n_vi, 2)
        self.assertEqual(rep.group_id_to_cols["anchor00__tb_p00000||vi_ndre"], [
            "anchor00__tb_p00000||vi_ndre||rel_+0||vi_ndre__tb_p00000",
        ])

        x_arr, col_names, group_map, smooth_pairs = build_group_design_matrix(rep.x, rep.feature_spec_df)
        self.assertEqual(col_names, [
            "anchor00__tb_p00000||vi_ndre||rel_+0||vi_ndre__tb_p00000",
            "anchor01__tb_p00020||vi_ndvi||rel_+0||vi_ndvi__tb_p00020",
        ])
        self.assertEqual(len(group_map), 2)
        self.assertEqual(int(smooth_pairs.shape[0]), 0)
        self.assertEqual(group_map["anchor00__tb_p00000||vi_ndre"], [0])

    def test_temporal_tensor_uses_anchor_bin_by_vi_grid(self) -> None:
        bundle = _make_bundle()
        rep = build_h_group_representation(bundle, n_anchor_bins=2, input_variant="H_ANCHOR_AUTO")

        tensor, mask, anchor_tokens, vi_names, offsets = build_temporal_tensor(rep.x, rep.feature_spec_df)
        self.assertEqual(tensor.shape, (2, 2, 2, 1))
        self.assertEqual(mask.shape, (2, 2, 2, 1))
        self.assertEqual(anchor_tokens, ["anchor00__tb_p00000", "anchor01__tb_p00020"])
        self.assertEqual(vi_names, ["vi_ndre", "vi_ndvi"])
        self.assertEqual(offsets.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
