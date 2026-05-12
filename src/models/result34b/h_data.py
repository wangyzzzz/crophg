from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from models.anchorwise.data_loader import MultiTargetDataBundle
from models.anchorwise.modeling import sanitize_feature_matrix
from models.result34.feature_engineering import AnchorLocalConfig, build_h_anchor_local_features, build_h_anchor_vi_features


@dataclass(frozen=True)
class HRepresentationBundle:
    x: pd.DataFrame
    feature_spec_df: pd.DataFrame
    group_axis_df: pd.DataFrame
    group_id_to_cols: dict[str, list[str]]
    input_variant: str
    n_groups: int
    n_vi: int


def _canonical_h_variant(input_variant: str) -> str:
    token = str(input_variant).strip().upper()
    if token in {"H_ANCHOR_LOCAL", "H_ANCHOR_AUTO"}:
        return "H_ANCHOR_AUTO"
    if token in {"G+H_ANCHOR_LOCAL", "G+H_ANCHOR_AUTO"}:
        return "G+H_ANCHOR_AUTO"
    raise ValueError(f"未知 H variant: {input_variant}")


def build_h_group_representation(
    bundle: MultiTargetDataBundle,
    *,
    n_anchor_bins: int = 20,
    enable_phase_internal_tail_fill: bool = True,
    input_variant: str = "H_ANCHOR_AUTO",
) -> HRepresentationBundle:
    cfg = AnchorLocalConfig(
        n_anchor_bins=int(n_anchor_bins),
        enable_phase_internal_tail_fill=bool(enable_phase_internal_tail_fill),
    )
    x_h, info = build_h_anchor_local_features(bundle, config=cfg)

    x_h = sanitize_feature_matrix(x_h.copy())
    spec_df = info.get("feature_spec_df", pd.DataFrame()).copy()
    if spec_df.empty:
        raise RuntimeError("H 表示的 feature_spec_df 为空。")

    spec_df["group_id"] = (
        spec_df["anchor_token"].astype(str).fillna("na_anchor")
        + "||"
        + spec_df["vi_name"].astype(str).fillna("na_vi")
    )
    if "relative_offset" not in spec_df.columns:
        spec_df["relative_offset"] = 0
    spec_df["relative_offset"] = spec_df["relative_offset"].fillna(0).astype(int)
    spec_df["group_pos"] = spec_df.groupby("group_id")["relative_offset"].rank(method="dense").astype(int) - 1

    group_axis_df = (
        spec_df.loc[:, ["group_id", "anchor_idx", "anchor_token", "anchor_tb", "anchor_phase", "vi_name", "family"]]
        .drop_duplicates()
        .sort_values(["anchor_idx", "vi_name", "group_id"])
        .reset_index(drop=True)
    )
    group_axis_df["anchor_idx"] = group_axis_df["anchor_idx"].astype(int)
    if "anchor_tb" in group_axis_df.columns:
        group_axis_df["anchor_tb"] = group_axis_df["anchor_tb"].astype(int)

    grouped = (
        spec_df.loc[:, ["group_id", "feature", "relative_offset"]]
        .drop_duplicates()
        .sort_values(["group_id", "relative_offset", "feature"])
        .groupby("group_id")["feature"]
        .apply(lambda s: [str(x) for x in s.tolist()])
    )
    group_id_to_cols = {str(group_id): cols for group_id, cols in grouped.items()}
    n_vi = int(spec_df["vi_name"].astype(str).nunique())
    canonical_variant = _canonical_h_variant(input_variant)
    return HRepresentationBundle(
        x=x_h.loc[bundle.meta.index].copy(),
        feature_spec_df=spec_df,
        group_axis_df=group_axis_df,
        group_id_to_cols=group_id_to_cols,
        input_variant=canonical_variant,
        n_groups=int(len(group_id_to_cols)),
        n_vi=n_vi,
    )


def build_group_design_matrix(
    x_h: pd.DataFrame,
    feature_spec_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str], dict[str, list[int]], np.ndarray]:
    spec_df = feature_spec_df.copy()
    if spec_df.empty:
        raise RuntimeError("feature_spec_df 为空，无法构造 group design matrix。")
    x_use = x_h.loc[:, [col for col in spec_df["feature"].astype(str).tolist() if col in x_h.columns]].copy()
    x_use = sanitize_feature_matrix(x_use)
    col_names = [str(c) for c in x_use.columns.tolist()]
    col_index = {col: idx for idx, col in enumerate(col_names)}
    group_to_feature_indices: dict[str, list[int]] = {}
    smooth_pairs: list[tuple[int, int]] = []
    for group_id, sub in spec_df.groupby("group_id", sort=False):
        sub_ord = sub.drop_duplicates(subset=["feature"]).sort_values(["relative_offset", "feature"])
        idxs = [col_index[str(f)] for f in sub_ord["feature"].astype(str).tolist() if str(f) in col_index]
        if not idxs:
            continue
        group_to_feature_indices[str(group_id)] = idxs
        for left, right in zip(idxs[:-1], idxs[1:]):
            smooth_pairs.append((int(left), int(right)))
    x_arr = x_use.to_numpy(dtype=float)
    smooth_pairs_arr = np.asarray(smooth_pairs, dtype=int) if smooth_pairs else np.zeros((0, 2), dtype=int)
    return x_arr, col_names, group_to_feature_indices, smooth_pairs_arr


def build_temporal_tensor(
    x_h: pd.DataFrame,
    feature_spec_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    spec_df = feature_spec_df.copy()
    if spec_df.empty:
        raise RuntimeError("feature_spec_df 为空，无法构造 temporal tensor。")
    anchor_tokens = sorted(spec_df["anchor_token"].astype(str).unique().tolist(), key=lambda x: x)
    vi_names = sorted(spec_df["vi_name"].astype(str).unique().tolist())
    offsets = sorted(spec_df["relative_offset"].astype(int).unique().tolist())
    anchor_to_idx = {token: i for i, token in enumerate(anchor_tokens)}
    vi_to_idx = {name: i for i, name in enumerate(vi_names)}
    offset_to_idx = {offset: i for i, offset in enumerate(offsets)}

    x_use = sanitize_feature_matrix(x_h.copy())
    n = int(x_use.shape[0])
    tensor = np.zeros((n, len(anchor_tokens), len(vi_names), len(offsets)), dtype=float)
    mask = np.zeros_like(tensor, dtype=float)

    for _, row in spec_df.iterrows():
        feature = str(row["feature"])
        if feature not in x_use.columns:
            continue
        a_idx = anchor_to_idx[str(row["anchor_token"])]
        v_idx = vi_to_idx[str(row["vi_name"])]
        o_idx = offset_to_idx[int(row["relative_offset"])]
        values = pd.to_numeric(x_use[feature], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        tensor[:, a_idx, v_idx, o_idx] = np.where(finite_mask, values, 0.0)
        mask[:, a_idx, v_idx, o_idx] = finite_mask.astype(float)
    return tensor, mask, anchor_tokens, vi_names, np.asarray(offsets, dtype=int)
