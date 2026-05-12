from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from models.anchorwise.data_loader import MultiTargetDataBundle, TimeKey, time_key_sort_key, time_key_to_token


VI_FAMILY_MAP: dict[str, str] = {
    "NDVI": "nir_red_vigor",
    "EVI2": "nir_red_vigor",
    "RDVI": "nir_red_vigor",
    "SAVI": "soil_adjusted_vigor",
    "OSAVI": "soil_adjusted_vigor",
    "MSAVI": "soil_adjusted_vigor",
    "GNDVI": "rededge_chlorophyll",
    "NDRE": "rededge_chlorophyll",
    "MSR": "rededge_chlorophyll",
    "GRVI": "visible_structure",
    "MGRVI": "visible_structure",
    "VARI": "visible_structure",
}


@dataclass(frozen=True)
class PhaseStateConfig:
    sow_segments: int = 4
    head_segments: int = 3
    enable_phase_internal_tail_fill: bool = True


@dataclass(frozen=True)
class AnchorLocalConfig:
    n_anchor_bins: int = 20
    enable_phase_internal_tail_fill: bool = True
    radius: int = 0


def _h_feature_base_name(col_name: str) -> str:
    if "__ph_" in col_name:
        return col_name.split("__ph_")[0]
    if "__tb_" in col_name:
        return col_name.split("__tb_")[0]
    return col_name


def _phase_ordered_vi_entries(bundle: MultiTargetDataBundle) -> dict[str, dict[str, list[tuple[TimeKey, str]]]]:
    out: dict[str, dict[str, list[tuple[tuple[int, int], TimeKey, str]]]] = {}
    for time_key, cols in bundle.hyperspectral_tb_map.items():
        phase = str(time_key[0] or "sow")
        sort_key = time_key_sort_key(time_key)
        for col in cols:
            vi_name = _h_feature_base_name(str(col))
            out.setdefault(vi_name, {}).setdefault(phase, []).append((sort_key, time_key, str(col)))

    finalized: dict[str, dict[str, list[tuple[TimeKey, str]]]] = {}
    for vi_name, phase_map in out.items():
        finalized[vi_name] = {}
        for phase, triples in phase_map.items():
            ordered = [(time_key, col) for _, time_key, col in sorted(triples, key=lambda x: x[0])]
            finalized[vi_name][phase] = ordered
    return dict(sorted(finalized.items(), key=lambda x: x[0]))


def _timeline_ordered_vi_entries(bundle: MultiTargetDataBundle) -> dict[str, list[tuple[TimeKey, str]]]:
    out: dict[str, list[tuple[TimeKey, str]]] = {}
    for vi_name, phase_map in _phase_ordered_vi_entries(bundle).items():
        merged: list[tuple[TimeKey, str]] = []
        for entries in phase_map.values():
            merged.extend(entries)
        out[vi_name] = sorted(merged, key=lambda x: time_key_sort_key(x[0]))
    return out


def _tail_fill_within_phase(
    x_h: pd.DataFrame,
    vi_phase_cols: dict[str, dict[str, list[str]]],
) -> pd.DataFrame:
    filled = x_h.copy()
    for phase_map in vi_phase_cols.values():
        for cols in phase_map.values():
            if not cols:
                continue
            filled.loc[:, cols] = filled.loc[:, cols].ffill(axis=1)
    return filled


def _tail_fill_full_h_tail_only(bundle: MultiTargetDataBundle) -> pd.DataFrame:
    if bundle.x_hyperspectral.empty or not bundle.hyperspectral_tb_map:
        return bundle.x_hyperspectral.copy().loc[bundle.meta.index].copy()

    filled = bundle.x_hyperspectral.copy()
    phase_to_keys: dict[str | None, list[TimeKey]] = {}
    for key in sorted(bundle.hyperspectral_tb_map.keys(), key=lambda x: (str(x[0]), int(x[1]))):
        phase_to_keys.setdefault(key[0], []).append(key)

    for _, keys in phase_to_keys.items():
        ordered_keys = sorted(keys, key=lambda x: int(x[1]))
        key_cols = {
            key: [c for c in bundle.hyperspectral_tb_map.get(key, []) if c in filled.columns]
            for key in ordered_keys
        }
        key_cols = {k: v for k, v in key_cols.items() if v}
        if not key_cols:
            continue

        phase_cols = [c for key in ordered_keys for c in key_cols.get(key, [])]
        phase_df = filled.loc[:, phase_cols].copy()

        block_missing = {
            key: phase_df.loc[:, cols].isna().all(axis=1)
            for key, cols in key_cols.items()
        }
        ordered_base_names = sorted({_h_feature_base_name(col) for cols in key_cols.values() for col in cols})
        key_to_base_col = {
            key: {_h_feature_base_name(col): col for col in cols}
            for key, cols in key_cols.items()
        }

        for base_name in ordered_base_names:
            series_by_key: list[pd.Series] = []
            block_missing_cols: list[pd.Series] = []
            raw_col_names: list[str] = []
            for key in ordered_keys:
                raw_col_name = key_to_base_col.get(key, {}).get(base_name)
                if raw_col_name is None:
                    continue
                series_by_key.append(phase_df.loc[:, raw_col_name].rename(time_key_to_token(key)))
                block_missing_cols.append(block_missing[key].rename(time_key_to_token(key)))
                raw_col_names.append(raw_col_name)

            if not series_by_key:
                continue

            vi_df = pd.concat(series_by_key, axis=1)
            missing_df = pd.concat(block_missing_cols, axis=1).reindex(columns=vi_df.columns)
            carried_df = vi_df.ffill(axis=1)
            out_df = vi_df.where(~missing_df, carried_df)

            for col_token, raw_col_name in zip(vi_df.columns.tolist(), raw_col_names):
                phase_df.loc[:, raw_col_name] = out_df.loc[:, col_token]
        filled.loc[:, phase_cols] = phase_df

    return filled.loc[bundle.meta.index].copy()


def _family_for_vi(vi_name: str) -> str:
    return VI_FAMILY_MAP.get(str(vi_name).upper(), "other")


def _normalize_phase_name(phase: str | None) -> str:
    return str(phase or "sow")


def _build_anchor_bin_definitions(bundle: MultiTargetDataBundle, n_anchor_bins: int) -> list[dict[str, Any]]:
    common_tbs = list(dict.fromkeys(sorted(bundle.common_tbs, key=time_key_sort_key)))
    if not common_tbs:
        raise ValueError("common_tbs 为空，无法切分 anchor bins。")

    n_effective = min(max(1, int(n_anchor_bins)), len(common_tbs))
    index_bins = np.array_split(np.arange(len(common_tbs)), n_effective)
    out: list[dict[str, Any]] = []
    for bin_idx, idxs in enumerate(index_bins):
        if len(idxs) == 0:
            continue
        time_keys = [common_tbs[int(i)] for i in idxs.tolist()]
        start_key = time_keys[0]
        end_key = time_keys[-1]
        center_key = time_keys[len(time_keys) // 2]
        out.append(
            {
                "anchor_idx": int(bin_idx),
                "anchor_phase": str(center_key[0] or end_key[0] or start_key[0] or "sow"),
                "anchor_tb": int(center_key[1]),
                "anchor_start_time_token": time_key_to_token(start_key),
                "anchor_end_time_token": time_key_to_token(end_key),
                "anchor_time_token": time_key_to_token(center_key),
                "anchor_token": f"anchor{bin_idx:02d}__{time_key_to_token(center_key)}",
                "center_time_key": center_key,
                "time_keys": time_keys,
                "bin_size": int(len(time_keys)),
            }
        )
    return out


def build_h_full_features(
    bundle: MultiTargetDataBundle,
    config: PhaseStateConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = config or PhaseStateConfig()
    vi_timeline_entries = _timeline_ordered_vi_entries(bundle)
    if config.enable_phase_internal_tail_fill:
        x_h_full = _tail_fill_full_h_tail_only(bundle)
    else:
        x_h_full = bundle.x_hyperspectral.copy().loc[bundle.meta.index].copy()

    feature_spec_df = pd.DataFrame(
        [
            {
                "feature": col,
                "vi_name": vi_name,
                "family": _family_for_vi(vi_name),
                "raw_time_token": time_key_to_token(time_key),
                "phase": str(time_key[0] or "sow"),
                "feature_type": "full_h_raw_value",
            }
            for vi_name, entries in vi_timeline_entries.items()
            for time_key, col in entries
        ]
    )
    info = {
        "representation": "H_full",
        "n_samples": int(x_h_full.shape[0]),
        "n_features_total": int(x_h_full.shape[1]),
        "n_vi": int(len(vi_timeline_entries)),
        "vi_names": list(vi_timeline_entries.keys()),
        "families": {vi: _family_for_vi(vi) for vi in vi_timeline_entries.keys()},
        "tail_fill_mode": "tail_only_within_phase" if config.enable_phase_internal_tail_fill else "none",
        "feature_spec_df": feature_spec_df,
    }
    return x_h_full, info


def attach_anchor_bin_prefix_to_full_h_spec(
    bundle: MultiTargetDataBundle,
    full_h_spec_df: pd.DataFrame,
    *,
    n_anchor_bins: int,
) -> pd.DataFrame:
    if full_h_spec_df.empty or "feature" not in full_h_spec_df.columns:
        return full_h_spec_df.copy()

    anchor_definitions = _build_anchor_bin_definitions(bundle, n_anchor_bins=max(1, int(n_anchor_bins)))
    meta_rows: list[dict[str, Any]] = []
    for anchor_def in anchor_definitions:
        anchor_idx = int(anchor_def["anchor_idx"])
        anchor_phase = _normalize_phase_name(anchor_def["anchor_phase"])
        for time_key in anchor_def["time_keys"]:
            for col in bundle.hyperspectral_tb_map.get(time_key, []):
                meta_rows.append(
                    {
                        "feature": str(col),
                        "anchor_idx": anchor_idx,
                        "anchor_token": str(anchor_def["anchor_token"]),
                        "anchor_tb": int(anchor_def["anchor_tb"]),
                        "anchor_phase": anchor_phase,
                        "anchor_time_token": str(anchor_def["anchor_time_token"]),
                        "anchor_start_time_token": str(anchor_def["anchor_start_time_token"]),
                        "anchor_end_time_token": str(anchor_def["anchor_end_time_token"]),
                        "bin_size": int(anchor_def["bin_size"]),
                        "source_anchor_idx": anchor_idx,
                    }
                )

    if not meta_rows:
        return full_h_spec_df.copy()

    anchor_meta = pd.DataFrame(meta_rows).drop_duplicates(subset=["feature"], keep="first")
    out = full_h_spec_df.copy()
    drop_cols = [col for col in anchor_meta.columns if col != "feature" and col in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out.merge(anchor_meta, on="feature", how="left")


def build_h_phase_state_features(
    bundle: MultiTargetDataBundle,
    config: PhaseStateConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return build_h_full_features(bundle, config=config)


def _build_h_anchor_bin_features(
    bundle: MultiTargetDataBundle,
    config: AnchorLocalConfig | None = None,
    *,
    representation: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = config or AnchorLocalConfig()
    radius = max(0, int(config.radius))
    vi_phase_entries = _phase_ordered_vi_entries(bundle)
    vi_timeline_entries = _timeline_ordered_vi_entries(bundle)
    anchor_definitions = _build_anchor_bin_definitions(bundle, n_anchor_bins=max(1, int(config.n_anchor_bins)))
    center_key_by_anchor = {int(anchor_def["anchor_idx"]): anchor_def["center_time_key"] for anchor_def in anchor_definitions}
    x_h = bundle.x_hyperspectral.copy().loc[bundle.meta.index].copy()
    if config.enable_phase_internal_tail_fill:
        x_h = _tail_fill_within_phase(
            x_h,
            {
                vi_name: {phase: [col for _, col in entries] for phase, entries in phase_map.items()}
                for vi_name, phase_map in vi_phase_entries.items()
            },
        )

    feature_frames: list[pd.DataFrame] = []
    feature_specs: list[dict[str, Any]] = []
    anchor_counts_by_phase: dict[str, int] = {}
    anchor_definition_rows: list[dict[str, Any]] = []

    for anchor_def in anchor_definitions:
        anchor_phase = _normalize_phase_name(anchor_def["anchor_phase"])
        anchor_counts_by_phase[anchor_phase] = anchor_counts_by_phase.get(anchor_phase, 0) + 1
        anchor_definition_rows.append(
            {
                "anchor_idx": int(anchor_def["anchor_idx"]),
                "anchor_phase": anchor_phase,
                "anchor_tb": int(anchor_def["anchor_tb"]),
                "anchor_token": str(anchor_def["anchor_token"]),
                "anchor_time_token": str(anchor_def["anchor_time_token"]),
                "anchor_start_time_token": str(anchor_def["anchor_start_time_token"]),
                "anchor_end_time_token": str(anchor_def["anchor_end_time_token"]),
                "bin_size": int(anchor_def["bin_size"]),
                "anchor_window_radius": int(radius),
            }
        )
        center_anchor_idx = int(anchor_def["anchor_idx"])
        neighbor_anchor_indices = [
            idx
            for idx in range(center_anchor_idx - radius, center_anchor_idx + radius + 1)
            if idx in center_key_by_anchor
        ]
        neighbor_time_keys = {center_key_by_anchor[idx] for idx in neighbor_anchor_indices}
        for vi_name, entries in vi_timeline_entries.items():
            local_entries = [(time_key, raw_col_name) for time_key, raw_col_name in entries if time_key in neighbor_time_keys]
            if not local_entries:
                continue
            group_id = f"{anchor_def['anchor_token']}||{vi_name}"
            for group_pos, (time_key, raw_col_name) in enumerate(sorted(local_entries, key=lambda x: time_key_sort_key(x[0]))):
                raw_anchor_idx = next(
                    (idx for idx, center_key in center_key_by_anchor.items() if center_key == time_key),
                    center_anchor_idx,
                )
                relative_offset = int(raw_anchor_idx - center_anchor_idx)
                feat_name = f"{group_id}||rel_{relative_offset:+d}||{raw_col_name}"
                feature_frames.append(x_h.loc[:, raw_col_name].astype(float).rename(feat_name).to_frame())
                feature_specs.append(
                    {
                        "feature": feat_name,
                        "source_feature": str(raw_col_name),
                        "vi_name": vi_name,
                        "family": _family_for_vi(vi_name),
                        "phase": str(time_key[0] or anchor_phase),
                        "anchor_phase": anchor_phase,
                        "anchor_idx": int(anchor_def["anchor_idx"]),
                        "anchor_token": str(anchor_def["anchor_token"]),
                        "anchor_tb": int(anchor_def["anchor_tb"]),
                        "anchor_time_token": str(anchor_def["anchor_time_token"]),
                        "anchor_start_time_token": str(anchor_def["anchor_start_time_token"]),
                        "anchor_end_time_token": str(anchor_def["anchor_end_time_token"]),
                        "center_time_token": str(anchor_def["anchor_time_token"]),
                        "feature_type": "anchor_point_raw_value" if radius == 0 else "anchor_point_local_value",
                        "raw_time_token": time_key_to_token(time_key),
                        "relative_offset": int(relative_offset),
                        "relative_side": "center" if relative_offset == 0 else ("left" if relative_offset < 0 else "right"),
                        "window_size": int(len(local_entries)),
                        "left_context": int(min(radius, center_anchor_idx)),
                        "right_context": int(min(radius, len(anchor_definitions) - center_anchor_idx - 1)),
                        "group_id": group_id,
                        "anchor_vi_group": group_id,
                        "group_pos": int(group_pos),
                        "bin_size": 1,
                        "anchor_window_radius": int(radius),
                        "source_anchor_idx": int(raw_anchor_idx),
                    }
                )

    x_anchor = pd.concat(feature_frames, axis=1) if feature_frames else pd.DataFrame(index=bundle.meta.index)
    x_anchor = x_anchor.loc[bundle.meta.index].copy()
    feature_spec_df = pd.DataFrame(feature_specs)
    group_axis_df = (
        feature_spec_df.loc[
            :,
            [
                "group_id",
                "anchor_idx",
                "anchor_token",
                "anchor_phase",
                "anchor_tb",
                "anchor_start_time_token",
                "anchor_end_time_token",
                "vi_name",
                "family",
                "bin_size",
            ],
        ]
        .drop_duplicates()
        .sort_values(["anchor_idx", "vi_name"], kind="stable")
        .reset_index(drop=True)
        if not feature_spec_df.empty
        else pd.DataFrame()
    )
    info = {
        "representation": representation,
        "n_samples": int(x_anchor.shape[0]),
        "n_features_total": int(x_anchor.shape[1]),
        "n_vi": int(len(vi_timeline_entries)),
        "vi_names": list(vi_timeline_entries.keys()),
        "families": {vi: _family_for_vi(vi) for vi in vi_timeline_entries.keys()},
        "n_anchor_bins_requested": int(config.n_anchor_bins),
        "n_anchor_bins_effective": int(len(anchor_definitions)),
        "anchor_window_radius": int(radius),
        "n_anchor_by_phase": anchor_counts_by_phase,
        "n_group_candidates": int(feature_spec_df["anchor_vi_group"].astype(str).nunique()) if not feature_spec_df.empty else 0,
        "anchor_bin_definition_df": pd.DataFrame(anchor_definition_rows),
        "group_axis_df": group_axis_df,
        "feature_spec_df": feature_spec_df,
    }
    return x_anchor, info


def build_h_anchor_local_features(
    bundle: MultiTargetDataBundle,
    config: AnchorLocalConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return _build_h_anchor_bin_features(bundle, config=config, representation="H_anchor_bin")


def build_h_anchor_vi_features(
    bundle: MultiTargetDataBundle,
    config: AnchorLocalConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return _build_h_anchor_bin_features(bundle, config=config, representation="H_anchor_bin_vi")
