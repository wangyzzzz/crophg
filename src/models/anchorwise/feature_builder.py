from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data_loader import DayDataBundle, TimeKey, time_key_sort_key, time_key_to_token


@dataclass
class AnchorDefinition:
    anchor_idx: int
    anchor_phase: Optional[str]
    anchor_tb: int
    anchor_tb_token: str
    anchor_key: TimeKey
    climate_static_col_count: int
    hyperspectral_static_col_count: int
    climate_prefix_col_count: int
    hyperspectral_prefix_col_count: int


def _linspace_unique_indices(n_points: int, n_samples: int) -> List[int]:
    if n_samples <= 0:
        return []
    if n_samples >= n_points:
        return list(range(n_points))

    raw = np.linspace(0, n_points - 1, n_samples)
    idxs = [int(round(x)) for x in raw]

    # 去重后若数量不足，补最近未使用索引
    used = []
    seen = set()
    for idx in idxs:
        if idx not in seen:
            used.append(idx)
            seen.add(idx)

    if len(used) < n_samples:
        for idx in range(n_points):
            if idx not in seen:
                used.append(idx)
                seen.add(idx)
                if len(used) == n_samples:
                    break

    return sorted(used[:n_samples])


def select_anchor_tbs(common_tbs: List[TimeKey], n_anchor_bins: int) -> List[TimeKey]:
    if not common_tbs:
        raise ValueError("common_tbs 为空，无法选取锚点。")
    n = min(n_anchor_bins, len(common_tbs))
    idxs = _linspace_unique_indices(len(common_tbs), n)
    return [common_tbs[i] for i in idxs]


def build_anchor_definitions(bundle: DayDataBundle, n_anchor_bins: int) -> List[AnchorDefinition]:
    anchors = select_anchor_tbs(bundle.common_tbs, n_anchor_bins)
    out: List[AnchorDefinition] = []
    for i, time_key in enumerate(anchors):
        phase, tb = time_key
        climate_static = bundle.climate_tb_map.get(time_key, [])
        hypers_static = bundle.hyperspectral_tb_map.get(time_key, [])

        climate_prefix = [
            c for t, cols in bundle.climate_tb_map.items() if time_key_sort_key(t) <= time_key_sort_key(time_key) for c in cols
        ]
        hypers_prefix = [
            c
            for t, cols in bundle.hyperspectral_tb_map.items()
            if time_key_sort_key(t) <= time_key_sort_key(time_key)
            for c in cols
        ]

        out.append(
            AnchorDefinition(
                anchor_idx=i,
                anchor_phase=phase,
                anchor_tb=tb,
                anchor_tb_token=time_key_to_token(time_key),
                anchor_key=time_key,
                climate_static_col_count=len(climate_static),
                hyperspectral_static_col_count=len(hypers_static),
                climate_prefix_col_count=len(climate_prefix),
                hyperspectral_prefix_col_count=len(hypers_prefix),
            )
        )
    return out


def _select_modal_cols(tb_map: Dict[TimeKey, List[str]], anchor_key: TimeKey, input_type: str) -> List[str]:
    if input_type == "static":
        return list(tb_map.get(anchor_key, []))
    if input_type == "temporal":
        cols: List[str] = []
        anchor_sort = time_key_sort_key(anchor_key)
        for tb, group_cols in tb_map.items():
            if time_key_sort_key(tb) <= anchor_sort:
                cols.extend(group_cols)
        return cols
    raise ValueError(f"未知 input_type: {input_type}")


def build_feature_matrix(bundle: DayDataBundle, anchor_key: TimeKey, input_type: str) -> tuple[pd.DataFrame, dict]:
    anchor_phase, anchor_tb = anchor_key
    h_cols = _select_modal_cols(bundle.hyperspectral_tb_map, anchor_key, input_type)
    c_cols = _select_modal_cols(bundle.climate_tb_map, anchor_key, input_type)
    g_cols = list(bundle.x_genotype.columns)

    x_h = bundle.x_hyperspectral[h_cols] if h_cols else pd.DataFrame(index=bundle.meta.index)
    x_c = bundle.x_climate[c_cols] if c_cols else pd.DataFrame(index=bundle.meta.index)
    x_g = bundle.x_genotype[g_cols]

    x = pd.concat([x_h, x_c, x_g], axis=1)
    x = x.loc[bundle.meta.index]

    info = {
        "anchor_tb": anchor_tb,
        "anchor_phase": anchor_phase,
        "input_type": input_type,
        "n_features_total": int(x.shape[1]),
        "n_features_h": int(len(h_cols)),
        "n_features_c": int(len(c_cols)),
        "n_features_g": int(len(g_cols)),
        "h_cols": h_cols,
        "c_cols": c_cols,
        "g_cols": g_cols,
    }
    return x, info
