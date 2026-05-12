import pandas as pd

from crophg.public.result_3_4b import (
    build_anchor_frequency_table,
    build_kept_group_evolution,
    build_vi_frequency_table,
)


def _mock_metrics() -> pd.DataFrame:
    rows = []
    for scenario in ["reference"]:
        for target in ["ActualYD"]:
            for variant in ["H_ANCHOR_AUTO", "G+H_ANCHOR_AUTO"]:
                for anchor_idx, anchor_order in [(0, 1), (1, 2)]:
                    rows.append(
                        {
                            "scenario": scenario,
                            "target": target,
                            "input_variant": variant,
                            "anchor_idx": anchor_idx,
                            "anchor_order": anchor_order,
                            "anchor_tb": (anchor_order - 1) * 150,
                            "anchor_phase": "sow",
                            "n_group_kept": 3,
                            "n_anchor_kept": 2,
                            "n_vi_kept": 1,
                            "growth_aware_min_groups": 4,
                            "selected_window_radius": 1,
                            "kept_group_ids_json": '["a||NDRE"]',
                            "kept_anchor_tokens_json": '["H+70"]',
                            "kept_vi_names_json": '["NDRE"]',
                        }
                    )
    return pd.DataFrame(rows)


def test_3_4b_builders_work() -> None:
    df = _mock_metrics()
    evo = build_kept_group_evolution(df)
    vi = build_vi_frequency_table(evo)
    anchor = build_anchor_frequency_table(evo)

    assert not evo.empty
    assert not vi.empty
    assert not anchor.empty
    assert "union_anchor_tokens_text" in evo.columns
    assert "vi_name" in vi.columns
    assert "kept_anchor_token" in anchor.columns

