# Step 0c: migrate 3.2A into crophg

## Entry

```bash
PYTHONPATH=src python3 scripts/result_3_2a.py --config configs/3_2a/formal_run_example.yaml --output-dir /path/to/output
```

## Meaning

- `3.2A` 固定为 `G` 与 `GH_SINGLE` 的单 anchor 主体。
- 该入口用于生成 `single_anchor_delta.csv`、`best_anchor_by_trait_scenario.csv` 与 factor-group importance。

## Expected outputs

- `metrics_summary.csv`
- `single_anchor_delta.csv`
- `best_anchor_by_trait_scenario.csv`
- `factor_group_importance.csv`
- `summary.md`

## Validation

```bash
PYTHONPATH=src python3 scripts/result_3_2a.py --print-spec
```

Formal analysis:

```bash
PYTHONPATH=src python scripts/result_3_2a_analysis.py \
  --input-dir /path/to/result_3_2a_output \
  --output-dir /path/to/report_dir
```

Formal analysis config template:

- `configs/3_2a/formal_analysis_example.yaml`

Formal analysis expected outputs:

- `result_3_2a_formal_analysis.md`
- `anchor_delta_grouped.csv`
- `anchor_delta_overview.csv`
- `top_anchor_by_scenario_trait.csv`
- `best_anchor_consistency.csv`
- `factor_group_importance_summary.csv`
- `result_3_2a_anchor_delta_heatmap.png`
- `run_notes.json`

## Smoke status

- 已在本地 `PEG2P` 环境完成最小 smoke。
- 结果目录示例：`/private/tmp/crophg_smoke_result_3_2a_models`。
- 当前最稳定的 smoke 配置会关闭 smoke 层的额外 ablation，只保留 `G` 与 `GH_SINGLE` 主任务，用于验证 `single_anchor_delta.csv`、`best_anchor_by_trait_scenario.csv`、`summary.md` 的独立落盘。
