# Step 0b: migrate 3.1B into crophg

## Entry

```bash
PYTHONPATH=src python3 scripts/result_3_1b.py --config configs/3_1b/formal_run_example.yaml --output-dir /path/to/output
```

## Meaning

- `3.1B` 固定为四场景、六个 trait、`G+H_FULL`、`5` 个模型。
- `G` 表示默认采用 `grm_pca` 作为基因型表示。

## Expected outputs

- `metrics_summary.csv`
- `metrics_by_scenario_target.csv`
- `summary.md`
- `run_config.yaml`

## Validation

```bash
PYTHONPATH=src python3 scripts/result_3_1b.py --print-spec
```

Formal analysis:

```bash
PYTHONPATH=src python scripts/result_3_1b_analysis.py \
  --input-dir /path/to/result_3_1b_output \
  --output-dir /path/to/report_dir
```

Formal analysis config template:

- `configs/3_1b/formal_analysis_example.yaml`

Formal analysis expected outputs:

- `result_3_1b_formal_analysis.md`
- `feature_overview.csv`
- `best_accuracy_by_scenario_target.csv`
- `scenario_mean_summary.csv`
- `trait_mean_across_scenarios.csv`
- `run_notes.json`

## Smoke status

- 已在本地 `PEG2P` 环境完成最小 smoke。
- 结果目录示例：`/private/tmp/crophg_smoke_result_3_1b`。
- 该 smoke 使用 `reference + ActualYD + ridge` 的最小 task filter，用于验证 `G+H_FULL` 路径可独立执行。
