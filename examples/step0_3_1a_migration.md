# Step 0: migrate 3.1A into crophg

## Entry

```bash
PYTHONPATH=src python3 scripts/result_3_1a.py --config configs/3_1a/formal_run_example.yaml --output-dir /path/to/output
```

## Meaning

- `3.1A` 固定为四场景、六个 trait、`H_FULL`、`5` 个模型。
- 该入口会在 wrapper 层自动收紧配置到当前论文正式口径。

## Expected outputs

- `metrics_summary.csv`
- `metrics_by_scenario_target.csv`
- `summary.md`
- `run_config.yaml`

## Validation

```bash
PYTHONPATH=src python3 scripts/result_3_1a.py --print-spec
```

Formal analysis:

```bash
PYTHONPATH=src python scripts/result_3_1a_analysis.py \
  --input-dir /path/to/result_3_1a_output \
  --output-dir /path/to/report_dir
```

Formal analysis config template:

- `configs/3_1a/formal_analysis_example.yaml`

Formal analysis expected outputs:

- `result_3_1a_formal_analysis.md`
- `feature_overview.csv`
- `best_accuracy_by_scenario_target.csv`
- `scenario_mean_summary.csv`
- `trait_deployment_gaps.csv`
- `run_notes.json`

## Smoke status

- 已在本地 `PEG2P` 环境完成最小 smoke。
- 结果目录示例：`/private/tmp/crophg_smoke_result_3_1a`。
- 该 smoke 使用 `reference + ActualYD + ridge` 的最小 task filter，用于验证 `CropHG` 可独立落出 `summary.md` 与 `run_config.yaml`。
