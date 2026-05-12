# Step 0d: migrate 3.2B into crophg

## Entry

```bash
PYTHONPATH=src python3 scripts/result_3_2b.py \
  --config configs/3_2b/formal_run_example.yaml \
  --input-dir /path/to/result_3_2a_output \
  --output-dir /path/to/output
```

## Meaning

- `3.2B` 会先从 `3.2A/single_anchor_delta.csv` 自动构建正式 shared-anchor。
- 然后以 `no svr` 口径运行 shared-anchor 单 VI 任务。

## Expected outputs

- `best_anchor_input.csv`
- `available_vi_by_anchor.csv`
- `metrics_summary.csv`
- `summary.md`

## Validation

```bash
PYTHONPATH=src python3 scripts/result_3_2b.py --print-spec
```

Formal analysis:

```bash
PYTHONPATH=src python scripts/result_3_2b_analysis.py \
  --input-dir /path/to/result_3_2b_output \
  --anchor-source-dir /path/to/result_3_2a_output \
  --output-dir /path/to/report_dir
```

Formal analysis config template:

- `configs/3_2b/formal_analysis_example.yaml`

Formal analysis expected outputs:

- `result_3_2b_formal_analysis.md`
- `shared_anchor_table.csv`
- `same_vi_by_predictor.csv`
- `same_vi_by_scenario_trait.csv`
- `scenario_trait_overview.csv`
- `top_h_vi_by_scenario_trait.csv`
- `cross_scenario_stability_all.csv`
- `cross_scenario_severe_drop_reference_ge_030.csv`
- `cross_scenario_stable_h_vi.csv`
- `result_3_2b_only_h_vi_trait_heatmaps.png`
- `run_notes.json`

## Smoke status

- 已在本地 `PEG2P` 环境完成最小 smoke。
- 结果目录示例：`/private/tmp/crophg_smoke_result_3_2b`。
- 该 smoke 以 `/private/tmp/crophg_smoke_result_3_2a_models` 作为 `--input-dir`，验证 `3.2A -> shared anchor -> 3.2B` 的独立衔接。
