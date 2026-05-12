# Step 0e: migrate 3.2C into crophg

## Entry

```bash
PYTHONPATH=src python3 scripts/result_3_2c.py \
  --config configs/3_2c/formal_run_example.yaml \
  --input-dir /path/to/result_3_2a_output \
  --output-dir /path/to/output
```

## Meaning

- `3.2C` 与 `3.2B` 使用同一 shared-anchor/no-svr 运行主体。
- 论文层面的差异体现在后续分析解释：`3.2B` 关注 `H_SINGLE_VI`，`3.2C` 关注 `GH_SINGLE_VI` 相对 `G` 的边际增量。

## Expected outputs

- `best_anchor_input.csv`
- `available_vi_by_anchor.csv`
- `metrics_summary.csv`
- `summary.md`

## Validation

```bash
PYTHONPATH=src python3 scripts/result_3_2c.py --print-spec
```

Formal analysis:

```bash
PYTHONPATH=src python scripts/result_3_2c_analysis.py \
  --input-dir /path/to/result_3_2c_output \
  --anchor-source-dir /path/to/result_3_2a_output \
  --output-dir /path/to/report_dir
```

Formal analysis config template:

- `configs/3_2c/formal_analysis_example.yaml`

Formal analysis expected outputs:

- `result_3_2c_formal_analysis.md`
- `shared_anchor_table.csv`
- `g_baseline_by_scenario_trait.csv`
- `same_vi_by_predictor.csv`
- `same_vi_by_scenario_trait.csv`
- `scenario_trait_overview.csv`
- `top_gh_delta_vi_by_scenario_trait.csv`
- `useful_h_but_not_gh.csv`
- `h_weak_but_g_helps.csv`
- `result_3_2c_g_baseline_heatmap.png`
- `result_3_2c_gh_minus_g_vi_trait_heatmaps.png`
- `run_notes.json`

## Smoke status

- 已在本地 `PEG2P` 环境完成最小 smoke。
- 结果目录示例：`/private/tmp/crophg_smoke_result_3_2c`。
- 该 smoke 以 `/private/tmp/crophg_smoke_result_3_2a_models` 作为 `--input-dir`，验证 `3.2A -> shared anchor -> 3.2C` 的独立衔接。
