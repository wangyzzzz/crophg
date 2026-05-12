# Step 3: migrate 3.4A into crophg

## Entry

```bash
PYTHONPATH=src python3 scripts/result_3_4a.py --input-dir /path/to/result_dir --output-dir /path/to/report_dir
```

## Config

Reference template:

- `configs/3_4a/formal_analysis_example.yaml`

Current implementation reads the result directory directly and expects:

- `metrics_summary.csv`

Paper-facing question:

- `G+FULLH / H_ANCHOR_AUTO / G+H_ANCHOR_AUTO` 是否比 `H_FULL` 更早建立有效预测。

## Output validation

A successful run should create:

- `result_3_4a_formal_analysis.md`
- `prefix_curve_summary.csv`
- `trait_level_compare.csv`
- `scenario_summary.csv`
- `overall_summary.csv`
- `anchor_delta_summary.csv`
- `early_advantage_summary.csv`
- `saturation_summary.csv`
- `run_notes.json`

Recommended test command:

```bash
PYTHONPATH=src python -m pytest tests/test_result_3_4a.py tests/test_framework.py -q
```
