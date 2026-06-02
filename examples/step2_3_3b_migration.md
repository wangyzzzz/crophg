# Step 2: migrate 3.3B into crophg

## Entry

```bash
PYTHONPATH=src python scripts/cropvig_2.py --input-dir /path/to/result_dir --output-dir /path/to/report_dir
```

## Config

Reference template:

- `configs/3_3b/formal_analysis_example.yaml`

Current implementation reads the result directory directly and expects:

- `metrics_summary.csv`
- `metrics_by_fold.csv`

## Output validation

A successful run should create:

- `cropvig_2_formal_analysis.md`
- `trait_level_compare.csv`
- `trait_overall_summary.csv`
- `scenario_summary.csv`
- `overall_summary.csv`
- `gain_summary.csv`
- `feature_summary.csv`
- `compression_summary.csv`
- `auto_window_selection_counts.csv`
- `run_notes.json`

Recommended test command:

```bash
PYTHONPATH=src python -m pytest tests/test_cropvig_2.py tests/test_framework.py -q
```
