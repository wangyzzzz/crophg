# Step 1: migrate 3.3A into crophg

## Entry

```bash
PYTHONPATH=src python3 scripts/cropvig_1.py --input-dir /path/to/result_dir --output-dir /path/to/report_dir
```

## Config

Reference template:

- `configs/3_3a/formal_analysis_example.yaml`

Current implementation reads the result directory directly and expects:

- `metrics_summary.csv`
- `metrics_by_fold.csv`

## Output validation

A successful run should create:

- `cropvig_1_formal_analysis.md`
- `trait_level_compare.csv`
- `scenario_summary.csv`
- `overall_summary.csv`
- `auto_window_selection_counts.csv`
- `run_notes.json`

Recommended test command:

```bash
PYTHONPATH=src python -m pytest tests/test_cropvig_1.py tests/test_framework.py -q
```
