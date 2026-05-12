# Step 4: migrate 3.4B into crophg

## Entry

```bash
PYTHONPATH=src python3 scripts/result_3_4b.py --input-dir /path/to/result_dir --output-dir /path/to/report_dir
```

## Config

Reference template:

- `configs/3_4b/formal_analysis_example.yaml`

Current implementation reads the result directory directly and expects:

- `metrics_by_fold.csv`

## Output validation

A successful run should create:

- `result_3_4b_formal_analysis.md`
- `kept_group_evolution.csv`
- `kept_vi_frequency.csv`
- `kept_anchor_frequency.csv`
- `run_notes.json`

Recommended test command:

```bash
PYTHONPATH=src python -m pytest tests/test_result_3_4b.py tests/test_framework.py -q
```
