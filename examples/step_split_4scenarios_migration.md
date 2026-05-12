# 四场景 Train/Val/Test 迁移

## Entry

```bash
PYTHONPATH=src python scripts/build_four_scenario_splits.py --config configs/data_splits_4scenarios_example.yaml
```

## Config

Reference template:

- `configs/data_splits_4scenarios_example.yaml`

Current implementation expects:

- `data/processed/master_plot_metadata.parquet`
- `outputs/reports/loso_genotype_nested_genotype_fold_map.csv`

## Output validation

A successful run should create:

- `data/processed/splits_4scenarios/*.json`
- `outputs/reports/data_split_summary_4scenarios.md`
- `outputs/reports/data_split_summary_4scenarios.json`

Recommended test command:

```bash
PYTHONPATH=src python -m pytest tests/test_data_splits_4scenarios.py tests/test_framework.py -q
```
