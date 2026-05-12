from pathlib import Path
import tempfile

import pandas as pd
import yaml

from crophg.internal.data_splits import build_four_scenario_splits


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


def _create_mock_inputs(root: Path) -> None:
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "reports").mkdir(parents=True, exist_ok=True)

    rows = []
    for year in [2022, 2024, 2025]:
        for i in range(12):
            rows.append(
                {
                    "year": year,
                    "plot_id": f"P{year}_{i:03d}",
                    "genotype_id": f"G{i % 6:02d}",
                }
            )
    pd.DataFrame(rows).to_parquet(root / "data" / "processed" / "master_plot_metadata.parquet", index=False)

    geno_map = pd.DataFrame(
        {
            "genotype_id": [f"G{i:02d}" for i in range(6)],
            "genotype_fold": [0, 1, 2, 3, 4, 0],
        }
    )
    geno_map.to_csv(root / "outputs" / "reports" / "loso_genotype_nested_genotype_fold_map.csv", index=False)


def _create_config(root: Path) -> Path:
    cfg = {
        "version": 1,
        "name": "data_splits_4scenarios_test",
        "input": {
            "path": "data/processed/master_plot_metadata.parquet",
            "file_type": "parquet",
            "sample_id_column": "plot_id",
            "required_columns": ["year", "plot_id", "genotype_id"],
        },
        "splits": {
            "random_seed": 42,
            "scenarios": {
                "reference": {
                    "custom_split_strategy": "reference",
                    "genotype_fold_map_path": "outputs/reports/loso_genotype_nested_genotype_fold_map.csv",
                },
                "within_season": {
                    "custom_split_strategy": "within_season_known_year_unknown_genotype",
                    "genotype_fold_map_path": "outputs/reports/loso_genotype_nested_genotype_fold_map.csv",
                },
                "loso": {
                    "custom_split_strategy": "loso_known_genotype",
                    "genotype_fold_map_path": "outputs/reports/loso_genotype_nested_genotype_fold_map.csv",
                },
                "loso_genotype": {
                    "custom_split_strategy": "loso_genotype_unknown",
                    "genotype_fold_map_path": "outputs/reports/loso_genotype_nested_genotype_fold_map.csv",
                },
            },
        },
        "output": {
            "split_dir": "data/processed/splits_4scenarios",
            "summary_report": "outputs/reports/data_split_summary_4scenarios.md",
            "clean_previous_generated_files": True,
        },
    }
    path = root / "configs" / "data_splits_4scenarios_example.yaml"
    _write_yaml(path, cfg)
    return path


def test_build_four_scenario_splits() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _create_mock_inputs(root)
        cfg_path = _create_config(root)
        result = build_four_scenario_splits(config_path=cfg_path, repo_root=root)

        split_dir = root / "data" / "processed" / "splits_4scenarios"
        split_files = sorted(split_dir.glob("*.json"))
        assert split_files
        assert result["total_splits"] == len(split_files)
        assert result["scenario_counts"]["reference"] > 0
        assert result["scenario_counts"]["within_season"] > 0
        assert result["scenario_counts"]["loso"] > 0
        assert result["scenario_counts"]["loso_genotype"] > 0

        report = root / "outputs" / "reports" / "data_split_summary_4scenarios.md"
        assert report.exists()
        text = report.read_text(encoding="utf-8")
        assert "四场景 Train/Val/Test 划分汇总" in text
        assert "reference" in text
        assert "within_season" in text
        assert "loso" in text
        assert "loso_genotype" in text
