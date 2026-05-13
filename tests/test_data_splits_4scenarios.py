from pathlib import Path
import json
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


def _create_config(root: Path, *, hard_inner_policy: str | None = None) -> Path:
    loso_cfg = {
        "custom_split_strategy": "loso_known_genotype",
        "genotype_fold_map_path": "outputs/reports/loso_genotype_nested_genotype_fold_map.csv",
    }
    loso_genotype_cfg = {
        "custom_split_strategy": "loso_genotype_unknown",
        "genotype_fold_map_path": "outputs/reports/loso_genotype_nested_genotype_fold_map.csv",
    }
    if hard_inner_policy:
        loso_cfg["inner_validation_policy"] = hard_inner_policy
        loso_genotype_cfg["inner_validation_policy"] = hard_inner_policy

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
                "loso": loso_cfg,
                "loso_genotype": loso_genotype_cfg,
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


def test_genotype_novel_validation_fold_is_globally_unseen() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _create_mock_inputs(root)
        cfg_path = _create_config(root)
        build_four_scenario_splits(config_path=cfg_path, repo_root=root)

        meta = pd.read_parquet(root / "data" / "processed" / "master_plot_metadata.parquet")
        geno_by_sample = dict(zip(meta["plot_id"].astype(str), meta["genotype_id"].astype(str)))
        split_paths = sorted(
            (root / "data" / "processed" / "splits_4scenarios").glob("within_season_known_year_2022_outer0_inner*.json")
        )
        assert len(split_paths) == 4
        split = json.loads(split_paths[0].read_text(encoding="utf-8"))

        train_genotypes = {geno_by_sample[x] for x in split["train_ids"]}
        val_genotypes = {geno_by_sample[x] for x in split["val_ids"]}
        test_genotypes = {geno_by_sample[x] for x in split["test_ids"]}

        assert train_genotypes.isdisjoint(val_genotypes)
        assert train_genotypes.isdisjoint(test_genotypes)
        assert {x[:5] for x in split["val_ids"]} == {"P2022", "P2024", "P2025"}


def test_inner_oof_only_expands_first_outer_fold() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _create_mock_inputs(root)
        cfg_path = _create_config(root)
        build_four_scenario_splits(config_path=cfg_path, repo_root=root)

        split_dir = root / "data" / "processed" / "splits_4scenarios"

        assert len(sorted(split_dir.glob("reference_2022_outer0_inner*.json"))) == 4
        assert len(sorted(split_dir.glob("reference_2022_outer1_inner*.json"))) == 1
        assert len(sorted(split_dir.glob("within_season_known_year_2022_outer0_inner*.json"))) == 4
        assert len(sorted(split_dir.glob("within_season_known_year_2022_outer1_inner*.json"))) == 1
        assert len(sorted(split_dir.glob("loso_known_genotype_2022_outer0_inner*.json"))) == 10
        assert len(sorted(split_dir.glob("loso_known_genotype_2022_outer1_inner*.json"))) == 1
        assert len(sorted(split_dir.glob("loso_genotype_unknown_2022_outer0_inner*.json"))) == 4
        assert len(sorted(split_dir.glob("loso_genotype_unknown_2022_outer1_inner*.json"))) == 1


def test_reference_inner_oof_stays_within_target_year() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _create_mock_inputs(root)
        cfg_path = _create_config(root)
        build_four_scenario_splits(config_path=cfg_path, repo_root=root)

        split_dir = root / "data" / "processed" / "splits_4scenarios"
        for split_path in sorted(split_dir.glob("reference_2022_outer0_inner*.json")):
            split = json.loads(split_path.read_text(encoding="utf-8"))
            assert {x[:5] for x in split["val_ids"]} == {"P2022"}
            assert {x[:5] for x in split["test_ids"]} == {"P2022"}


def test_year_novel_inner_oof_uses_non_test_year_cells() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _create_mock_inputs(root)
        cfg_path = _create_config(root)
        build_four_scenario_splits(config_path=cfg_path, repo_root=root)

        split_dir = root / "data" / "processed" / "splits_4scenarios"
        split_paths = sorted(split_dir.glob("loso_known_genotype_2022_outer0_inner*.json"))
        assert len(split_paths) == 10
        val_year_sets = []
        for split_path in split_paths:
            split = json.loads(split_path.read_text(encoding="utf-8"))
            val_years = {x[:5] for x in split["val_ids"]}
            test_years = {x[:5] for x in split["test_ids"]}
            train_years = {x[:5] for x in split["train_ids"]}
            assert test_years == {"P2022"}
            assert "P2022" not in val_years
            assert "P2022" not in train_years
            assert len(val_years) == 1
            val_year_sets.append(next(iter(val_years)))
        assert set(val_year_sets) == {"P2024", "P2025"}


def test_joint_novel_validation_fold_is_globally_unseen_in_non_test_years() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _create_mock_inputs(root)
        cfg_path = _create_config(root)
        build_four_scenario_splits(config_path=cfg_path, repo_root=root)

        meta = pd.read_parquet(root / "data" / "processed" / "master_plot_metadata.parquet")
        geno_by_sample = dict(zip(meta["plot_id"].astype(str), meta["genotype_id"].astype(str)))
        split_paths = sorted(
            (root / "data" / "processed" / "splits_4scenarios").glob("loso_genotype_unknown_2022_outer0_inner*.json")
        )
        assert len(split_paths) == 4
        split = json.loads(split_paths[0].read_text(encoding="utf-8"))

        train_genotypes = {geno_by_sample[x] for x in split["train_ids"]}
        val_genotypes = {geno_by_sample[x] for x in split["val_ids"]}
        test_genotypes = {geno_by_sample[x] for x in split["test_ids"]}

        assert train_genotypes.isdisjoint(val_genotypes)
        assert train_genotypes.isdisjoint(test_genotypes)
        assert {x[:5] for x in split["val_ids"]} == {"P2024", "P2025"}


def test_ood_cell_policy_uses_single_non_test_year_for_year_novel_inner_validation() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _create_mock_inputs(root)
        cfg_path = _create_config(root, hard_inner_policy="ood_cell")
        build_four_scenario_splits(config_path=cfg_path, repo_root=root)

        split_paths = sorted(
            (root / "data" / "processed" / "splits_4scenarios").glob("loso_known_genotype_2022_outer0_inner*.json")
        )
        assert len(split_paths) == 10
        for split_path in split_paths:
            split = json.loads(split_path.read_text(encoding="utf-8"))
            val_years = {x[:5] for x in split["val_ids"]}
            train_years = {x[:5] for x in split["train_ids"]}
            test_years = {x[:5] for x in split["test_ids"]}

            assert test_years == {"P2022"}
            assert len(val_years) == 1
            assert "P2022" not in val_years
            assert "P2022" not in train_years
            assert val_years.isdisjoint(train_years)


def test_ood_cell_policy_uses_year_by_fold_cells_for_joint_novel_inner_validation() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _create_mock_inputs(root)
        cfg_path = _create_config(root, hard_inner_policy="ood_cell")
        build_four_scenario_splits(config_path=cfg_path, repo_root=root)

        meta = pd.read_parquet(root / "data" / "processed" / "master_plot_metadata.parquet")
        geno_by_sample = dict(zip(meta["plot_id"].astype(str), meta["genotype_id"].astype(str)))
        split_paths = sorted(
            (root / "data" / "processed" / "splits_4scenarios").glob("loso_genotype_unknown_2022_outer0_inner*.json")
        )
        assert len(split_paths) == 8

        for split_path in split_paths:
            split = json.loads(split_path.read_text(encoding="utf-8"))
            val_years = {x[:5] for x in split["val_ids"]}
            train_years = {x[:5] for x in split["train_ids"]}
            test_years = {x[:5] for x in split["test_ids"]}
            train_genotypes = {geno_by_sample[x] for x in split["train_ids"]}
            val_genotypes = {geno_by_sample[x] for x in split["val_ids"]}
            test_genotypes = {geno_by_sample[x] for x in split["test_ids"]}

            assert test_years == {"P2022"}
            assert len(val_years) == 1
            assert "P2022" not in val_years
            assert "P2022" not in train_years
            assert val_years.isdisjoint(train_years)
            assert train_genotypes.isdisjoint(val_genotypes)
            assert train_genotypes.isdisjoint(test_genotypes)
