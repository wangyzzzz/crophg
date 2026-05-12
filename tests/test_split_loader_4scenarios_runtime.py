from pathlib import Path

from models.common.split_loader import load_split_groups


def test_split_loader_supports_migrated_four_scenarios() -> None:
    split_dir = Path('data/processed/splits_4scenarios')

    ref = load_split_groups(split_dir, validation_scenario='reference')
    ws = load_split_groups(split_dir, validation_scenario='within_season')
    loso = load_split_groups(split_dir, validation_scenario='loso')
    loso_geno = load_split_groups(split_dir, validation_scenario='loso_genotype')

    assert ref
    assert ws
    assert loso
    assert loso_geno
    assert (2022, 0) in ref
    assert (2022, 0) in ws
    assert (2022, 0) in loso
    assert (2022, 0) in loso_geno
