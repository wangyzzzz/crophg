from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .io_utils import read_json


@dataclass
class WithinSeasonSplit:
    split_name: str
    target_year: int
    outer_fold: int
    inner_fold: int
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]
    random_seed: int | None


REFERENCE_PATTERN = re.compile(r"^reference_(\d+)_outer(\d+)_inner(\d+)\.json$")
WITHIN_SEASON_PATTERN = re.compile(r"^within_season(?:_known_year)?_(\d+)_outer(\d+)_inner(\d+)\.json$")
LOSO_PATTERN = re.compile(r"^loso_(\d+)_fold(\d+)\.json$")
LOSO_KNOWN_GENOTYPE_PATTERN = re.compile(r"^loso_known_genotype_(\d+)_outer(\d+)_inner(\d+)\.json$")
LOSO_GENOTYPE_NESTED_PATTERN = re.compile(r"^loso_genotype(?:_unknown)?_(\d+)_outer(\d+)_inner(\d+)\.json$")


def _build_split(path: Path, *, target_year: int, outer_fold: int, inner_fold: int) -> WithinSeasonSplit:
    obj = read_json(path)
    return WithinSeasonSplit(
        split_name=obj.get('split_name', path.stem),
        target_year=int(obj.get('target_year', target_year)),
        outer_fold=int(obj.get('outer_fold', outer_fold)),
        inner_fold=int(obj.get('inner_fold', inner_fold)),
        train_ids=[str(x) for x in obj.get('train_ids', [])],
        val_ids=[str(x) for x in obj.get('val_ids', [])],
        test_ids=[str(x) for x in obj.get('test_ids', [])],
        random_seed=obj.get('random_seed'),
    )


def _load_nested_splits(split_dir: Path, *, pattern: re.Pattern[str], glob_pattern: str) -> Dict[Tuple[int, int], List[WithinSeasonSplit]]:
    grouped: Dict[Tuple[int, int], List[WithinSeasonSplit]] = {}
    for path in sorted(split_dir.glob(glob_pattern)):
        match = pattern.match(path.name)
        if not match:
            continue
        target_year = int(match.group(1))
        outer_fold = int(match.group(2))
        inner_fold = int(match.group(3))
        split = _build_split(path, target_year=target_year, outer_fold=outer_fold, inner_fold=inner_fold)
        key = (split.target_year, split.outer_fold)
        grouped.setdefault(key, []).append(split)
    for key, splits in grouped.items():
        grouped[key] = sorted(splits, key=lambda x: x.inner_fold)
    return dict(sorted(grouped.items(), key=lambda x: x[0]))


def load_reference_splits(split_dir: Path) -> Dict[Tuple[int, int], List[WithinSeasonSplit]]:
    return _load_nested_splits(split_dir, pattern=REFERENCE_PATTERN, glob_pattern='reference_*.json')


def load_within_season_splits(split_dir: Path) -> Dict[Tuple[int, int], List[WithinSeasonSplit]]:
    return _load_nested_splits(split_dir, pattern=WITHIN_SEASON_PATTERN, glob_pattern='within_season*.json')


def load_loso_splits(split_dir: Path) -> Dict[Tuple[int, int], List[WithinSeasonSplit]]:
    grouped: Dict[Tuple[int, int], List[WithinSeasonSplit]] = {}

    for path in sorted(split_dir.glob('loso*.json')):
        obj = read_json(path)

        m_known = LOSO_KNOWN_GENOTYPE_PATTERN.match(path.name)
        if m_known:
            target_year = int(m_known.group(1))
            outer_fold = int(obj.get('outer_fold', m_known.group(2)))
            inner_fold = int(obj.get('inner_fold', m_known.group(3)))
        else:
            m_geno = LOSO_GENOTYPE_NESTED_PATTERN.match(path.name)
            if m_geno:
                continue
            match = LOSO_PATTERN.match(path.name)
            if not match:
                continue
            target_year = int(match.group(1))
            fold_idx = int(match.group(2))
            outer_fold = obj.get('outer_fold')
            if outer_fold is None:
                outer_fold = 0
            inner_fold = obj.get('fold')
            if inner_fold is None:
                inner_fold = obj.get('inner_fold')
            if inner_fold is None:
                inner_fold = fold_idx

        split = WithinSeasonSplit(
            split_name=obj.get('split_name', path.stem),
            target_year=int(obj.get('target_year', target_year)),
            outer_fold=int(outer_fold),
            inner_fold=int(inner_fold),
            train_ids=[str(x) for x in obj.get('train_ids', [])],
            val_ids=[str(x) for x in obj.get('val_ids', [])],
            test_ids=[str(x) for x in obj.get('test_ids', [])],
            random_seed=obj.get('random_seed'),
        )
        key = (split.target_year, split.outer_fold)
        grouped.setdefault(key, []).append(split)

    for key, splits in grouped.items():
        grouped[key] = sorted(splits, key=lambda x: x.inner_fold)
    return dict(sorted(grouped.items(), key=lambda x: x[0]))


def load_loso_genotype_splits(split_dir: Path) -> Dict[Tuple[int, int], List[WithinSeasonSplit]]:
    return _load_nested_splits(split_dir, pattern=LOSO_GENOTYPE_NESTED_PATTERN, glob_pattern='loso_genotype*.json')


def load_split_groups(split_dir: Path, validation_scenario: str) -> Dict[Tuple[int, int], List[WithinSeasonSplit]]:
    scenario = str(validation_scenario).strip().lower().replace('-', '_')
    if scenario == 'reference':
        return load_reference_splits(split_dir)
    if scenario == 'within_season':
        return load_within_season_splits(split_dir)
    if scenario == 'loso':
        return load_loso_splits(split_dir)
    if scenario in {'loso_genotype', 'genotype_loso', 'loso_plus_genotype'}:
        return load_loso_genotype_splits(split_dir)
    raise ValueError(f'不支持的 validation_scenario: {validation_scenario}')
