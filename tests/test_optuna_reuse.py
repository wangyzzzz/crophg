import pytest

from models.common.optuna_reuse import (
    normalize_outer_groups,
    resolve_optuna_reuse_policy,
    should_search_on_outer_fold,
)


def test_optuna_reuse_is_disabled_by_default() -> None:
    policy = resolve_optuna_reuse_policy({})
    assert policy == {"enabled": False, "mode": "disabled"}
    assert should_search_on_outer_fold(
        ordered_outer_keys=[(2021, 1), (2021, 2)],
        current_key=(2021, 2),
        reuse_policy=policy,
    )


def test_optuna_reuse_searches_only_first_ordered_outer_fold() -> None:
    groups = {(2022, 2): object(), (2021, 3): object(), (2021, 1): object()}
    ordered_items = normalize_outer_groups(groups)
    ordered_keys = [key for key, _ in ordered_items]
    policy = resolve_optuna_reuse_policy(
        {
            "reuse_best_params_across_outer_folds": True,
            "reuse_scope": "task_first_outer_fold",
        }
    )

    assert ordered_keys == [(2021, 1), (2021, 3), (2022, 2)]
    assert should_search_on_outer_fold(
        ordered_outer_keys=ordered_keys,
        current_key=(2021, 1),
        reuse_policy=policy,
    )
    assert not should_search_on_outer_fold(
        ordered_outer_keys=ordered_keys,
        current_key=(2021, 3),
        reuse_policy=policy,
    )


def test_invalid_optuna_reuse_scope_fails_loudly() -> None:
    with pytest.raises(ValueError, match="reuse_scope"):
        resolve_optuna_reuse_policy(
            {
                "reuse_best_params_across_outer_folds": True,
                "reuse_scope": "per_year",
            }
        )
