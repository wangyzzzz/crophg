from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple


OuterKey = Tuple[int, int]


def normalize_outer_groups(
    prepared_groups: Dict[OuterKey, Any],
) -> list[tuple[OuterKey, Any]]:
    return sorted(
        prepared_groups.items(),
        key=lambda kv: (int(kv[0][0]), int(kv[0][1])),
    )


def resolve_optuna_reuse_policy(optuna_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(optuna_cfg or {})
    enabled = bool(cfg.get("reuse_best_params_across_outer_folds", False))
    mode = str(cfg.get("reuse_scope", "task_first_outer_fold")).strip().lower()
    if mode not in {"task_first_outer_fold", "disabled"}:
        raise ValueError(
            "optuna.reuse_scope 仅支持 'task_first_outer_fold' 或 'disabled'，"
            f"当前为: {cfg.get('reuse_scope')!r}"
        )
    if not enabled:
        mode = "disabled"
    return {
        "enabled": bool(enabled and mode != "disabled"),
        "mode": mode,
    }


def should_search_on_outer_fold(
    *,
    ordered_outer_keys: Iterable[OuterKey],
    current_key: OuterKey,
    reuse_policy: dict[str, Any],
) -> bool:
    if not bool(reuse_policy.get("enabled", False)):
        return True
    ordered = list(ordered_outer_keys)
    if not ordered:
        return True
    return tuple(current_key) == tuple(ordered[0])
