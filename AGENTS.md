# AGENTS Notes for CropHG

## Scope

- `CropHG` is an independent minimal library for the final paper-facing Result 3 pipeline.
- Do not call `multi-mic-codex` runtime or import old-repo execution entrypoints as a backend shortcut.
- Reuse old-repo logic only by migrating the necessary code into `CropHG` and then running inside `CropHG` itself.

## Section Boundary

- Internal only:
  - `3.1A`
  - `3.1B`
  - `3.2A`
  - `3.2B`
  - `3.2C`
- Public:
  - `3.3A`
  - `3.3B`
  - `3.4A`
  - `3.4B`

## Migration Principles

- Migrate one paper step at a time.
- Every migrated step must document:
  - entry
  - config
  - output files
  - validation command
- Prefer keeping only the paper-main path; do not carry over old ablation branches unless they are required by the current framework.
- Prefer `CropHG` local modules such as `crophg.*`, `models.*`, and local configs.
- Remove redundant glue code from the old repository during migration instead of copying whole script stacks unchanged.

## Analysis Principles

- Formal analysis in `CropHG` must read `CropHG` outputs directly.
- Do not mix old snapshot outputs, old scenario-specific anchor results, or temporary notebook-style summaries into formal `CropHG` reports.
- For shared-anchor analysis, keep the formal rule explicit in the report and in the generated metadata.
- Markdown/table rendering should keep the no-`tabulate` fallback so analysis remains runnable in `PEG2P`.
- `3.2B` and `3.2C` share one shared-anchor single-VI experiment:
  - the experiment may live under `outputs/experiments/.../3_2b`
  - it must contain both `H_SINGLE_VI` and `GH_SINGLE_VI` rows in `metrics_summary.csv`
  - `3.2B` formal analysis reads `H_SINGLE_VI`
  - `3.2C` formal analysis reads `GH_SINGLE_VI` relative to `G`
  - do not treat a missing standalone `3_2c` experiment directory as missing 3.2C evidence if the combined `3_2b` metrics are present
  - the full pipeline must still write a distinct `outputs/reports/.../3_2c_analysis` directory

## Result 3.3B / 3.4A Boundary

- `3.3B` must be generated from its own independent full-season 20 no-bin experiment.
- `3.3B` should not read `3.4A` prefix outputs as its formal input.
- Pipeline config for `3.3B`: `configs/pipeline_two_traits_gpu2/3_3b_midpoint_r0_3_k0_48_ood_cell_cached_ridge_grid_tie001.yaml`.
- Formal current two-trait `3.3B` experiment output: `outputs/experiments/two_traits_full_pipeline_gpu2/3_3b_midpoint_r0_3_k0_48_ood_cell_cached_ridge_grid_tie001`.
- `3.4A` is the growth-prefix experiment: anchor orders `1..20`, where each prefix cumulatively uses no-bin groups from the first bin through the current bin.
- Pipeline config for `3.4A`: `configs/pipeline_two_traits_gpu2/3_4a_midpoint_r0_3_k0_48_ood_cell_cached_ridge_grid_tie001.yaml`.
- Formal current two-trait `3.4A` experiment output: `outputs/experiments/two_traits_full_pipeline_gpu2/3_4a_midpoint_r0_3_k0_48_ood_cell_cached_ridge_grid_tie001`.
- Current `3.3B/3.4A` H-reduce uses midpoint no-bin anchors:
  - split the full H timeline into 20 continuous bins
  - use each bin's midpoint time key as the target anchor point
  - do not flatten the full bin as one group for the formal current run
- Use the complete hyperspectral H time axis for these 20 bins, not only `common_tbs`.
- `H_FULL` and `G+FULLH` growth-prefix baselines must retain all raw H features inside elapsed H bins.
- Current `3.3B/3.4A` AUTO radius search uses `anchor_window_search.radius_candidates: [0, 1, 2, 3]`.
- Current `3.3B/3.4A` group-count search uses `anchor_local_pruning.min_groups: 0` and `anchor_local_pruning.max_groups: 48`.
- Current `3.3B/3.4A` ridge selection uses a fixed alpha grid `0.001..100` with 21 values, raw inner-val Pearson objective, `tie_tolerance: 0.001`, `fusion_tie_tolerance: 0.001`, and larger-alpha tie preference.
- In `3.4A` growth-prefix runs, radius-expanded features must also obey the current prefix:
  - both center `anchor_idx` and actual `source_anchor_idx` must be `<= anchor_order - 1`
  - never allow radius context to include future anchors beyond the current prefix
- Prefix filtering must apply consistently to `H_FULL`, `G+FULLH`, `H_ANCHOR_AUTO`, and `G+H_ANCHOR_AUTO`; `G` is repeated as a prefix-level reference with unchanged G features.
- Prefix result files must keep `anchor_order`, `anchor_idx`, `anchor_tb`, `anchor_phase`, and `anchor_token` in fold metrics, OOF predictions, inner predictions, and aggregated summaries.
- Formal H-reduce scoring target is always the raw phenotype `y`.
- `anchor_local_pruning.g_aware_score_blend` must stay `1.0`; any non-`1.0` value is a formal口径 error.
- Do not use `y - G_pred`, residualized phenotypes, or branch residuals for `3.3B/3.4A` H group scoring.
- For `G+H_*` variants, H group count/window selection and final prediction use the same raw-`y` inner-val rule: compare `G only`, `H only`, and least-squares `G/H` fusion, then keep the candidate with the best selection metric.

## Optuna Reuse Policy

- For full pipeline runs, enable `optuna.reuse_best_params_across_outer_folds: true`.
- Use `optuna.reuse_scope: task_first_outer_fold`.
- The fixed task unit is the natural runner task:
  - `3.1`: `scenario × timeline × target × predictor`
  - `3.2`: `scenario × timeline × target × modality_combo × predictor × anchor`
  - `3.3`: `scenario × timeline × target × modality_variant × predictor × anchor/vi/ablation_group`
  - `3.4`: `scenario × timeline × target × input_variant × predictor`
- Within each fixed task, only the first valid outer-fold runs Optuna.
- Later outer-folds reuse the first valid outer-fold's best hyperparameters, but must still refit on their own outer training data and predict their own outer test fold.
- Metrics rows must keep every outer-fold's `test_*` metrics.
- Only the fold that actually ran Optuna may have `val_*`, `train_*`, `best_mean_*`, `best_objective`, and nonzero `n_optuna_trials`.
- Reused folds must set `optuna_search_performed: false`, `n_optuna_trials: 0`, and leave formal `val_*` / Optuna objective fields empty.
- Reused folds may record source fields such as `optuna_source_target_year`, `optuna_source_outer_fold`, `optuna_source_trial_number`, and `optuna_source_objective` for auditability.

## Inner-OOF Policy

- Scenario-aware inner-OOF is used only on the first valid outer-fold of each fixed task, aligned with `optuna.reuse_scope: task_first_outer_fold`.
- Later outer-folds must not rerun H-reduce/window selection; they reuse the first outer-fold's selected H columns, selected K, selected radius, and hyperparameters, then refit/predict their own outer test fold.
- Runtime metrics must keep audit fields for H-reduce reuse:
  - `feature_selection_search_performed`
  - `feature_selection_reused_from_first_outer_fold`
  - `feature_selection_source_target_year`
  - `feature_selection_source_outer_fold`
- Current split builder expands only the first sorted `(target_year, outer_fold)` into multiple inner folds:
  - `Reference`: target-year genotype folds excluding the outer test fold; no cross-year same-genotype OOF.
  - `Genotype-Novel`: all years for each validation genotype fold excluding the outer test fold.
  - `Year-Novel`: non-test-year × genotype-fold cells.
  - `Joint-Novel`: non-test years for each validation genotype fold excluding the outer test fold.

## OOD-Cell Inner Validation Experiment

- The `ood_cell` split policy is an explicit follow-up experiment for the observed Joint-Novel PHM failure where ordinary inner-OOF still overestimated `G+H_ANCHOR_AUTO`.
- It must not expose or use outer-test samples for H-reduce, fusion, or model selection.
- Config for building these splits:
  - `configs/data_splits_4scenarios_ood_cell.yaml`
  - output split dir: `data/processed/splits_4scenarios_ood_cell`
- Config for the two-trait 3.3B test:
  - `configs/pipeline_two_traits_gpu2/3_3b_midpoint_r0_3_k0_48_ood_cell_cached.yaml`
- Config for the two-trait 3.4A growth-prefix test:
  - `configs/pipeline_two_traits_gpu2/3_4a_midpoint_r0_3_k0_48_ood_cell_cached.yaml`
- Parameter changes relative to ordinary `inner_oof_cached`:
  - only split generation changes for `Year-Novel` and `Joint-Novel`
  - `Year-Novel`: first outer-fold inner-val uses one non-test-year × one genotype-fold cell; inner-train excludes the outer-test year and the validation year
  - `Joint-Novel`: first outer-fold inner-val uses one non-test-year × one genotype-fold cell; inner-train excludes the outer-test year, validation year, outer-test genotype fold, and validation genotype fold
  - `Reference` and `Genotype-Novel` remain unchanged
  - 20 midpoint no-bin anchors, `radius_candidates: [0, 1, 2, 3]`, `min_groups: 0`, `max_groups: 48`, and `g_aware_score_blend: 1.0` remain unchanged
- Treat this as a candidate robustness fix until the new results are inspected; do not overwrite the prior `3_3b_midpoint_r0_3_k0_48_inner_oof_cached` result.
- Do not overwrite the prior `3_4a_midpoint_r0_3_k0_48` result when testing `ood_cell`; use the explicit `3_4a_midpoint_r0_3_k0_48_ood_cell_cached` output directory.

## Predictor Set

- Formal executable predictors are `ridge`, `lasso`, `elasticnet`, `lightgbm`, and `random_forest`.
- Do not reintroduce `svr`; historical file names containing `no_svr` can remain as provenance.

## Smoke Pipeline

- End-to-end CropHG smoke entry: `scripts/run_smoke_pipeline_gpu2.sh`.
- If `3.2A` smoke outputs already exist and only downstream plumbing needs validation, use:
  - `scripts/run_smoke_pipeline_resume_after_3_2a_gpu2.sh`
- Smoke outputs must stay under:
  - `outputs/experiments/smoke_pipeline_gpu2`
  - `outputs/reports/smoke_pipeline_gpu2`
- Smoke is for plumbing validation only:
  - four deployment scenarios are retained
  - target is reduced to `ActualYD`
  - predictor is reduced to `ridge`
  - `3.2A` uses only the `Reference` scenario and 2 anchor bins
  - `3.4A` uses anchor orders `[1, 10, 20]`
  - `3.3B/3.4A` keep raw-`y` scoring and `g_aware_score_blend: 1.0`, but use `radius_candidates: [0, 1]`, `max_groups: 12`, and a 5-point ridge alpha grid for speed
- Never interpret smoke accuracy as a formal result.

## Six-Trait Full Pipeline

- End-to-end six-trait formal execution entry: `scripts/run_six_traits_full_pipeline_gpu2.sh`.
- Six-trait formal outputs must stay under:
  - `outputs/experiments/six_traits_full_pipeline_gpu2`
  - `outputs/reports/six_traits_full_pipeline_gpu2`
- Six-trait targets are fixed as:
  - `ActualYD`, `CM`, `LM`, `PHM`, `Spike`, `TKW`
- Six-trait `3.3B/3.4A` must keep the same formal `tie001`口径 as the accepted two-trait run:
  - raw-`y` H-reduce scoring
  - `g_aware_score_blend: 1.0`
  - G branch uses GBLUP
  - ridge alpha grid `0.001..100` with 21 values
  - `tie_tolerance: 0.001`
  - `fusion_tie_tolerance: 0.001`
  - `radius_candidates: [0, 1, 2, 3]`
  - `min_groups: 0`, `max_groups: 48`

## Environment

- Local work should use `conda activate PEG2P`.
- When adding smoke or regression checks, prefer small local outputs under `/private/tmp` or `CropHG/outputs`.
