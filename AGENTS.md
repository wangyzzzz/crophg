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
- Pipeline config for `3.3B`: `configs/pipeline_two_traits_gpu2/3_3b.yaml`.
- Formal current two-trait `3.3B` experiment output: `outputs/experiments/two_traits_full_pipeline_gpu2/3_3b_midpoint_r0_3_k0_48`.
- `3.4A` is the growth-prefix experiment: anchor orders `1..20`, where each prefix cumulatively uses no-bin groups from the first bin through the current bin.
- Pipeline config for `3.4A`: `configs/pipeline_two_traits_gpu2/3_4a.yaml`.
- Formal current two-trait `3.4A` experiment output: `outputs/experiments/two_traits_full_pipeline_gpu2/3_4a_midpoint_r0_3_k0_48`.
- Current `3.3B/3.4A` H-reduce uses midpoint no-bin anchors:
  - split the full H timeline into 20 continuous bins
  - use each bin's midpoint time key as the target anchor point
  - do not flatten the full bin as one group for the formal current run
- Current `3.3B/3.4A` AUTO radius search uses `anchor_window_search.radius_candidates: [0, 1, 2, 3]`.
- Current `3.3B/3.4A` group-count search uses `anchor_local_pruning.min_groups: 0` and `anchor_local_pruning.max_groups: 48`.
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

## Predictor Set

- Formal executable predictors are `ridge`, `lasso`, `elasticnet`, `lightgbm`, and `random_forest`.
- Do not reintroduce `svr`; historical file names containing `no_svr` can remain as provenance.

## Environment

- Local work should use `conda activate PEG2P`.
- When adding smoke or regression checks, prefer small local outputs under `/private/tmp` or `CropHG/outputs`.
