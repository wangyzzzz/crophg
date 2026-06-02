#!/usr/bin/env bash
set -euo pipefail

N_WORKERS="${1:-60}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYRUN=(/usr/local/bin/micromamba run -n PEG2P env PYTHONPATH=src python)
EXP_ROOT="outputs/experiments/eight_traits_full_pipeline_gpu2"
REPORT_ROOT="outputs/reports/eight_traits_full_pipeline_gpu2"

run_step() {
  echo "[$(date '+%F %T')] $*"
  "$@"
}

mkdir -p "$EXP_ROOT"
mkdir -p "$REPORT_ROOT"

run_step "${PYRUN[@]}" scripts/build_four_scenario_splits.py \
  --config configs/data_splits_4scenarios_ood_cell.yaml

run_step "${PYRUN[@]}" -m models.result31.parallel_launcher \
  --config configs/pipeline_eight_traits_gpu2/3_1a.yaml \
  --n-workers "$N_WORKERS"

run_step "${PYRUN[@]}" scripts/result_3_1a_analysis.py \
  --input-dir "$EXP_ROOT/3_1a" \
  --output-dir "$REPORT_ROOT/3_1a_analysis"

run_step "${PYRUN[@]}" -m models.result31.parallel_launcher \
  --config configs/pipeline_eight_traits_gpu2/3_1b.yaml \
  --n-workers "$N_WORKERS"

run_step "${PYRUN[@]}" scripts/result_3_1b_analysis.py \
  --input-dir "$EXP_ROOT/3_1b" \
  --output-dir "$REPORT_ROOT/3_1b_analysis"

run_step "${PYRUN[@]}" -m models.result33.parallel_launcher \
  --config configs/pipeline_eight_traits_gpu2/3_2a.yaml \
  --n-workers "$N_WORKERS"

run_step "${PYRUN[@]}" scripts/result_3_2a_analysis.py \
  --input-dir "$EXP_ROOT/3_2a" \
  --output-dir "$REPORT_ROOT/3_2a_analysis"

run_step "${PYRUN[@]}" scripts/result_3_2b.py \
  --config configs/pipeline_eight_traits_gpu2/3_2b.yaml \
  --input-dir "$EXP_ROOT/3_2a" \
  --output-dir "$EXP_ROOT/3_2b" \
  --n-workers "$N_WORKERS"

run_step "${PYRUN[@]}" scripts/result_3_2b_analysis.py \
  --input-dir "$EXP_ROOT/3_2b" \
  --anchor-source-dir "$EXP_ROOT/3_2a" \
  --output-dir "$REPORT_ROOT/3_2b_analysis"

run_step "${PYRUN[@]}" scripts/result_3_2c_analysis.py \
  --input-dir "$EXP_ROOT/3_2b" \
  --anchor-source-dir "$EXP_ROOT/3_2a" \
  --output-dir "$REPORT_ROOT/3_2c_analysis"

run_step "${PYRUN[@]}" scripts/run_result3_4_parallel_launcher.py \
  --config configs/pipeline_eight_traits_gpu2/3_3b_tie001.yaml \
  --n-workers "$N_WORKERS"

run_step "${PYRUN[@]}" scripts/cropvig_1.py \
  --input-dir "$EXP_ROOT/3_3b_tie001" \
  --output-dir "$REPORT_ROOT/3_3a_analysis"

run_step "${PYRUN[@]}" scripts/cropvig_2.py \
  --input-dir "$EXP_ROOT/3_3b_tie001" \
  --output-dir "$REPORT_ROOT/3_3b_analysis"

run_step "${PYRUN[@]}" scripts/run_result3_4_parallel_launcher.py \
  --config configs/pipeline_eight_traits_gpu2/3_4a_tie001.yaml \
  --n-workers "$N_WORKERS"

run_step "${PYRUN[@]}" scripts/cropvig_3.py \
  --input-dir "$EXP_ROOT/3_4a_tie001" \
  --output-dir "$REPORT_ROOT/3_4a_analysis"

run_step "${PYRUN[@]}" scripts/result_3_4b.py \
  --input-dir "$EXP_ROOT/3_4a_tie001" \
  --output-dir "$REPORT_ROOT/3_4b_analysis"

echo "[$(date '+%F %T')] Eight-trait full pipeline completed."
