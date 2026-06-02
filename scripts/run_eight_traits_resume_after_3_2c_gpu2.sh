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

test -f "$EXP_ROOT/3_2b/metrics_summary.csv"
test -f "$REPORT_ROOT/3_2b_analysis/result_3_2b_formal_analysis.md"
test -f "$REPORT_ROOT/3_2c_analysis/result_3_2c_formal_analysis.md"

run_step "${PYRUN[@]}" scripts/build_four_scenario_splits.py \
  --config configs/data_splits_4scenarios_ood_cell.yaml

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

echo "[$(date '+%F %T')] Eight-trait resume-after-3.2C pipeline completed."
