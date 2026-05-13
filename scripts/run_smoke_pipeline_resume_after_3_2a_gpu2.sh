#!/usr/bin/env bash
set -euo pipefail

N_WORKERS="${1:-20}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYRUN=(/usr/local/bin/micromamba run -n PEG2P env PYTHONPATH=src python)
EXP_ROOT="outputs/experiments/smoke_pipeline_gpu2"
REPORT_ROOT="outputs/reports/smoke_pipeline_gpu2"

run_step() {
  echo "[$(date '+%F %T')] $*"
  "$@"
}

mkdir -p "$EXP_ROOT"
mkdir -p "$REPORT_ROOT"

test -f "$EXP_ROOT/3_2a/metrics_summary.csv"
test -f "$EXP_ROOT/3_2a/single_anchor_delta.csv"

run_step "${PYRUN[@]}" scripts/result_3_2a_analysis.py \
  --input-dir "$EXP_ROOT/3_2a" \
  --output-dir "$REPORT_ROOT/3_2a_analysis"

run_step "${PYRUN[@]}" scripts/result_3_2b.py \
  --config configs/smoke_pipeline_gpu2/3_2b_smoke.yaml \
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
  --config configs/smoke_pipeline_gpu2/3_3b_smoke.yaml \
  --n-workers "$N_WORKERS"

run_step "${PYRUN[@]}" scripts/result_3_3a.py \
  --input-dir "$EXP_ROOT/3_3b" \
  --output-dir "$REPORT_ROOT/3_3a_analysis"

run_step "${PYRUN[@]}" scripts/result_3_3b.py \
  --input-dir "$EXP_ROOT/3_3b" \
  --output-dir "$REPORT_ROOT/3_3b_analysis"

run_step "${PYRUN[@]}" scripts/run_result3_4_parallel_launcher.py \
  --config configs/smoke_pipeline_gpu2/3_4a_smoke.yaml \
  --n-workers "$N_WORKERS"

run_step "${PYRUN[@]}" scripts/result_3_4a.py \
  --input-dir "$EXP_ROOT/3_4a" \
  --output-dir "$REPORT_ROOT/3_4a_analysis"

run_step "${PYRUN[@]}" scripts/result_3_4b.py \
  --input-dir "$EXP_ROOT/3_4a" \
  --output-dir "$REPORT_ROOT/3_4b_analysis"

echo "[$(date '+%F %T')] Smoke resume pipeline completed."
