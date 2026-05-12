#!/usr/bin/env bash
set -euo pipefail

N_WORKERS="${1:-60}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYRUN=(/usr/local/bin/micromamba run -n PEG2P env PYTHONPATH=src python)

run_step() {
  echo "[$(date '+%F %T')] $*"
  "$@"
}

mkdir -p outputs/experiments/two_traits_full_pipeline_gpu2
mkdir -p outputs/reports/two_traits_full_pipeline_gpu2

run_step "${PYRUN[@]}" -m models.result31.parallel_launcher \
  --config configs/pipeline_two_traits_gpu2/3_1a.yaml \
  --n-workers "$N_WORKERS"

run_step "${PYRUN[@]}" scripts/result_3_1a_analysis.py \
  --input-dir outputs/experiments/two_traits_full_pipeline_gpu2/3_1a \
  --output-dir outputs/reports/two_traits_full_pipeline_gpu2/3_1a_analysis

run_step "${PYRUN[@]}" -m models.result31.parallel_launcher \
  --config configs/pipeline_two_traits_gpu2/3_1b.yaml \
  --n-workers "$N_WORKERS"

run_step "${PYRUN[@]}" scripts/result_3_1b_analysis.py \
  --input-dir outputs/experiments/two_traits_full_pipeline_gpu2/3_1b \
  --output-dir outputs/reports/two_traits_full_pipeline_gpu2/3_1b_analysis

run_step "${PYRUN[@]}" -m models.result33.parallel_launcher \
  --config configs/pipeline_two_traits_gpu2/3_2a.yaml \
  --n-workers "$N_WORKERS"

run_step "${PYRUN[@]}" scripts/result_3_2a_analysis.py \
  --input-dir outputs/experiments/two_traits_full_pipeline_gpu2/3_2a \
  --output-dir outputs/reports/two_traits_full_pipeline_gpu2/3_2a_analysis

run_step "${PYRUN[@]}" scripts/result_3_2b.py \
  --config configs/3_2b/formal_run_two_traits_gpu2.yaml \
  --input-dir outputs/experiments/two_traits_full_pipeline_gpu2/3_2a \
  --output-dir outputs/experiments/two_traits_full_pipeline_gpu2/3_2b

run_step "${PYRUN[@]}" scripts/result_3_2b_analysis.py \
  --input-dir outputs/experiments/two_traits_full_pipeline_gpu2/3_2b \
  --anchor-source-dir outputs/experiments/two_traits_full_pipeline_gpu2/3_2a \
  --output-dir outputs/reports/two_traits_full_pipeline_gpu2/3_2b_analysis

# 3.2B and 3.2C share the same shared-anchor single-VI run.
# The 3.2B experiment writes both H_SINGLE_VI and GH_SINGLE_VI rows; 3.2C
# is a distinct formal analysis of GH_SINGLE_VI relative to G, not a duplicate run.
run_step "${PYRUN[@]}" scripts/result_3_2c_analysis.py \
  --input-dir outputs/experiments/two_traits_full_pipeline_gpu2/3_2b \
  --anchor-source-dir outputs/experiments/two_traits_full_pipeline_gpu2/3_2a \
  --output-dir outputs/reports/two_traits_full_pipeline_gpu2/3_2c_analysis

run_step "${PYRUN[@]}" scripts/run_result3_4_parallel_launcher.py \
  --config configs/pipeline_two_traits_gpu2/3_3b.yaml \
  --n-workers "$N_WORKERS"

run_step "${PYRUN[@]}" scripts/result_3_3a.py \
  --input-dir outputs/experiments/two_traits_full_pipeline_gpu2/3_3b \
  --output-dir outputs/reports/two_traits_full_pipeline_gpu2/3_3a_analysis

run_step "${PYRUN[@]}" scripts/result_3_3b.py \
  --input-dir outputs/experiments/two_traits_full_pipeline_gpu2/3_3b \
  --output-dir outputs/reports/two_traits_full_pipeline_gpu2/3_3b_analysis

run_step "${PYRUN[@]}" scripts/run_result3_4_parallel_launcher.py \
  --config configs/pipeline_two_traits_gpu2/3_4a.yaml \
  --n-workers "$N_WORKERS"

run_step "${PYRUN[@]}" scripts/result_3_4a.py \
  --input-dir outputs/experiments/two_traits_full_pipeline_gpu2/3_4a \
  --output-dir outputs/reports/two_traits_full_pipeline_gpu2/3_4a_analysis

run_step "${PYRUN[@]}" scripts/result_3_4b.py \
  --input-dir outputs/experiments/two_traits_full_pipeline_gpu2/3_4a \
  --output-dir outputs/reports/two_traits_full_pipeline_gpu2/3_4b_analysis
