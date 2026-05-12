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

echo "[start] $(date '+%F %T')"
echo "[repo] $REPO_ROOT"
echo "[workers] $N_WORKERS"

STAMP="$(date '+%Y%m%d_%H%M%S')"
BACKUP_ROOT="outputs/experiments/two_traits_full_pipeline_gpu2/_rerun_backup_${STAMP}"
REPORT_BACKUP_ROOT="outputs/reports/two_traits_full_pipeline_gpu2/_rerun_backup_${STAMP}"
mkdir -p "$BACKUP_ROOT" "$REPORT_BACKUP_ROOT"
for name in 3_3b 3_4a; do
  if [[ -e "outputs/experiments/two_traits_full_pipeline_gpu2/${name}" ]]; then
    echo "[backup] outputs/experiments/two_traits_full_pipeline_gpu2/${name} -> ${BACKUP_ROOT}/${name}"
    mv "outputs/experiments/two_traits_full_pipeline_gpu2/${name}" "${BACKUP_ROOT}/${name}"
  fi
  if [[ -e "outputs/reports/two_traits_full_pipeline_gpu2/${name}_analysis" ]]; then
    echo "[backup] outputs/reports/two_traits_full_pipeline_gpu2/${name}_analysis -> ${REPORT_BACKUP_ROOT}/${name}_analysis"
    mv "outputs/reports/two_traits_full_pipeline_gpu2/${name}_analysis" "${REPORT_BACKUP_ROOT}/${name}_analysis"
  fi
done
if [[ -e "outputs/reports/two_traits_full_pipeline_gpu2/3_3a_analysis" ]]; then
  echo "[backup] outputs/reports/two_traits_full_pipeline_gpu2/3_3a_analysis -> ${REPORT_BACKUP_ROOT}/3_3a_analysis"
  mv "outputs/reports/two_traits_full_pipeline_gpu2/3_3a_analysis" "${REPORT_BACKUP_ROOT}/3_3a_analysis"
fi

run_step "${PYRUN[@]}" -m compileall -q src tests/test_result34_fusion_policy.py
if "${PYRUN[@]}" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("pytest") is not None else 1)
PY
then
  run_step "${PYRUN[@]}" -m pytest -q tests/test_result34_fusion_policy.py
else
  echo "[skip] pytest is not installed in PEG2P; compile/config checks will be used for this remote rerun."
fi

run_step "${PYRUN[@]}" - <<'PY'
from pathlib import Path

for path in [
    Path("configs/pipeline_two_traits_gpu2/3_3b.yaml"),
    Path("configs/pipeline_two_traits_gpu2/3_4a.yaml"),
]:
    text = path.read_text(encoding="utf-8")
    print(
        f"[config] {path}: "
        f"g_aware_1={'g_aware_score_blend: 1.0' in text}, "
        f"old_alpha={'alpha_candidates' in text or 'min_alpha_gain' in text}"
    )
PY

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

echo "[done] $(date '+%F %T')"
