from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / 'examples' / 'cropvig_minimal_input'
TMP_ROOT = Path(tempfile.gettempdir()) / 'cropvig_entrypoints_check'
TMP_ROOT.mkdir(parents=True, exist_ok=True)


def run_entry(script_name: str, out_name: str) -> Path:
    out_dir = TMP_ROOT / out_name
    if out_dir.exists():
        for child in out_dir.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src'
    subprocess.run(
        [sys.executable, f'scripts/{script_name}', '--input-dir', str(FIXTURE_DIR), '--output-dir', str(out_dir)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return out_dir


for script, out_name, expected in [
    ('cropvig_1.py', 'cropvig_1', 'cropvig_1_formal_analysis.md'),
    ('cropvig_2.py', 'cropvig_2', 'cropvig_2_formal_analysis.md'),
    ('cropvig_3.py', 'cropvig_3', 'cropvig_3_formal_analysis.md'),
]:
    out_dir = run_entry(script, out_name)
    target = out_dir / expected
    if not target.exists():
        raise SystemExit(f'missing expected output: {target}')
    print(target)
