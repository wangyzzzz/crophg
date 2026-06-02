from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / 'examples' / 'cropvig_minimal_input'
SMOKE_CONFIG = REPO_ROOT / 'configs' / '3_4a' / 'smoke_run_local.yaml'
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


for script, out_name, expected_variant in [
    ('cropvig_1.py', 'cropvig_1_run', 'G+FULLH'),
    ('cropvig_2.py', 'cropvig_2_run', 'H_ANCHOR_AUTO'),
    ('cropvig_3.py', 'cropvig_3_run', 'G+H_ANCHOR_AUTO'),
]:
    out_dir = TMP_ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src'
    subprocess.run(
        [
            sys.executable,
            f'scripts/{script}',
            'run',
            '--config',
            str(SMOKE_CONFIG),
            '--output-dir',
            str(out_dir),
            '--allow-overwrite',
            '--dry-run',
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    resolved = sorted(out_dir.glob('*_resolved_config_*.yaml'))[-1]
    cfg = yaml.safe_load(resolved.read_text(encoding='utf-8'))
    variants = cfg['experiment']['input_variants']
    if variants != [expected_variant]:
        raise SystemExit(f'{script} resolved variants {variants}, expected {[expected_variant]}')
    print(resolved)
