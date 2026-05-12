from __future__ import annotations

import argparse
from pathlib import Path

from crophg.internal.canonical_configs import (
    build_result32bc_section_cfg,
    read_base_config,
    write_resolved_config,
)
from crophg.internal.shared_anchor import write_shared_anchor_artifacts
from models.result33bc_light.parallel_launcher import run_result33bc_light_parallel_launcher
from models.result33bc_light.runner import run_result33bc_light

SECTION_CODE = "3.2B"
SECTION_SCOPE = "internal"
SECTION_SLUG = "single_vi_shift_under_h_only"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"crophg {SECTION_CODE} scaffold entry")
    parser.add_argument("--input-dir", type=str, default="", help="Optional existing result directory")
    parser.add_argument("--output-dir", type=str, default="", help="Optional output directory")
    parser.add_argument("--config", type=str, default="", help="Optional config file")
    parser.add_argument("--n-workers", type=int, default=1, help="Use parallel light runner when greater than 1")
    parser.add_argument("--print-spec", action="store_true", help="Print scaffold metadata and exit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.print_spec:
        print(f"section={SECTION_CODE}")
        print(f"scope={SECTION_SCOPE}")
        print(f"slug={SECTION_SLUG}")
        return 0
    if not args.config:
        raise ValueError(f"{SECTION_CODE} 需要 --config。")
    if not args.input_dir:
        raise ValueError(f"{SECTION_CODE} 需要 --input-dir，指向 3.2A / result33 输出目录。")

    repo_root = Path.cwd().resolve()
    config_path = Path(args.config).resolve()
    input_dir = Path(args.input_dir).resolve()

    delta_csv = input_dir / "single_anchor_delta.csv"
    if not delta_csv.exists():
        raise FileNotFoundError(f"缺少 3.2A 产物: {delta_csv}")

    tmp_dir = repo_root / "outputs" / "tmp_shared_anchor"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shared_csv = tmp_dir / "best_anchor_by_trait_shared_4scenarios_no_svr.csv"
    shared_md = tmp_dir / "best_anchor_by_trait_shared_4scenarios_no_svr.md"
    write_shared_anchor_artifacts(
        delta_csv=delta_csv,
        output_csv=shared_csv,
        summary_md=shared_md,
    )

    base_cfg = read_base_config(config_path)
    resolved_cfg = build_result32bc_section_cfg(
        base_cfg=base_cfg,
        best_anchor_csv=shared_csv,
        output_dir=args.output_dir,
    )
    resolved_path = write_resolved_config(cfg=resolved_cfg, section_code=SECTION_CODE, repo_root=repo_root)
    if int(args.n_workers) > 1:
        out_dir = run_result33bc_light_parallel_launcher(resolved_path, n_workers=int(args.n_workers))
    else:
        out_dir = run_result33bc_light(resolved_path)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
