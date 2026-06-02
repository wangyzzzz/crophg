from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cropvig_mpl_cache"))

from models.common.io_utils import ensure_dir, read_yaml, write_yaml
from models.result34.parallel_launcher import run_result34_parallel_launcher
from models.result34.runner import run_result34


@dataclass(frozen=True)
class CropVIGModelSpec:
    command_name: str
    model_name: str
    input_variant: str
    description: str


AnalysisRunner = Callable[[argparse.Namespace], int]
ParserFactory = Callable[[], argparse.ArgumentParser]


def _with_model_suffix(path_text: str, suffix: str) -> str:
    path = Path(path_text)
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}").as_posix()


def _resolve_run_config(
    *,
    spec: CropVIGModelSpec,
    config_path: Path,
    output_dir: str | None,
    allow_overwrite: bool,
    latest_pointer_file: str | None,
) -> tuple[dict, Path]:
    cfg = copy.deepcopy(read_yaml(config_path))
    exp_cfg = cfg.setdefault("experiment", {})
    out_cfg = cfg.setdefault("output", {})

    exp_cfg["input_variants"] = [spec.input_variant]
    exp_cfg["cropvig_model"] = spec.model_name
    exp_cfg["cropvig_input_variant"] = spec.input_variant

    if output_dir:
        out_cfg["output_dir_base"] = str(output_dir)
        out_cfg["append_timestamp"] = False
        out_cfg["allow_overwrite"] = bool(allow_overwrite)
        if latest_pointer_file is None:
            out_cfg["latest_pointer_file"] = None
    else:
        base = str(out_cfg.get("output_dir_base", f"outputs/experiments/{spec.command_name}"))
        out_cfg["output_dir_base"] = f"{base}_{spec.command_name}"
        if "allow_overwrite" not in out_cfg:
            out_cfg["allow_overwrite"] = False

    if latest_pointer_file is not None:
        out_cfg["latest_pointer_file"] = latest_pointer_file
    elif out_cfg.get("latest_pointer_file"):
        out_cfg["latest_pointer_file"] = _with_model_suffix(str(out_cfg["latest_pointer_file"]), spec.command_name)

    resolved_root = Path(out_cfg["output_dir_base"])
    if bool(out_cfg.get("append_timestamp", False)):
        resolved_root = resolved_root.parent / f"{resolved_root.name}_resolved_configs"
    ensure_dir(resolved_root)
    resolved_config_path = resolved_root / f"{spec.command_name}_resolved_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    write_yaml(cfg, resolved_config_path)
    return cfg, resolved_config_path


def build_run_parser(spec: CropVIGModelSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run {spec.model_name}: {spec.description}")
    parser.add_argument("--config", required=True, help="YAML config for a CropVIG/Result34 model run")
    parser.add_argument("--output-dir", default="", help="Optional output directory for this model run")
    parser.add_argument("--n-workers", type=int, default=1, help="Parallel workers. Use 1 for a single-process run")
    parser.add_argument("--allow-overwrite", action="store_true", help="Allow overwriting --output-dir when it exists")
    parser.add_argument("--latest-pointer-file", default=None, help="Optional path that stores the latest output directory")
    parser.add_argument("--dry-run", action="store_true", help="Only write the resolved model config, do not train")
    parser.add_argument("--print-spec", action="store_true", help="Print model metadata and exit")
    return parser


def run_cropvig_model(spec: CropVIGModelSpec, args: argparse.Namespace) -> int:
    if args.print_spec:
        print(f"model={spec.model_name}")
        print(f"command={spec.command_name}")
        print(f"input_variant={spec.input_variant}")
        print(f"description={spec.description}")
        return 0

    _, resolved_config_path = _resolve_run_config(
        spec=spec,
        config_path=Path(args.config).resolve(),
        output_dir=args.output_dir or None,
        allow_overwrite=bool(args.allow_overwrite),
        latest_pointer_file=args.latest_pointer_file,
    )
    print(f"Resolved {spec.model_name} config: {resolved_config_path}", flush=True)
    if args.dry_run:
        return 0

    if int(args.n_workers) > 1:
        run_result34_parallel_launcher(resolved_config_path, n_workers=int(args.n_workers))
    else:
        run_result34(resolved_config_path)
    return 0


def dispatch_cropvig_entrypoint(
    *,
    spec: CropVIGModelSpec,
    argv: Sequence[str] | None,
    analysis_parser_factory: ParserFactory,
    analysis_runner: AnalysisRunner,
) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    mode = ""
    if raw_args and raw_args[0] in {"run", "analyze"}:
        mode = raw_args.pop(0)
    elif "--config" in raw_args:
        mode = "run"
    else:
        mode = "analyze"

    if mode == "run":
        return run_cropvig_model(spec, build_run_parser(spec).parse_args(raw_args))
    if mode == "analyze":
        return analysis_runner(analysis_parser_factory().parse_args(raw_args))
    raise ValueError(f"Unsupported CropVIG mode: {mode}")
