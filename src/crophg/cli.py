from __future__ import annotations

import argparse

from crophg.framework import get_section, list_sections


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="crophg command line interface")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List available sections")
    p_list.add_argument("--public-only", action="store_true", help="Only show public sections")

    p_desc = sub.add_parser("describe", help="Describe one section")
    p_desc.add_argument("code", help="Section code, e.g. 3.3A")
    return parser


def _cmd_list(public_only: bool) -> int:
    for spec in list_sections(public_only=public_only):
        scope = "public" if spec.public else "internal"
        print(f"{spec.code}\t{scope}\t{spec.title}\t{spec.old_result}")
    return 0


def _cmd_describe(code: str) -> int:
    spec = get_section(code)
    scope = "public" if spec.public else "internal"
    print(f"code: {spec.code}")
    print(f"paper_section: {spec.paper_section}")
    print(f"scope: {scope}")
    print(f"title: {spec.title}")
    print(f"old_result: {spec.old_result}")
    print(f"description: {spec.description}")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "list":
        return _cmd_list(public_only=bool(args.public_only))
    if args.command == "describe":
        return _cmd_describe(args.code)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
