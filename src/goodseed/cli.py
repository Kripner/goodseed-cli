"""Command-line interface for Goodseed.

Usage:
    goodseed [dir]               - Start the local server (alias for 'goodseed serve')
    goodseed serve [dir]         - Start the local server
    goodseed list [dir]          - List runs
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from goodseed.config import get_projects_dir
from goodseed.server import _scan_runs, run_server


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the local HTTP server."""
    projects_dir = Path(args.dir) if args.dir else get_projects_dir()
    port = args.port

    if not projects_dir.exists():
        projects_dir.mkdir(parents=True, exist_ok=True)

    run_server(projects_dir, port=port)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List runs in the projects directory."""
    projects_dir = Path(args.dir) if args.dir else get_projects_dir()

    if not projects_dir.exists():
        print(f"Projects directory does not exist: {projects_dir}")
        return 0

    runs = _scan_runs(projects_dir)

    if not runs:
        print("No runs found.")
        return 0

    # Group by project
    by_project: dict[str, list] = {}
    for run in runs:
        by_project.setdefault(run["project"], []).append(run)

    for project_name in sorted(by_project):
        print(f"{project_name}/")
        for run in by_project[project_name]:
            status = run.get("status", "unknown")
            run_id = run.get("run_id", "?")
            experiment_name = run.get("experiment_name")
            created_at = run.get("created_at") or "-"

            print(f"  [{status}] {run_id}")
            if experiment_name:
                print(f"      name: {experiment_name}")
            print(f"      created: {created_at[:19] if len(created_at) > 19 else created_at}")
        print()

    print(f"Total: {len(runs)} run(s)")
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="goodseed",
        description="Goodseed ML experiment tracker",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the local server")
    serve_parser.add_argument(
        "dir", nargs="?",
        help="Directory containing run databases (default: ~/.goodseed/projects)",
    )
    serve_parser.add_argument(
        "--port", type=int, default=8765,
        help="Port to listen on (default: 8765)",
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List runs")
    list_parser.add_argument(
        "dir", nargs="?",
        help="Directory containing run databases (default: ~/.goodseed/projects)",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # Default: serve
        args.dir = None
        args.port = 8765
        return cmd_serve(args)

    if args.command == "serve":
        return cmd_serve(args)
    elif args.command == "list":
        return cmd_list(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
