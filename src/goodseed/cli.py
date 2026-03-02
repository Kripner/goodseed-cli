"""Command-line interface for Goodseed.

Usage:
    goodseed [dir]               - Start the local server (alias for 'goodseed serve')
    goodseed serve [dir]         - Start the local server
    goodseed list [dir]          - List projects (or runs with --project)
    goodseed upload -p <project> [--run-id RUN_ID]  - Upload unuploaded run data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from goodseed.config import get_api_key, get_projects_dir, get_run_db_path
from goodseed.server import _scan_projects, _scan_runs, run_server
from goodseed.sync import upload_run


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the local HTTP server."""
    projects_dir = Path(args.dir) if args.dir else get_projects_dir()
    run_server(projects_dir, port=args.port, verbose=args.verbose)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List projects, or runs within a specific project."""
    projects_dir = Path(args.dir) if args.dir else get_projects_dir()

    if not projects_dir.exists():
        print(f"Projects directory does not exist: {projects_dir}")
        return 0

    if args.project:
        # List runs for a specific project
        runs = _scan_runs(projects_dir)
        runs = [r for r in runs if r["project"] == args.project]

        if not runs:
            print(f"No runs found in project '{args.project}'.")
            return 0

        for run in runs:
            status = run.get("status", "unknown")
            run_id = run.get("run_id", "?")
            run_name = run.get("experiment_name")
            created_at = run.get("created_at") or "-"

            print(f"  [{status}] {run_id}")
            if run_name:
                print(f"      name: {run_name}")
            print(f"      created: {created_at[:19] if len(created_at) > 19 else created_at}")

        print(f"\n{len(runs)} run(s) in {args.project}")
    else:
        # List projects
        projects = _scan_projects(projects_dir)

        if not projects:
            print("No projects found.")
            return 0

        for proj in projects:
            count = proj["run_count"]
            modified = proj.get("last_modified") or "-"
            if len(modified) > 19:
                modified = modified[:19]
            print(f"  {proj['name']}  ({count} run{'s' if count != 1 else ''}, last modified: {modified})")

        print(f"\n{len(projects)} project(s)")

    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    """Upload unuploaded data from a run's local database."""
    project = args.project
    api_key = args.api_key or get_api_key()
    if not api_key:
        print("Error: No API key. Set GOODSEED_API_KEY or pass --api-key.")
        return 1

    db_paths: list[Path]
    if args.run_id:
        db_path = get_run_db_path(project, args.run_id)
        if not db_path.exists():
            print(f"Error: Database not found at {db_path}")
            return 1
        db_paths = [db_path]
    else:
        runs_dir = get_projects_dir() / project / "runs"
        db_paths = sorted(runs_dir.glob("*.sqlite"))
        if not db_paths:
            print(f"No runs found in project '{project}'.")
            return 0

    total = 0
    for db_path in db_paths:
        print(f"Uploading run '{db_path.stem}' from {db_path} ...")
        try:
            total += upload_run(db_path, api_key)
        except RuntimeError as e:
            print(f"Error: {e}")
            return 1
    print(f"Done. Uploaded {total} item(s).")
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
    serve_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print extra startup information",
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List projects (or runs with --project)")
    list_parser.add_argument(
        "dir", nargs="?",
        help="Directory containing run databases (default: ~/.goodseed/projects)",
    )
    list_parser.add_argument(
        "-p", "--project",
        help="List runs within a specific project (e.g. workspace/project-name)",
    )

    # upload command
    upload_parser = subparsers.add_parser(
        "upload", help="Upload unuploaded data from a local run",
    )
    upload_parser.add_argument(
        "-r", "--run-id",
        help="The run ID to upload (omit to upload all runs in project)",
    )
    upload_parser.add_argument(
        "-p", "--project", required=True,
        help="Project name (workspace/project)",
    )
    upload_parser.add_argument(
        "--api-key",
        help="API key (default: GOODSEED_API_KEY env var)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # Default: serve
        args.dir = None
        args.port = 8765
        args.verbose = False
        return cmd_serve(args)

    if args.command == "serve":
        return cmd_serve(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "upload":
        return cmd_upload(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
