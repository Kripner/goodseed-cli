"""Local HTTP server for serving experiment data to the frontend.

Reads SQLite run files from the projects directory and exposes a JSON API.
The frontend (served from goodseed.ai or 127.0.0.1) connects to this server.

Usage:
    goodseed serve [dir] [--port PORT]
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from goodseed.config import APP_URL
from goodseed.storage import (
    downsample_metrics,
    read_configs,
    read_metrics,
    read_metric_paths,
    read_run_summary,
    read_string_series,
    write_run_meta,
)

logger = logging.getLogger("goodseed.server")


def _sanitize_for_json(obj: object) -> object:
    """Convert NaN/Infinity floats to strings for JSON spec compliance.

    JSON does not support NaN/Infinity literals, so we encode them as
    the strings ``"NaN"``, ``"Infinity"``, and ``"-Infinity"`` to
    preserve the information for the frontend.
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    return obj


def _scan_runs(projects_dir: Path) -> list[dict[str, Any]]:
    """Scan the projects directory for run SQLite files.

    Supports nested project names (e.g. ``workspace/project``) by recursively
    searching for ``runs/*.sqlite`` at any depth under *projects_dir*.
    """
    runs = []
    if not projects_dir.exists():
        return runs

    for db_path in sorted(projects_dir.glob("**/runs/*.sqlite")):
        project_name = str(db_path.parent.parent.relative_to(projects_dir))
        try:
            meta, string_series_paths, metric_paths = read_run_summary(db_path)

            runs.append({
                "project": project_name,
                "run_id": meta.get("run_id", meta.get("run_name", db_path.stem)),
                "experiment_name": meta.get("name", meta.get("experiment_name")),
                "created_at": meta.get("created_at"),
                "closed_at": meta.get("closed_at"),
                "status": meta.get("status", "unknown"),
                "trashed": meta.get("trashed") == "true",
                "string_series_paths": string_series_paths,
                "metric_paths": metric_paths,
            })
        except Exception:
            logger.warning("Failed to read run database: %s", db_path, exc_info=True)
            continue

    runs.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return runs


def _scan_projects(projects_dir: Path) -> list[dict[str, Any]]:
    """Scan the projects directory and return project metadata.

    Lightweight: uses file mtime instead of opening SQLite databases.
    """
    projects: dict[str, dict[str, Any]] = {}
    if not projects_dir.exists():
        return []

    for db_path in sorted(projects_dir.glob("**/runs/*.sqlite")):
        name = str(db_path.parent.parent.relative_to(projects_dir))
        if name not in projects:
            projects[name] = {"name": name, "run_count": 0, "last_modified": None}
        projects[name]["run_count"] += 1
        mtime = datetime.fromtimestamp(db_path.stat().st_mtime, tz=timezone.utc).isoformat()
        if projects[name]["last_modified"] is None or mtime > projects[name]["last_modified"]:
            projects[name]["last_modified"] = mtime

    result = list(projects.values())
    result.sort(key=lambda p: p["last_modified"] or "", reverse=True)
    return result


def _resolve_run_db(projects_dir: Path, project: str, run_name: str) -> Path | None:
    """Resolve a run database file path, return None if not found."""
    db_path = projects_dir / project / "runs" / f"{run_name}.sqlite"
    if db_path.exists():
        return db_path
    return None


# Route patterns
_ROUTE_PROJECTS = re.compile(r"^/api/projects$")
_ROUTE_RUNS = re.compile(r"^/api/runs$")
_ROUTE_CONFIGS = re.compile(r"^/api/runs/(.+)/([^/]+)/configs$")
_ROUTE_METRICS = re.compile(r"^/api/runs/(.+)/([^/]+)/metrics$")
_ROUTE_METRIC_PATHS = re.compile(r"^/api/runs/(.+)/([^/]+)/metric-paths$")
_ROUTE_STRING_SERIES = re.compile(r"^/api/runs/(.+)/([^/]+)/string_series$")
_ROUTE_TRASH_RESTORE = re.compile(r"^/api/runs/(.+)/([^/]+)/trash/restore$")
_ROUTE_TRASH = re.compile(r"^/api/runs/(.+)/([^/]+)/trash$")


class _RequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Goodseed API."""

    # Set by the server
    projects_dir: Path

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        """Route GET requests to the appropriate handler."""
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)

        try:
            # GET /api/projects
            m = _ROUTE_PROJECTS.match(path)
            if m:
                projects = _scan_projects(self.projects_dir)
                self._send_json({"projects": projects})
                return

            # GET /api/runs
            m = _ROUTE_RUNS.match(path)
            if m:
                runs = _scan_runs(self.projects_dir)
                project_filter = query.get("project", [None])[0]
                if project_filter:
                    runs = [r for r in runs if r["project"] == project_filter]
                self._send_json({"runs": runs})
                return

            # GET /api/runs/<project>/<run_name>/configs
            m = _ROUTE_CONFIGS.match(path)
            if m:
                project, run_name = m.group(1), m.group(2)
                db_path = _resolve_run_db(self.projects_dir, project, run_name)
                if not db_path:
                    self._send_error(404, f"Run not found: {project}/{run_name}")
                    return
                configs = read_configs(db_path)
                self._send_json({"configs": configs})
                return

            # GET /api/runs/<project>/<run_name>/metrics
            m = _ROUTE_METRICS.match(path)
            if m:
                project, run_name = m.group(1), m.group(2)
                db_path = _resolve_run_db(self.projects_dir, project, run_name)
                if not db_path:
                    self._send_error(404, f"Run not found: {project}/{run_name}")
                    return
                metric_path = query.get("path", [None])[0]
                point_count_str = query.get("pointCount", [None])[0]

                if point_count_str:
                    if not metric_path:
                        self._send_error(400, "pointCount requires a path parameter")
                        return
                    try:
                        point_count = int(point_count_str)
                    except ValueError:
                        self._send_error(400, "pointCount must be an integer")
                        return
                    first_idx_str = query.get("firstPointIndex", [None])[0]
                    last_idx_str = query.get("lastPointIndex", [None])[0]
                    try:
                        first_idx = int(first_idx_str) if first_idx_str else None
                        last_idx = int(last_idx_str) if last_idx_str else None
                    except ValueError:
                        self._send_error(400, "firstPointIndex and lastPointIndex must be integers")
                        return
                    result = downsample_metrics(
                        db_path, metric_path, point_count,
                        first_point_index=first_idx,
                        last_point_index=last_idx,
                    )
                    self._send_json(result)
                else:
                    metrics = read_metrics(db_path, metric_path)
                    self._send_json({"metrics": metrics})
                return

            # GET /api/runs/<project>/<run_name>/metric-paths
            m = _ROUTE_METRIC_PATHS.match(path)
            if m:
                project, run_name = m.group(1), m.group(2)
                db_path = _resolve_run_db(self.projects_dir, project, run_name)
                if not db_path:
                    self._send_error(404, f"Run not found: {project}/{run_name}")
                    return
                paths = read_metric_paths(db_path)
                self._send_json({"paths": paths})
                return

            # GET /api/runs/<project>/<run_name>/string_series
            m = _ROUTE_STRING_SERIES.match(path)
            if m:
                project, run_name = m.group(1), m.group(2)
                db_path = _resolve_run_db(self.projects_dir, project, run_name)
                if not db_path:
                    self._send_error(404, f"Run not found: {project}/{run_name}")
                    return
                series_path = query.get("path", [None])[0]
                limit_str = query.get("limit", [None])[0]
                offset_str = query.get("offset", ["0"])[0]
                tail_str = query.get("tail", [None])[0]
                try:
                    limit = int(limit_str) if limit_str else None
                    offset = int(offset_str) if offset_str else 0
                    tail = int(tail_str) if tail_str else None
                except ValueError:
                    self._send_error(400, "limit, offset, and tail must be integers")
                    return
                result = read_string_series(db_path, series_path, limit=limit, offset=offset, tail=tail)
                self._send_json({
                    "string_series": result["points"],
                    "total": result["total"],
                })
                return

            self._send_error(404, "Not found")

        except Exception as e:
            self._send_error(500, str(e))

    def do_POST(self) -> None:
        """Route POST requests."""
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        try:
            # Match /restore before /trash (longer prefix first)
            m = _ROUTE_TRASH_RESTORE.match(path)
            if m:
                project, run_name = m.group(1), m.group(2)
                db_path = _resolve_run_db(self.projects_dir, project, run_name)
                if not db_path:
                    self._send_error(404, f"Run not found: {project}/{run_name}")
                    return
                write_run_meta(db_path, "trashed", None)
                self._send_json({"ok": True})
                return

            m = _ROUTE_TRASH.match(path)
            if m:
                project, run_name = m.group(1), m.group(2)
                db_path = _resolve_run_db(self.projects_dir, project, run_name)
                if not db_path:
                    self._send_error(404, f"Run not found: {project}/{run_name}")
                    return
                write_run_meta(db_path, "trashed", "true")
                self._send_json({"ok": True})
                return

            self._send_error(404, "Not found")

        except Exception as e:
            self._send_error(500, str(e))

    def _send_json(self, data: Any) -> None:
        """Send a JSON response with CORS headers."""
        body = json.dumps(_sanitize_for_json(data)).encode("utf-8")
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str) -> None:
        """Send a JSON error response."""
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        """Add CORS headers to the response."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default request logging."""
        pass


def run_server(projects_dir: Path, port: int = 8765, verbose: bool = False) -> None:
    """Start the local HTTP server.

    Args:
        projects_dir: Directory containing project subdirectories with .sqlite files.
        port: Port to listen on (default: 8765).
        verbose: Print extra startup information.
    """
    _RequestHandler.projects_dir = projects_dir

    server = ThreadingHTTPServer(("127.0.0.1", port), _RequestHandler)

    if verbose:
        print(f"Goodseed server running at http://127.0.0.1:{port}")
        print(f"Data directory: {projects_dir}")
    app_base_url = APP_URL.rstrip("/")
    print(f"View your runs at {app_base_url}/local?port={port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()
