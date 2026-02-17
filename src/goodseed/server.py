"""Local HTTP server for serving experiment data to the frontend.

Reads SQLite run files from the projects directory and exposes a JSON API.
The frontend (served from goodseed.ai or localhost) connects to this server.

Usage:
    goodseed serve [dir] [--port PORT]
"""

import json
import re
import sqlite3
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from goodseed.utils import deserialize_value


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite database in read-only mode."""
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout=3000")
    conn.row_factory = sqlite3.Row
    return conn


def _read_meta(conn: sqlite3.Connection) -> Dict[str, str]:
    """Read all run_meta key-value pairs."""
    rows = conn.execute("SELECT key, value FROM run_meta").fetchall()
    return {row["key"]: row["value"] for row in rows}


def _scan_runs(projects_dir: Path) -> List[Dict[str, Any]]:
    """Scan the projects directory for run SQLite files."""
    runs = []
    if not projects_dir.exists():
        return runs

    for project_dir in sorted(projects_dir.iterdir()):
        if not project_dir.is_dir():
            continue

        project_name = project_dir.name
        runs_dir = project_dir / "runs"
        if not runs_dir.is_dir():
            continue
        for db_path in sorted(runs_dir.glob("*.sqlite")):
            try:
                conn = _open_readonly(db_path)
                try:
                    meta = _read_meta(conn)
                finally:
                    conn.close()

                runs.append({
                    "project": project_name,
                    "run_id": meta.get("run_name", db_path.stem),
                    "experiment_name": meta.get("experiment_name"),
                    "created_at": meta.get("created_at"),
                    "closed_at": meta.get("closed_at"),
                    "status": meta.get("status", "unknown"),
                })
            except Exception:
                continue

    runs.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return runs


def _get_configs(db_path: Path) -> Dict[str, Any]:
    """Read configs from a run database and deserialize values."""
    conn = _open_readonly(db_path)
    try:
        rows = conn.execute("SELECT path, type_tag, value FROM configs").fetchall()
    finally:
        conn.close()

    configs = {}
    for row in rows:
        configs[row["path"]] = deserialize_value(row["type_tag"], row["value"])
    return configs


def _ts_to_iso(ts: int) -> str:
    """Convert a unix timestamp to ISO 8601 string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _get_metrics(db_path: Path, metric_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read metric points from a run database."""
    conn = _open_readonly(db_path)
    try:
        if metric_path:
            rows = conn.execute(
                """SELECT s.path, p.step, p.y, p.ts
                   FROM metric_points p
                   JOIN metric_series s ON p.series_id = s.id
                   WHERE s.path = ?
                   ORDER BY p.step""",
                (metric_path,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT s.path, p.step, p.y, p.ts
                   FROM metric_points p
                   JOIN metric_series s ON p.series_id = s.id
                   ORDER BY s.path, p.step"""
            ).fetchall()
    finally:
        conn.close()

    return [
        {
            "path": row["path"],
            "step": row["step"],
            "value": row["y"],
            "is_preview": False,
            "preview_completion": None,
            "logged_at": _ts_to_iso(row["ts"]),
        }
        for row in rows
    ]


def _get_metric_paths(db_path: Path) -> List[str]:
    """Read distinct metric paths from a run database."""
    conn = _open_readonly(db_path)
    try:
        rows = conn.execute(
            """SELECT DISTINCT s.path FROM metric_series s
               JOIN metric_points p ON s.id = p.series_id
               ORDER BY s.path"""
        ).fetchall()
    finally:
        conn.close()
    return [row["path"] for row in rows]


def _resolve_run_db(projects_dir: Path, project: str, run_name: str) -> Optional[Path]:
    """Resolve a run database file path, return None if not found."""
    db_path = projects_dir / project / "runs" / f"{run_name}.sqlite"
    if db_path.exists():
        return db_path
    return None


# Route patterns
_ROUTE_RUNS = re.compile(r"^/api/runs$")
_ROUTE_CONFIGS = re.compile(r"^/api/runs/([^/]+)/([^/]+)/configs$")
_ROUTE_METRICS = re.compile(r"^/api/runs/([^/]+)/([^/]+)/metrics$")
_ROUTE_METRIC_PATHS = re.compile(r"^/api/runs/([^/]+)/([^/]+)/metric-paths$")


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
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            # GET /api/runs
            m = _ROUTE_RUNS.match(path)
            if m:
                runs = _scan_runs(self.projects_dir)
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
                configs = _get_configs(db_path)
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
                metrics = _get_metrics(db_path, metric_path)
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
                paths = _get_metric_paths(db_path)
                self._send_json({"paths": paths})
                return

            self._send_error(404, "Not found")

        except Exception as e:
            self._send_error(500, str(e))

    def _send_json(self, data: Any) -> None:
        """Send a JSON response with CORS headers."""
        # Keep NaN/Infinity behavior consistent with normal Python JSON handling.
        body = json.dumps(data, allow_nan=True).encode("utf-8")
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
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default request logging."""
        pass


def run_server(projects_dir: Path, port: int = 8765) -> None:
    """Start the local HTTP server.

    Args:
        projects_dir: Directory containing project subdirectories with .sqlite files.
        port: Port to listen on (default: 8765).
    """
    _RequestHandler.projects_dir = projects_dir

    server = ThreadingHTTPServer(("127.0.0.1", port), _RequestHandler)

    print(f"Goodseed server running at http://localhost:{port}")
    print(f"View your runs at https://goodseed.ai/app/local?port={port}")
    print(f"Data directory: {projects_dir}")
    print(f"Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()
