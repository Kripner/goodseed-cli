"""Local SQLite storage for experiment data.

Each run is stored as a single SQLite file. Data is written by the main
process during training and read by the HTTP server for visualization.

Thread-safe via RLock. WAL mode enables concurrent reads from the server
while the training process writes.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from goodseed.config import ensure_dir
from goodseed.utils import deserialize_value

logger = logging.getLogger("goodseed.storage")


def _get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-4000")
    conn.row_factory = sqlite3.Row
    return conn


SCHEMA_VERSION = 1


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS run_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS configs (
            path TEXT PRIMARY KEY,
            type_tag TEXT NOT NULL,
            value TEXT,
            updated_at TEXT NOT NULL,
            uploaded INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS metric_series (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS metric_points (
            series_id INTEGER NOT NULL REFERENCES metric_series(id),
            step REAL NOT NULL,
            y REAL NOT NULL,
            ts INTEGER NOT NULL,
            uploaded INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (series_id, step)
        );

        CREATE TABLE IF NOT EXISTS string_series (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS string_points (
            series_id INTEGER NOT NULL REFERENCES string_series(id),
            step REAL NOT NULL,
            value TEXT NOT NULL,
            ts INTEGER NOT NULL,
            uploaded INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (series_id, step)
        );
    """)
    # Stamp schema version on fresh databases.
    row = conn.execute(
        "SELECT value FROM run_meta WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO run_meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
    conn.commit()


class LocalStorage:
    """Thread-safe SQLite storage for a single run."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._series_cache: dict[str, int] = {}
        self._string_series_cache: dict[str, int] = {}

        ensure_dir(self.db_path.parent)
        self._conn = _get_connection(self.db_path)
        _init_schema(self._conn)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _get_series_id(self, conn: sqlite3.Connection, path: str) -> int:
        """Get or create a series ID for a metric path. Must hold the lock."""
        sid = self._series_cache.get(path)
        if sid is not None:
            return sid
        conn.execute(
            "INSERT OR IGNORE INTO metric_series (path) VALUES (?)", (path,)
        )
        row = conn.execute(
            "SELECT id FROM metric_series WHERE path = ?", (path,)
        ).fetchone()
        sid = row["id"]
        self._series_cache[path] = sid
        return sid

    def set_meta(self, key: str, value: str) -> None:
        with self._transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO run_meta (key, value) VALUES (?, ?)",
                (key, value),
            )

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            row = self._conn.execute(
                "SELECT value FROM run_meta WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None

    def log_configs(self, data: dict[str, tuple[str, str]]) -> None:
        """Log config values. data: {path: (type_tag, serialized_value)}"""
        now = datetime.now(timezone.utc).isoformat()
        with self._transaction() as conn:
            for path, (type_tag, value) in data.items():
                conn.execute(
                    """INSERT OR REPLACE INTO configs
                       (path, type_tag, value, updated_at)
                       VALUES (?, ?, ?, ?)""",
                    (path, type_tag, value, now),
                )

    def get_configs(self) -> dict[str, tuple[str, str]]:
        """Get all configs as {path: (type_tag, value)}."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            rows = self._conn.execute(
                "SELECT path, type_tag, value FROM configs"
            ).fetchall()
            return {row["path"]: (row["type_tag"], row["value"]) for row in rows}

    def get_config(self, path: str) -> tuple[str, str] | None:
        """Get a single config value as (type_tag, value), or None."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            row = self._conn.execute(
                "SELECT type_tag, value FROM configs WHERE path = ?", (path,)
            ).fetchone()
            return (row["type_tag"], row["value"]) if row else None

    def log_metric_points(
        self, points: list[tuple[str, int | float, float, int]]
    ) -> None:
        """Log metric points. Each tuple: (path, step, y, ts_unix)."""
        with self._transaction() as conn:
            for path, step, y, ts in points:
                series_id = self._get_series_id(conn, path)
                conn.execute(
                    """INSERT OR REPLACE INTO metric_points
                       (series_id, step, y, ts) VALUES (?, ?, ?, ?)""",
                    (series_id, step, y, ts),
                )

    def get_last_metric_step(self, path: str) -> int | float | None:
        """Get the last (maximum) step value for a metric series."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            sid = self._series_cache.get(path)
            if sid is None:
                row = self._conn.execute(
                    "SELECT id FROM metric_series WHERE path = ?", (path,)
                ).fetchone()
                if row is None:
                    return None
                sid = row["id"]
            row = self._conn.execute(
                "SELECT MAX(step) as max_step FROM metric_points WHERE series_id = ?",
                (sid,),
            ).fetchone()
            if row is None or row["max_step"] is None:
                return None
            return row["max_step"]

    def get_metric_points(
        self, path: str | None = None
    ) -> list[dict[str, Any]]:
        """Get metric points, optionally filtered by path."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            if path:
                rows = self._conn.execute(
                    """SELECT s.path, p.step, p.y, p.ts
                       FROM metric_points p
                       JOIN metric_series s ON p.series_id = s.id
                       WHERE s.path = ?
                       ORDER BY p.step""",
                    (path,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT s.path, p.step, p.y, p.ts
                       FROM metric_points p
                       JOIN metric_series s ON p.series_id = s.id
                       ORDER BY s.path, p.step"""
                ).fetchall()
            return [dict(row) for row in rows]

    def get_all_max_steps(self) -> dict[str, int | float]:
        """Get the max step for every metric and string series path."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            result: dict[str, int | float] = {}
            rows = self._conn.execute(
                """SELECT s.path, MAX(p.step) as max_step
                   FROM metric_points p
                   JOIN metric_series s ON p.series_id = s.id
                   GROUP BY s.path"""
            ).fetchall()
            for row in rows:
                result[row["path"]] = row["max_step"]
            rows = self._conn.execute(
                """SELECT s.path, MAX(p.step) as max_step
                   FROM string_points p
                   JOIN string_series s ON p.series_id = s.id
                   GROUP BY s.path"""
            ).fetchall()
            for row in rows:
                result[row["path"]] = row["max_step"]
            return result

    def get_metric_paths(self) -> list[str]:
        """Get all metric paths that have at least one point."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            rows = self._conn.execute(
                """SELECT DISTINCT s.path FROM metric_series s
                   JOIN metric_points p ON s.id = p.series_id
                   ORDER BY s.path"""
            ).fetchall()
            return [row["path"] for row in rows]

    def _get_string_series_id(self, conn: sqlite3.Connection, path: str) -> int:
        """Get or create a series ID for a string series path. Must hold the lock."""
        sid = self._string_series_cache.get(path)
        if sid is not None:
            return sid
        conn.execute(
            "INSERT OR IGNORE INTO string_series (path) VALUES (?)", (path,)
        )
        row = conn.execute(
            "SELECT id FROM string_series WHERE path = ?", (path,)
        ).fetchone()
        sid = row["id"]
        self._string_series_cache[path] = sid
        return sid

    def log_string_points(
        self, points: list[tuple[str, int | float, str, int]]
    ) -> None:
        """Log string series points. Each tuple: (path, step, value, ts_unix)."""
        with self._transaction() as conn:
            for path, step, value, ts in points:
                series_id = self._get_string_series_id(conn, path)
                conn.execute(
                    """INSERT OR REPLACE INTO string_points
                       (series_id, step, value, ts) VALUES (?, ?, ?, ?)""",
                    (series_id, step, value, ts),
                )

    def get_last_string_step(self, path: str) -> int | float | None:
        """Get the last (maximum) step value for a string series."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            sid = self._string_series_cache.get(path)
            if sid is None:
                row = self._conn.execute(
                    "SELECT id FROM string_series WHERE path = ?", (path,)
                ).fetchone()
                if row is None:
                    return None
                sid = row["id"]
            row = self._conn.execute(
                "SELECT MAX(step) as max_step FROM string_points WHERE series_id = ?",
                (sid,),
            ).fetchone()
            if row is None or row["max_step"] is None:
                return None
            return row["max_step"]

    def get_string_points(
        self, path: str | None = None
    ) -> list[dict[str, Any]]:
        """Get string series points, optionally filtered by path."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            if path:
                rows = self._conn.execute(
                    """SELECT s.path, p.step, p.value, p.ts
                       FROM string_points p
                       JOIN string_series s ON p.series_id = s.id
                       WHERE s.path = ?
                       ORDER BY p.step""",
                    (path,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT s.path, p.step, p.value, p.ts
                       FROM string_points p
                       JOIN string_series s ON p.series_id = s.id
                       ORDER BY s.path, p.step"""
                ).fetchall()
            return [dict(row) for row in rows]

    def get_string_series_paths(self) -> list[str]:
        """Get all string series paths that have at least one point."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            rows = self._conn.execute(
                """SELECT DISTINCT s.path FROM string_series s
                   JOIN string_points p ON s.id = p.series_id
                   ORDER BY s.path"""
            ).fetchall()
            return [row["path"] for row in rows]

    def get_unuploaded_metric_points(self, limit: int = 5000) -> list[dict[str, Any]]:
        """Get metric points not yet uploaded, with their series path."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            rows = self._conn.execute(
                """SELECT s.path, p.series_id, p.step, p.y, p.ts
                   FROM metric_points p
                   JOIN metric_series s ON p.series_id = s.id
                   WHERE p.uploaded = 0
                   ORDER BY p.ts
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def mark_metric_points_uploaded(
        self, keys: list[tuple[int, float, float, int]]
    ) -> None:
        """Mark metric points as uploaded (optimistic concurrency).

        keys: [(series_id, step, y, ts)] -- values in WHERE prevent marking
        a row that was overwritten by the main process since it was read.
        """
        with self._transaction() as conn:
            conn.executemany(
                """UPDATE metric_points SET uploaded = 1
                   WHERE series_id = ? AND step = ? AND y = ? AND ts = ?""",
                keys,
            )

    def get_unuploaded_string_points(self, limit: int = 5000) -> list[dict[str, Any]]:
        """Get string points not yet uploaded, with their series path."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            rows = self._conn.execute(
                """SELECT s.path, p.series_id, p.step, p.value, p.ts
                   FROM string_points p
                   JOIN string_series s ON p.series_id = s.id
                   WHERE p.uploaded = 0
                   ORDER BY p.ts
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def mark_string_points_uploaded(
        self, keys: list[tuple[int, float, str, int]]
    ) -> None:
        """Mark string points as uploaded (optimistic concurrency).

        keys: [(series_id, step, value, ts)]
        """
        with self._transaction() as conn:
            conn.executemany(
                """UPDATE string_points SET uploaded = 1
                   WHERE series_id = ? AND step = ? AND value = ? AND ts = ?""",
                keys,
            )

    def get_unuploaded_configs(self) -> list[dict[str, Any]]:
        """Get configs not yet uploaded."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            rows = self._conn.execute(
                "SELECT path, type_tag, value FROM configs WHERE uploaded = 0"
            ).fetchall()
            return [dict(row) for row in rows]

    def mark_configs_uploaded(
        self, keys: list[tuple[str, str, str]]
    ) -> None:
        """Mark configs as uploaded (optimistic concurrency).

        keys: [(path, type_tag, value)]
        """
        with self._transaction() as conn:
            conn.executemany(
                """UPDATE configs SET uploaded = 1
                   WHERE path = ? AND type_tag = ? AND value = ?""",
                keys,
            )

    def field_exists(self, path: str) -> bool:
        """Check if a field exists in configs, metric_series, or string_series."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            row = self._conn.execute(
                "SELECT 1 FROM configs WHERE path = ? LIMIT 1", (path,)
            ).fetchone()
            if row:
                return True
            row = self._conn.execute(
                "SELECT 1 FROM metric_series WHERE path = ? LIMIT 1", (path,)
            ).fetchone()
            if row:
                return True
            row = self._conn.execute(
                "SELECT 1 FROM string_series WHERE path = ? LIMIT 1", (path,)
            ).fetchone()
            return row is not None

    def get_last_metric_value(self, path: str) -> tuple[float, float] | None:
        """Get the last (step, y) for a metric series, or None."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            sid = self._series_cache.get(path)
            if sid is None:
                row = self._conn.execute(
                    "SELECT id FROM metric_series WHERE path = ?", (path,)
                ).fetchone()
                if row is None:
                    return None
                sid = row["id"]
            row = self._conn.execute(
                "SELECT step, y FROM metric_points WHERE series_id = ? ORDER BY step DESC LIMIT 1",
                (sid,),
            ).fetchone()
            if row is None:
                return None
            return (row["step"], row["y"])

    def get_last_string_value(self, path: str) -> tuple[float, str] | None:
        """Get the last (step, value) for a string series, or None."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            sid = self._string_series_cache.get(path)
            if sid is None:
                row = self._conn.execute(
                    "SELECT id FROM string_series WHERE path = ?", (path,)
                ).fetchone()
                if row is None:
                    return None
                sid = row["id"]
            row = self._conn.execute(
                "SELECT step, value FROM string_points WHERE series_id = ? ORDER BY step DESC LIMIT 1",
                (sid,),
            ).fetchone()
            if row is None:
                return None
            return (row["step"], row["value"])

    def count_unuploaded(self) -> int:
        """Return total count of unuploaded items across all tables."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            total = 0
            for table in ("metric_points", "string_points", "configs"):
                row = self._conn.execute(
                    f"SELECT COUNT(*) as cnt FROM {table} WHERE uploaded = 0"
                ).fetchone()
                total += row["cnt"]
            return total

    def checkpoint_wal(self) -> None:
        """Checkpoint WAL to consolidate into a single .sqlite file."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def delete_db_file(self) -> None:
        """Close connection and delete the database file."""
        self.close()
        for path in (
            self.db_path,
            Path(str(self.db_path) + "-wal"),
            Path(str(self.db_path) + "-shm"),
        ):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def __enter__(self) -> LocalStorage:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# Read-only query functions used by the server and CLI.
# Each function opens a temporary read-only connection.


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite database in read-only mode."""
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout=3000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _ts_to_iso(ts: int) -> str:
    """Convert a unix timestamp to ISO 8601 string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def read_run_meta(db_path: Path) -> dict[str, str]:
    """Read all run_meta key-value pairs from a run database."""
    conn = _open_readonly(db_path)
    try:
        rows = conn.execute("SELECT key, value FROM run_meta").fetchall()
        return {row["key"]: row["value"] for row in rows}
    finally:
        conn.close()


def read_run_summary(
    db_path: Path,
) -> tuple[dict[str, str], list[str], list[str]]:
    """Read run metadata, string series paths, and metric paths.

    Returns (meta_dict, string_series_paths, metric_paths).
    """
    conn = _open_readonly(db_path)
    try:
        meta_rows = conn.execute("SELECT key, value FROM run_meta").fetchall()
        meta = {row["key"]: row["value"] for row in meta_rows}

        ss_rows = conn.execute(
            """SELECT DISTINCT s.path FROM string_series s
               JOIN string_points p ON s.id = p.series_id
               ORDER BY s.path"""
        ).fetchall()
        string_series_paths = [row["path"] for row in ss_rows]

        mp_rows = conn.execute(
            """SELECT DISTINCT s.path FROM metric_series s
               JOIN metric_points p ON s.id = p.series_id
               ORDER BY s.path"""
        ).fetchall()
        metric_paths = [row["path"] for row in mp_rows]

        return meta, string_series_paths, metric_paths
    finally:
        conn.close()


def read_configs(db_path: Path) -> dict[str, Any]:
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


def read_metrics(
    db_path: Path, metric_path: str | None = None,
) -> list[dict[str, Any]]:
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
            "logged_at": _ts_to_iso(row["ts"]),
        }
        for row in rows
    ]


def read_metric_paths(db_path: Path) -> list[str]:
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


def read_string_series(
    db_path: Path,
    series_path: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    tail: int | None = None,
) -> dict[str, Any]:
    """Read string series points from a run database.

    Returns {"points": [...], "total": <int>} where total is the full
    count before limit/offset are applied.

    If tail is given, returns the last tail rows (overrides limit/offset).
    """
    conn = _open_readonly(db_path)
    try:
        where = "WHERE s.path = ?" if series_path else ""
        base_params: list = [series_path] if series_path else []

        total_row = conn.execute(
            f"""SELECT COUNT(*) as cnt
                FROM string_points p
                JOIN string_series s ON p.series_id = s.id
                {where}""",
            base_params,
        ).fetchone()
        total = total_row["cnt"] if total_row else 0

        order_col = "p.step" if series_path else "s.path, p.step"

        if tail is not None:
            query = f"""SELECT * FROM (
                SELECT s.path, p.step, p.value, p.ts
                FROM string_points p
                JOIN string_series s ON p.series_id = s.id
                {where}
                ORDER BY {order_col} DESC
                LIMIT ?
            ) sub ORDER BY {"step" if series_path else "path, step"}"""
            rows = conn.execute(query, base_params + [tail]).fetchall()
        else:
            query = f"""SELECT s.path, p.step, p.value, p.ts
                        FROM string_points p
                        JOIN string_series s ON p.series_id = s.id
                        {where}
                        ORDER BY {order_col}"""
            params = list(base_params)
            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])
            rows = conn.execute(query, params).fetchall()
    except sqlite3.OperationalError as e:
        logger.debug("string_series table not available in %s: %s", db_path, e)
        return {"points": [], "total": 0}
    finally:
        conn.close()

    points = [
        {
            "path": row["path"],
            "step": row["step"],
            "value": row["value"],
            "logged_at": _ts_to_iso(row["ts"]),
        }
        for row in rows
    ]
    return {"points": points, "total": total}


def downsample_metrics(
    db_path: Path,
    metric_path: str,
    point_count: int,
    first_point_index: int | None = None,
    last_point_index: int | None = None,
) -> dict[str, Any]:
    """Downsample metric points using power-of-2 bucketing.

    Picks the smallest bucket size k = 2^n such that
    ceil(N/k) <= point_count, then returns one aggregated point per
    bucket (avg step/value, min/max Y) plus exact first/last bookend points.
    """
    conn = _open_readonly(db_path)
    try:
        count_row = conn.execute(
            """SELECT COUNT(*) AS cnt
               FROM metric_points p
               JOIN metric_series s ON p.series_id = s.id
               WHERE s.path = ?""",
            (metric_path,),
        ).fetchone()
        total_count = count_row["cnt"] if count_row else 0

        if total_count == 0:
            return {
                "points": [],
                "firstPointIndex": 0,
                "lastPointIndex": 0,
                "totalCount": 0,
                "downsampled": False,
            }

        fpi = max(0, first_point_index) if first_point_index is not None else 0
        lpi = min(total_count - 1, last_point_index) if last_point_index is not None else total_count - 1
        if fpi > lpi:
            return {
                "points": [],
                "firstPointIndex": fpi,
                "lastPointIndex": lpi,
                "totalCount": total_count,
                "downsampled": False,
            }

        rows = conn.execute(
            """WITH numbered AS (
                   SELECT p.step, p.y, p.ts,
                          ROW_NUMBER() OVER (ORDER BY p.step) - 1 AS rn
                   FROM metric_points p
                   JOIN metric_series s ON p.series_id = s.id
                   WHERE s.path = ?
               )
               SELECT step, y, ts, rn
               FROM numbered
               WHERE rn >= ? AND rn <= ?
               ORDER BY rn""",
            (metric_path, fpi, lpi),
        ).fetchall()
    finally:
        conn.close()

    raw_count = len(rows)

    if raw_count <= point_count:
        points = [
            {
                "path": metric_path,
                "step": row["step"],
                "value": row["y"],
                "minY": row["y"],
                "maxY": row["y"],
                "downsampled": False,
                "logged_at": _ts_to_iso(row["ts"]),
            }
            for row in rows
        ]
        return {
            "points": points,
            "firstPointIndex": fpi,
            "lastPointIndex": lpi,
            "totalCount": total_count,
            "downsampled": False,
        }

    k = 1
    while math.ceil(raw_count / k) > point_count:
        k *= 2
    num_buckets = math.ceil(raw_count / k)

    points: list[dict[str, Any]] = []
    for b in range(num_buckets):
        start = b * k
        end = min(start + k, raw_count)
        bucket = rows[start:end]

        sum_step = 0.0
        sum_y = 0.0
        min_y = float("inf")
        max_y = float("-inf")
        valid = 0

        for r in bucket:
            y = r["y"]
            sum_step += r["step"]
            if not (math.isnan(y) or math.isinf(y)):
                sum_y += y
                valid += 1
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

        n = len(bucket)
        if valid > 0:
            avg_y = sum_y / valid
        else:
            avg_y = float("nan")
            min_y = float("nan")
            max_y = float("nan")

        points.append({
            "path": metric_path,
            "step": sum_step / n,
            "value": avg_y,
            "minY": min_y,
            "maxY": max_y,
            "downsampled": True,
            "logged_at": _ts_to_iso(bucket[n // 2]["ts"]),
        })

    # Bookend: exact first and last raw points
    first_row, last_row = rows[0], rows[-1]
    if points[0]["step"] != first_row["step"]:
        points.insert(0, {
            "path": metric_path,
            "step": first_row["step"],
            "value": first_row["y"],
            "minY": first_row["y"],
            "maxY": first_row["y"],
            "downsampled": False,
            "logged_at": _ts_to_iso(first_row["ts"]),
        })
    if points[-1]["step"] != last_row["step"]:
        points.append({
            "path": metric_path,
            "step": last_row["step"],
            "value": last_row["y"],
            "minY": last_row["y"],
            "maxY": last_row["y"],
            "downsampled": False,
            "logged_at": _ts_to_iso(last_row["ts"]),
        })

    return {
        "points": points,
        "firstPointIndex": fpi,
        "lastPointIndex": lpi,
        "totalCount": total_count,
        "downsampled": True,
    }


def write_run_meta(db_path: Path, key: str, value: str | None) -> None:
    """Set or delete a run_meta key in a run database."""
    conn = _get_connection(db_path)
    try:
        if value is None:
            conn.execute("DELETE FROM run_meta WHERE key = ?", (key,))
        else:
            conn.execute(
                "INSERT OR REPLACE INTO run_meta (key, value) VALUES (?, ?)",
                (key, value),
            )
        conn.commit()
    finally:
        conn.close()
