"""Background sync thread for uploading run data to the remote API.

Reads unuploaded rows from SQLite, sends them to the remote API, and marks
them as uploaded on confirmed success. Runs in a background thread.

The SQLite database (in WAL mode) serves as the durable queue between the
producer and the sync worker.
"""

from __future__ import annotations

import gzip
import json
import logging
import struct
import threading
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from goodseed.config import API_BASE, APP_URL
from goodseed.storage import LocalStorage

logger = logging.getLogger("goodseed.sync")

# Internal dev toggle: "json" or "protobuf" for the ingest endpoint
_INGEST_FORMAT = "json"

# API limits (matching the backend)
_MAX_POINTS_PER_INGEST = 2_000_000
_MAX_CONFIGS_PER_REQUEST = 1000
_MAX_CONFIG_VALUE_CHARS = 10_000
_BATCH_SIZE = 5000
_HTTP_TIMEOUT = 30
_SYNC_INTERVAL = 5  # seconds between sync cycles
_MAX_DRAIN_ITERATIONS = 100  # safety limit for drain loop
_MAX_BODY_BYTES = 16 * 1024 * 1024  # 16 MiB per request


def _api_post(
    url: str,
    *,
    api_key: str,
    body: dict[str, Any] | None = None,
    raw_body: bytes | None = None,
    content_type: str = "application/json",
    compress: bool = False,
) -> tuple[int, bytes]:
    """POST to the API. Returns (status_code, response_bytes).

    When *compress* is True the body is gzip-compressed and a
    ``Content-Encoding: gzip`` header is added.

    Status 0 means network/connection error. Never raises.
    """
    if raw_body is not None:
        data = raw_body
    elif body is not None:
        data = json.dumps(body).encode("utf-8")
    else:
        data = b""

    if compress:
        data = gzip.compress(data, compresslevel=6)

    req = Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", content_type)
    if compress:
        req.add_header("Content-Encoding", "gzip")

    try:
        with urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            return resp.status, resp.read()
    except HTTPError as e:
        body_bytes = b""
        try:
            body_bytes = e.read()
        except Exception:
            pass
        return e.code, body_bytes
    except (URLError, OSError) as e:
        logger.debug("Network error: %s", e)
        return 0, b""


def _api_get(
    url: str,
    *,
    api_key: str,
    params: dict[str, Any] | None = None,
) -> tuple[int, bytes]:
    """GET from the API. Returns (status_code, response_bytes).

    Status 0 means network/connection error. Never raises.
    """
    if params:
        qs = urllib.parse.urlencode(
            {k: str(v) for k, v in params.items() if v is not None}
        )
        url = f"{url}?{qs}"

    req = Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            return resp.status, resp.read()
    except HTTPError as e:
        body_bytes = b""
        try:
            body_bytes = e.read()
        except Exception:
            pass
        return e.code, body_bytes
    except (URLError, OSError) as e:
        logger.debug("Network error: %s", e)
        return 0, b""


def api_get_json(
    path: str,
    *,
    api_key: str,
    params: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    """GET JSON payload from GoodSeed API."""
    status, resp = _api_get(f"{API_BASE}{path}", api_key=api_key, params=params)
    if not resp:
        return status, None
    try:
        return status, json.loads(resp)
    except Exception:
        return status, resp.decode("utf-8", errors="replace")


def api_post_json(
    path: str,
    *,
    api_key: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    """POST JSON payload to GoodSeed API."""
    status, resp = _api_post(f"{API_BASE}{path}", api_key=api_key, body=body)
    if not resp:
        return status, None
    try:
        return status, json.loads(resp)
    except Exception:
        return status, resp.decode("utf-8", errors="replace")


# Protobuf hand-encoding (for _INGEST_FORMAT = "protobuf")

def _encode_varint(value: int) -> bytes:
    """Encode an unsigned integer as a protobuf varint."""
    parts = []
    while value > 0x7F:
        parts.append((value & 0x7F) | 0x80)
        value >>= 7
    parts.append(value & 0x7F)
    return bytes(parts)


def _encode_field_varint(field_number: int, value: int) -> bytes:
    """Encode a varint field (wire type 0)."""
    tag = _encode_varint((field_number << 3) | 0)
    return tag + _encode_varint(value)


def _encode_field_string(field_number: int, value: str) -> bytes:
    """Encode a length-delimited string field (wire type 2)."""
    encoded = value.encode("utf-8")
    tag = _encode_varint((field_number << 3) | 2)
    return tag + _encode_varint(len(encoded)) + encoded


def _encode_field_double(field_number: int, value: float) -> bytes:
    """Encode a fixed64 double field (wire type 1)."""
    tag = _encode_varint((field_number << 3) | 1)
    return tag + struct.pack("<d", value)


def _encode_field_bytes(field_number: int, value: bytes) -> bytes:
    """Encode a length-delimited bytes field (wire type 2)."""
    tag = _encode_varint((field_number << 3) | 2)
    return tag + _encode_varint(len(value)) + value


def _encode_data_point(
    path: str, step: int, timestamp_ms: int,
    number: float | None = None, text: str | None = None,
) -> bytes:
    """Encode a single DataPoint protobuf message."""
    parts = _encode_field_string(1, path)
    parts += _encode_field_varint(2, step)
    parts += _encode_field_varint(3, timestamp_ms)
    if number is not None:
        parts += _encode_field_double(4, number)
    elif text is not None:
        parts += _encode_field_string(5, text)
    return parts


T = Any  # generic item type (tuples of original row + serialised bytes)


def _iter_sized_batches(
    items: list[T],
    item_size_fn: Callable[[T], int],
    overhead: int = 0,
    sep: int = 2,
    max_count: int = 0,
    max_bytes: int | None = None,
) -> Iterator[list[T]]:
    """Yield sub-lists of *items* fitting within *max_bytes* and *max_count*.

    Builds batches greedily in a single pass.  *item_size_fn(item)* returns
    the byte contribution of one item.  *overhead* is the fixed wrapper cost
    (e.g. the JSON envelope).  *sep* is the byte length of the delimiter
    between consecutive items (2 bytes for the JSON ``", "``, 0 for protobuf).
    *max_bytes* defaults to ``_MAX_BODY_BYTES`` when *None*.
    """
    if max_bytes is None:
        max_bytes = _MAX_BODY_BYTES
    batch: list[T] = []
    size = overhead
    for item in items:
        s = item_size_fn(item)
        gap = sep if batch else 0
        if batch and (
            size + gap + s > max_bytes
            or (max_count and len(batch) >= max_count)
        ):
            yield batch
            batch = []
            size = overhead
            gap = 0
        batch.append(item)
        size += gap + s
    if batch:
        yield batch


def _parse_api_error(resp: bytes | None) -> tuple[str | None, str]:
    """Parse API error payload into (error_code, message)."""
    if not resp:
        return None, ""
    text = resp.decode("utf-8", errors="replace")
    try:
        data = json.loads(text)
    except Exception:
        return None, text

    if not isinstance(data, dict):
        return None, text

    error_code = data.get("error_code")
    title = data.get("title")
    detail = data.get("detail")
    pieces = [p for p in [title, detail] if p]
    if pieces:
        return error_code, " ".join(pieces)
    return error_code, text


def _ensure_run_once(
    api_key: str,
    workspace: str,
    project_name: str,
    run_id: str,
    experiment_name: str | None,
    *,
    created_at: str | None = None,
    modified_at: str | None = None,
    log_errors: bool = True,
) -> tuple[str | None, str | None, bool]:
    """Ensure the run exists once.

    Returns (remote_id, error_message, retryable).
    """
    body: dict[str, Any] = {
        "workspace": workspace,
        "project": project_name,
        "run_id": run_id,
        "experiment_name": experiment_name,
    }
    if created_at:
        body["created_at"] = created_at
    if modified_at:
        body["modified_at"] = modified_at
    status, resp = _api_post(
        f"{API_BASE}/api/v1/runs",
        api_key=api_key,
        body=body,
    )

    if status == 200 and resp:
        data = json.loads(resp)
        return data.get("id"), None, False

    if status == 0:
        msg = (
            f"Could not connect to Goodseed ({API_BASE}).\n"
            "Check your network connection and try again."
        )
        if log_errors:
            logger.warning("ensure_run: %s", msg)
        return None, msg, True

    error_code, parsed_message = _parse_api_error(resp)
    project_full_name = f"{workspace}/{project_name}"

    if status == 404 and error_code == "Project.NotFound":
        msg = (
            f"Project '{project_full_name}' not found.\n"
            f"Create it at {APP_URL}."
        )
        if log_errors:
            logger.warning("ensure_run: %s", msg)
        return None, msg, False

    if status == 401:
        msg = (
            "Invalid API key.\n"
            "Set GOODSEED_API_KEY or pass api_key= to Run()."
        )
    elif status == 403:
        msg = (
            f"Access denied for project '{project_full_name}'.\n"
            f"Check that your API key has access to the '{workspace}' workspace."
        )
    else:
        msg = parsed_message or f"HTTP {status} while registering run."

    if log_errors:
        logger.warning("ensure_run: HTTP %d — %s", status, msg)

    retryable = status >= 500 or status == 429
    return None, msg, retryable


def _ensure_run(
    api_key: str, workspace: str, project_name: str,
    run_id: str, experiment_name: str | None,
    *,
    created_at: str | None = None,
    modified_at: str | None = None,
) -> str | None:
    """Ensure the run exists on the remote. Returns remote UUID or None."""
    remote_id, _, _ = _ensure_run_once(
        api_key,
        workspace,
        project_name,
        run_id,
        experiment_name,
        created_at=created_at,
        modified_at=modified_at,
        log_errors=True,
    )
    return remote_id


def _sync_configs(
    storage: LocalStorage, api_key: str, remote_id: str,
) -> int:
    """Upload unuploaded configs. Returns count uploaded."""
    configs = storage.get_unuploaded_configs()
    if not configs:
        return 0

    # Pre-serialise each config dict once, truncating oversized values.
    prepared: list[tuple[dict[str, Any], bytes]] = []
    for c in configs:
        value = c["value"]
        if isinstance(value, str) and len(value) > _MAX_CONFIG_VALUE_CHARS:
            value = value[:_MAX_CONFIG_VALUE_CHARS - 13] + "\n[truncated]"
            logger.debug(
                "Config value truncated to %d chars: %s",
                _MAX_CONFIG_VALUE_CHARS, c["path"],
            )
        raw = json.dumps(
            {"path": c["path"], "type_tag": c["type_tag"], "value": value}
        ).encode("utf-8")
        prepared.append((c, raw))

    uploaded = 0
    for batch in _iter_sized_batches(
        prepared, lambda x: len(x[1]),
        overhead=len(b'{"configs": []}'), max_count=_MAX_CONFIGS_PER_REQUEST,
    ):
        body = b'{"configs": [' + b", ".join(b for _, b in batch) + b"]}"
        status, resp = _api_post(
            f"{API_BASE}/api/v1/runs/{remote_id}/configs",
            api_key=api_key,
            raw_body=body,
            content_type="application/json",
            compress=True,
        )
        if status == 204:
            storage.mark_configs_uploaded([
                (c["path"], c["type_tag"], c["value"]) for c, _ in batch
            ])
            uploaded += len(batch)
        else:
            error_code, error_msg = _parse_api_error(resp)
            logger.warning(
                "Config upload failed (HTTP %d, %d items): %s",
                status, len(batch), error_msg or "no details",
            )
            break
    return uploaded


def _serialize_point(p: dict[str, Any], kind: str) -> bytes:
    """Serialise a single data point for the configured ingest format.

    Returns the wire bytes for one item (either a JSON object fragment or a
    protobuf field).
    """
    raw_step = p["step"]
    step = int(raw_step)
    ts_ms = int(p["ts"] * 1000)

    if _INGEST_FORMAT == "protobuf":
        if step != raw_step:
            raise ValueError(f"step must be an integer, got {raw_step!r}")
        if kind == "metric":
            pb = _encode_data_point(p["path"], step, ts_ms, number=p["y"])
        else:
            pb = _encode_data_point(p["path"], step, ts_ms, text=p["value"])
        return _encode_field_bytes(1, pb)
    else:
        pt: dict[str, Any] = {
            "path": p["path"], "step": step, "timestampMs": ts_ms,
        }
        if kind == "metric":
            pt["number"] = p["y"]
        else:
            pt["text"] = p["value"]
        return json.dumps(pt).encode("utf-8")


def _sync_ingest_points(
    points: list[dict[str, Any]],
    kind: str,
    api_key: str,
    remote_id: str,
    mark_uploaded: Callable[[list[tuple[Any, ...]]], None],
    upload_key: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> int:
    """Upload data points (metric or string) with size-limited batching.

    Returns count uploaded.
    """
    # Pre-serialise each point once.
    prepared: list[tuple[dict[str, Any], bytes]] = [
        (p, _serialize_point(p, kind)) for p in points
    ]

    if _INGEST_FORMAT == "protobuf":
        overhead, sep = 0, 0
    else:
        overhead, sep = len(b'{"points": []}'), 2

    uploaded = 0
    for batch in _iter_sized_batches(
        prepared, lambda x: len(x[1]),
        overhead=overhead, sep=sep, max_count=_MAX_POINTS_PER_INGEST,
    ):
        if _INGEST_FORMAT == "protobuf":
            body = b"".join(b for _, b in batch)
            ct = "application/protobuf"
        else:
            body = b'{"points": [' + b", ".join(b for _, b in batch) + b"]}"
            ct = "application/json"
        status, _ = _api_post(
            f"{API_BASE}/api/v1/runs/{remote_id}/ingest",
            api_key=api_key,
            raw_body=body,
            content_type=ct,
            compress=True,
        )
        if status == 200:
            mark_uploaded([upload_key(orig) for orig, _ in batch])
            uploaded += len(batch)
        else:
            logger.warning(
                "%s upload failed (HTTP %d, %d items)",
                kind.capitalize(), status, len(batch),
            )
            break
    return uploaded


def _sync_metric_points(
    storage: LocalStorage, api_key: str, remote_id: str,
) -> int:
    """Upload unuploaded metric points. Returns count uploaded."""
    points = storage.get_unuploaded_metric_points(limit=_BATCH_SIZE)
    if not points:
        return 0
    return _sync_ingest_points(
        points, "metric", api_key, remote_id,
        mark_uploaded=storage.mark_metric_points_uploaded,
        upload_key=lambda p: (p["series_id"], p["step"], p["y"], p["ts"]),
    )


def _sync_string_points(
    storage: LocalStorage, api_key: str, remote_id: str,
) -> int:
    """Upload unuploaded string points. Returns count uploaded."""
    points = storage.get_unuploaded_string_points(limit=_BATCH_SIZE)
    if not points:
        return 0
    return _sync_ingest_points(
        points, "string", api_key, remote_id,
        mark_uploaded=storage.mark_string_points_uploaded,
        upload_key=lambda p: (p["series_id"], p["step"], p["value"], p["ts"]),
    )


def _sync_cycle(
    storage: LocalStorage, api_key: str, remote_id: str,
) -> int:
    """Run one full sync cycle. Returns total items uploaded."""
    total = 0
    total += _sync_configs(storage, api_key, remote_id)
    total += _sync_metric_points(storage, api_key, remote_id)
    total += _sync_string_points(storage, api_key, remote_id)
    return total


def _send_status(api_key: str, remote_id: str, status: str) -> None:
    """Send the final run status to the remote API."""
    # Backend expects snake_case enum strings via JsonStringEnumConverter.
    valid = {"running", "finished", "failed"}
    status_value = status if status in valid else "finished"
    _api_post(
        f"{API_BASE}/api/v1/runs/{remote_id}/status",
        api_key=api_key,
        body={"status": status_value},
    )


def _sync_worker(
    db_path: str,
    api_key: str,
    workspace: str,
    project_name: str,
    run_id: str,
    experiment_name: str | None,
    shutdown_event: threading.Event,
    flush_event: threading.Event | None = None,
) -> None:
    """Entry point for the background sync thread."""
    storage = LocalStorage(Path(db_path))
    remote_id: str | None = None

    try:
        while True:
            if remote_id is None:
                remote_id = _ensure_run(
                    api_key, workspace, project_name, run_id, experiment_name,
                )

            if remote_id is not None:
                _sync_cycle(storage, api_key, remote_id)

            should_drain = shutdown_event.is_set()

            if should_drain:
                if remote_id is None:
                    remote_id = _ensure_run(
                        api_key, workspace, project_name, run_id, experiment_name,
                    )

                if remote_id is not None:
                    for _ in range(_MAX_DRAIN_ITERATIONS):
                        remaining = storage.count_unuploaded()
                        if remaining == 0:
                            break
                        _sync_cycle(storage, api_key, remote_id)

                    final_status = storage.get_meta("status") or "finished"
                    _send_status(api_key, remote_id, final_status)
                break

            # Sleep until the interval expires, shutdown is requested, or
            # a flush is requested (whichever comes first).
            if flush_event is not None and flush_event.is_set():
                flush_event.clear()
                # Skip the sleep — loop immediately for another sync cycle.
            else:
                shutdown_event.wait(_SYNC_INTERVAL)
                # Also check flush_event in case it fired during the wait.
                if flush_event is not None and flush_event.is_set():
                    flush_event.clear()
    except Exception:
        logger.exception("Sync thread error")
    finally:
        storage.close()


class SyncProcess:
    """Manages the background sync thread lifecycle."""

    def __init__(
        self,
        db_path: Path,
        api_key: str,
        workspace: str,
        project_name: str,
        run_id: str,
        experiment_name: str | None = None,
    ) -> None:
        self._db_path = db_path
        self._shutdown = threading.Event()
        self._flush = threading.Event()
        self._thread = threading.Thread(
            target=_sync_worker,
            args=(
                str(db_path), api_key, workspace, project_name,
                run_id, experiment_name, self._shutdown, self._flush,
            ),
            name=f"goodseed-sync-{run_id}",
            daemon=True,
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._thread.start()
        self._started = True

    def sync(self) -> None:
        """Request an immediate sync cycle (non-blocking)."""
        self._flush.set()

    def wait(self, timeout: float | None = None) -> None:
        """Block until all queued data has been uploaded.

        Args:
            timeout: Maximum seconds to wait. None means wait indefinitely.
        """
        import time

        self._flush.set()
        if not self._started:
            return
        deadline = None if timeout is None else time.monotonic() + timeout
        storage = LocalStorage(self._db_path)
        try:
            while True:
                remaining = storage.count_unuploaded()
                if remaining == 0:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break
                if not self._thread.is_alive():
                    break
                time.sleep(0.5)
        finally:
            storage.close()

    def close(self) -> None:
        """Signal shutdown and block until all data is uploaded."""
        self._shutdown.set()
        self._flush.set()  # Wake the worker immediately
        if self._started:
            self._thread.join()


def upload_run(db_path: Path, api_key: str) -> int:
    """Upload all unuploaded data from a run's database.

    Runs synchronously in the foreground. Returns total items uploaded.
    """
    storage = LocalStorage(db_path)
    try:
        project = storage.get_meta("project")
        run_id = storage.get_meta("run_id")
        name = storage.get_meta("name")
        created_at = storage.get_meta("created_at")
        modified_at = storage.get_meta("modified_at")

        if not project or not run_id:
            raise RuntimeError(
                f"Cannot upload: missing project or run_id in run metadata at {db_path}"
            )

        parts = project.split("/", 1)
        if len(parts) != 2:
            raise RuntimeError(
                f"project must be 'workspace/project', got: {project!r}"
            )
        workspace, project_name = parts

        remote_id = None
        last_error = "Could not establish run on remote."
        for _ in range(10):
            remote_id, error_message, retryable = _ensure_run_once(
                api_key,
                workspace,
                project_name,
                run_id,
                name,
                created_at=created_at,
                modified_at=modified_at,
                log_errors=False,
            )
            if remote_id is not None:
                break
            if error_message:
                last_error = error_message
            if not retryable:
                break
        if remote_id is None:
            raise RuntimeError(last_error)

        total = 0
        for _ in range(_MAX_DRAIN_ITERATIONS):
            remaining = storage.count_unuploaded()
            if remaining == 0:
                break
            total += _sync_cycle(storage, api_key, remote_id)

        status = storage.get_meta("status") or "finished"
        _send_status(api_key, remote_id, status)

        return total
    finally:
        storage.close()
