"""Capture stdout and stderr output as string series.

Monkey-patches ``sys.stdout`` and ``sys.stderr`` with ``StreamWithMemory``
wrappers that buffer all writes with timestamps while forwarding to the
original streams.  A background daemon thread periodically drains the
buffers and logs complete lines via the run's ``log_strings()``.

Multiple ``Run`` instances can share the same wrapped streams via the
subscriber model (each gets its own read offset into the shared buffer).
"""

from __future__ import annotations

import io
import sys
import threading
from datetime import datetime, timedelta
from collections.abc import Callable
from typing import Any

from goodseed.monitoring.daemon import MonitoringDaemon

_BUFFER_CHAR_CAPACITY = 10_000_000  # 10 M chars


class StreamWithMemory:
    """Transparent wrapper around a text stream that buffers writes."""

    def __init__(self, original: Any) -> None:
        self._original = original
        self._buffer: list[tuple[datetime, str]] = []
        self._chars = 0
        self._lock = threading.Lock()
        self._offsets: dict[str, int] = {}  # subscriber_id -> read offset

    def register(self, subscriber_id: str) -> None:
        with self._lock:
            if subscriber_id not in self._offsets:
                self._offsets[subscriber_id] = len(self._buffer)

    def unregister(self, subscriber_id: str) -> int:
        """Remove subscriber.  Returns number of remaining subscribers."""
        with self._lock:
            self._offsets.pop(subscriber_id, None)
            return len(self._offsets)

    def drain(self, subscriber_id: str) -> list[tuple[datetime, str]]:
        """Return all buffered data since the subscriber's last read."""
        with self._lock:
            offset = self._offsets.get(subscriber_id, len(self._buffer))
            data = self._buffer[offset:]
            self._offsets[subscriber_id] = len(self._buffer)
            return data

    def write(self, data: str) -> int:
        ts = datetime.now()
        n = self._original.write(data)
        if n is None:
            n = len(data)
        with self._lock:
            # Evict old data if buffer is full
            if self._chars + n > _BUFFER_CHAR_CAPACITY:
                self._evict(max(n, _BUFFER_CHAR_CAPACITY // 2))
            self._buffer.append((ts, data[:n]))
            self._chars += min(n, _BUFFER_CHAR_CAPACITY)
        return n

    def flush(self) -> None:
        self._original.flush()

    def _evict(self, chars_to_drop: int) -> None:
        """Drop oldest entries totalling *chars_to_drop* chars."""
        dropped_chars = 0
        dropped_entries = 0
        while dropped_chars < chars_to_drop and dropped_entries < len(self._buffer):
            dropped_chars += len(self._buffer[dropped_entries][1])
            dropped_entries += 1
        self._buffer = self._buffer[dropped_entries:]
        self._chars -= dropped_chars
        for sid in self._offsets:
            self._offsets[sid] = max(0, self._offsets[sid] - dropped_entries)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


# Module-level global state (shared across Run instances)
_lock = threading.Lock()
_stdout_mem: StreamWithMemory | None = None
_stderr_mem: StreamWithMemory | None = None
_original_stdout = sys.stdout
_original_stderr = sys.stderr


def _subscribe(subscriber_id: str, capture_stdout: bool, capture_stderr: bool) -> None:
    global _stdout_mem, _stderr_mem
    with _lock:
        if capture_stdout:
            if _stdout_mem is None:
                _stdout_mem = StreamWithMemory(sys.stdout)
                sys.stdout = _stdout_mem  # type: ignore[assignment]
            _stdout_mem.register(subscriber_id)
        if capture_stderr:
            if _stderr_mem is None:
                _stderr_mem = StreamWithMemory(sys.stderr)
                sys.stderr = _stderr_mem  # type: ignore[assignment]
            _stderr_mem.register(subscriber_id)


def _unsubscribe(subscriber_id: str) -> None:
    global _stdout_mem, _stderr_mem
    with _lock:
        if _stdout_mem is not None:
            if _stdout_mem.unregister(subscriber_id) == 0:
                sys.stdout = _original_stdout
                _stdout_mem = None
        if _stderr_mem is not None:
            if _stderr_mem.unregister(subscriber_id) == 0:
                sys.stderr = _original_stderr
                _stderr_mem = None


class _PartialLine:
    """Accumulates partial line data until a newline is seen."""

    def __init__(self) -> None:
        self._buf = io.StringIO()
        self._ts: datetime | None = None
        self._last_flush: datetime | None = None

    def write(self, ts: datetime, data: str) -> None:
        self._buf.write(data)
        self._ts = ts

    def clear(self) -> None:
        self._buf.truncate(0)
        self._buf.seek(0)
        self._ts = None

    def flush(self) -> tuple[datetime | None, str]:
        text = self._buf.getvalue()
        ts = self._ts
        self.clear()
        self._last_flush = datetime.now()
        return ts, text

    @property
    def last_flush(self) -> datetime | None:
        return self._last_flush


def _data_to_lines(
    partial: _PartialLine,
    data: list[tuple[datetime, str]],
    max_delay: timedelta,
) -> list[tuple[datetime, str]]:
    """Split buffered data into complete lines.

    Handles ``\\r`` for progress-bar output (resets partial line).
    If no newline arrives within *max_delay*, flushes the partial buffer.
    """
    lines: list[tuple[datetime, str]] = []
    for ts, chunk in data:
        if not chunk:
            continue
        pos = 0
        while True:
            lf = chunk.find("\n", pos)
            if lf == -1:
                break
            # Handle \r before the \n (progress bars)
            cr = chunk.rfind("\r", pos, lf)
            if cr != -1:
                partial.clear()
                pos = cr + 1
            partial.write(ts, chunk[pos:lf])
            fts, line = partial.flush()
            if fts and line:
                lines.append((fts, line))
            pos = lf + 1

        # Remaining text after last \n
        if pos < len(chunk):
            cr = chunk.rfind("\r", pos)
            if cr != -1:
                partial.clear()
                pos = cr + 1
            partial.write(ts, chunk[pos:])

    # Flush stale partial data
    if not partial.last_flush or datetime.now() - partial.last_flush >= max_delay:
        fts, line = partial.flush()
        if fts and line:
            lines.append((fts, line))

    return lines



class ConsoleCaptureDaemon(MonitoringDaemon):
    """Periodically drains captured stdout/stderr into string series."""

    def __init__(
        self,
        *,
        subscriber_id: str,
        namespace: str,
        capture_stdout: bool,
        capture_stderr: bool,
        log_fn: Callable[[dict[str, str], int], None],
        interval: float = 1.0,
    ) -> None:
        super().__init__(interval=interval, name="goodseed-console-capture")
        self._subscriber_id = subscriber_id
        self._capture_stdout = capture_stdout
        self._capture_stderr = capture_stderr
        self._log_fn = log_fn
        self._stdout_path = f"{namespace}/stdout"
        self._stderr_path = f"{namespace}/stderr"
        self._stdout_partial = _PartialLine()
        self._stderr_partial = _PartialLine()
        self._step = 0

        _subscribe(subscriber_id, capture_stdout, capture_stderr)

    def work(self) -> None:
        self._flush(max_delay=timedelta(seconds=5))

    def final_flush(self) -> None:
        """Flush all remaining data (call before unsubscribing)."""
        self._flush(max_delay=timedelta(seconds=0))

    def close(self) -> None:
        """Stop the daemon, flush remaining data, and restore streams."""
        self.stop()
        self.final_flush()
        _unsubscribe(self._subscriber_id)

    def _flush(self, max_delay: timedelta) -> None:
        if self._capture_stdout and _stdout_mem is not None:
            data = _stdout_mem.drain(self._subscriber_id)
            lines = _data_to_lines(self._stdout_partial, data, max_delay)
            for _ts, line in lines:
                self._log_fn({self._stdout_path: line}, self._step)
                self._step += 1

        if self._capture_stderr and _stderr_mem is not None:
            data = _stderr_mem.drain(self._subscriber_id)
            lines = _data_to_lines(self._stderr_partial, data, max_delay)
            for _ts, line in lines:
                self._log_fn({self._stderr_path: line}, self._step)
                self._step += 1
