"""Monitoring manager: console capture, hardware metrics, and traceback capture."""

from __future__ import annotations

import hashlib
import os
import platform
import sys
import threading
import traceback
from collections.abc import Callable
from typing import Any

from goodseed.monitoring.console_capture import ConsoleCaptureDaemon
from goodseed.monitoring.hardware import HardwareMonitorDaemon


def _generate_monitoring_hash() -> str:
    """Generate a short hash based on hostname, PID, and thread ID."""
    key = f"{platform.node()}:{os.getpid()}:{threading.get_ident()}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


class MonitoringManager:
    """Owns the full monitoring lifecycle for a single Run.

    Handles console capture, hardware metrics, traceback capture via
    excepthook, and static metadata logging.  Call ``start()`` after
    construction and ``close()`` when the run ends.
    """

    def __init__(
        self,
        run_id: str,
        namespace: str | None,
        log_metrics_fn: Callable[[dict[str, float], int], None],
        log_strings_fn: Callable[[dict[str, str], int], None],
        log_configs_fn: Callable[[dict[str, Any]], None],
        capture_stdout: bool,
        capture_stderr: bool,
        capture_hardware_metrics: bool,
        capture_traceback: bool,
    ) -> None:
        self._run_id = run_id
        self.namespace = (
            namespace or f"monitoring/{_generate_monitoring_hash()}"
        ).rstrip("/")
        self._log_metrics_fn = log_metrics_fn
        self._log_strings_fn = log_strings_fn
        self._log_configs_fn = log_configs_fn
        self._capture_stdout = capture_stdout
        self._capture_stderr = capture_stderr
        self._capture_hardware_metrics = capture_hardware_metrics
        self.capture_traceback = capture_traceback

        self._console_capture = None
        self._hardware_monitor = None
        self._original_excepthook: Any = None

    def start(self) -> None:
        """Log static metadata, start daemons, install excepthook."""
        ns = self.namespace

        # Static metadata
        self._log_configs_fn({
            f"{ns}/hostname": platform.node(),
            f"{ns}/pid": os.getpid(),
            f"{ns}/tid": threading.get_ident(),
        })

        # Console capture
        if self._capture_stdout or self._capture_stderr:
            self._console_capture = ConsoleCaptureDaemon(
                subscriber_id=self._run_id,
                namespace=ns,
                capture_stdout=self._capture_stdout,
                capture_stderr=self._capture_stderr,
                log_fn=self._log_strings_fn,
            )
            self._console_capture.start()

        # Hardware monitor
        if self._capture_hardware_metrics:
            self._hardware_monitor = HardwareMonitorDaemon(
                namespace=ns,
                log_fn=self._log_metrics_fn,
            )
            self._hardware_monitor.start()

        # Traceback capture via excepthook
        if self.capture_traceback:
            self._original_excepthook = sys.excepthook

            def _excepthook(exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
                try:
                    tb_text = "".join(
                        traceback.format_exception(exc_type, exc_value, exc_tb)
                    )
                    self._log_strings_fn({f"{ns}/traceback": tb_text}, 0)
                except Exception:
                    pass
                if self._original_excepthook:
                    self._original_excepthook(exc_type, exc_value, exc_tb)

            sys.excepthook = _excepthook

    def close(self) -> None:
        """Stop daemons and restore excepthook."""
        if self._hardware_monitor is not None:
            self._hardware_monitor.close()
            self._hardware_monitor = None

        if self._console_capture is not None:
            self._console_capture.close()
            self._console_capture = None

        if self._original_excepthook is not None:
            sys.excepthook = self._original_excepthook
            self._original_excepthook = None

    def log_traceback(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Log a formatted traceback as a string series entry."""
        try:
            tb_text = "".join(
                traceback.format_exception(exc_type, exc_val, exc_tb)
            )
            self._log_strings_fn(
                {f"{self.namespace}/traceback": tb_text}, 0
            )
        except Exception:
            pass
