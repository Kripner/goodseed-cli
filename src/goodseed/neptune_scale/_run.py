"""Neptune Scale Run adapter.

Wraps ``goodseed.Run`` to expose the Neptune Scale (v3) API surface.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Optional

from goodseed.run import Run


class ScaleRun:
    """Neptune Scale-compatible Run backed by goodseed storage.

    Accepts Neptune Scale's constructor parameters and maps them to
    ``goodseed.Run`` equivalents. In ``disabled`` mode, all methods
    are no-ops.
    """

    def __init__(
        self,
        *,
        project: Optional[str] = None,
        api_token: Optional[str] = None,
        run_id: Optional[str] = None,
        resume: bool = False,
        mode: str = "async",
        experiment_name: Optional[str] = None,
        creation_time: Optional[datetime] = None,
        # Callbacks — accepted for API compatibility, ignored
        on_error_callback: Optional[Callable[..., Any]] = None,
        on_warning_callback: Optional[Callable[..., Any]] = None,
        on_network_error_callback: Optional[Callable[..., Any]] = None,
        on_async_lag_callback: Optional[Callable[..., Any]] = None,
        async_lag_threshold: Optional[float] = None,
        # Monitoring
        capture_hardware_metrics: bool = True,
        capture_stdout: bool = True,
        capture_stderr: bool = True,
        capture_traceback: bool = True,
        **kwargs: Any,
    ) -> None:
        self._disabled = mode == "disabled"
        if self._disabled:
            self._run: Run | None = None
            return

        if mode == "async":
            gs_storage = "cloud"
        else:
            gs_storage = "local"
        resume_run_id = run_id if resume else None
        new_run_id = run_id if not resume else None

        self._run = Run(
            project=project,
            api_key=api_token,
            run_id=new_run_id,
            resume_run_id=resume_run_id,
            name=experiment_name,
            storage=gs_storage,
            created_at=creation_time.isoformat() if creation_time else None,
            capture_hardware_metrics=capture_hardware_metrics,
            capture_stdout=capture_stdout,
            capture_stderr=capture_stderr,
            capture_traceback=capture_traceback,
        )

    def _check_active(self) -> Run:
        if self._disabled or self._run is None:
            raise RuntimeError("Run is disabled or not initialized")
        return self._run

    # --- Implemented methods ---

    def log_metrics(
        self,
        data: dict[str, float],
        step: int | float | None = None,
        *,
        timestamp: Optional[datetime] = None,
        preview: bool = False,
    ) -> None:
        """Log metric values.

        *timestamp* and *preview* are accepted for API compatibility but
        ignored by goodseed.
        """
        if self._disabled:
            return
        self._check_active().log_metrics(data, step=step)

    def log_configs(
        self,
        data: dict[str, Any],
    ) -> None:
        """Log configuration values."""
        if self._disabled:
            return
        self._check_active().log_configs(data)

    def log_string_series(
        self,
        data: dict[str, str],
        step: int | float | None = None,
        *,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Log string series values.

        *timestamp* is accepted for API compatibility but ignored.
        """
        if self._disabled:
            return
        self._check_active().log_strings(data, step=step)

    def add_tags(
        self,
        tags: str | list[str],
        *,
        group_tags: bool = False,
    ) -> None:
        """Add tags to the run.

        *group_tags* is accepted for API compatibility but ignored.
        """
        if self._disabled:
            return
        run = self._check_active()
        run._add_to_string_set("sys/tags", tags)

    def remove_tags(
        self,
        tags: str | list[str],
        *,
        group_tags: bool = False,
    ) -> None:
        """Remove tags from the run.

        *group_tags* is accepted for API compatibility but ignored.
        """
        if self._disabled:
            return
        run = self._check_active()
        run._remove_from_string_set("sys/tags", tags)

    def close(self, timeout: float | None = None) -> None:
        """Close the run.

        Args:
            timeout: Ignored (goodseed drains all data on close).
        """
        if self._disabled or self._run is None:
            return
        self._run.close()

    def terminate(self) -> None:
        """Immediately stop the run with ``failed`` status."""
        if self._disabled or self._run is None:
            return
        self._run.close(status="failed")

    def wait_for_processing(self, timeout: float | None = None) -> None:
        """Block until all data has been synced to the remote API.

        Args:
            timeout: Maximum seconds to wait. None means wait indefinitely.
        """
        if self._disabled or self._run is None:
            return
        self._run.wait(timeout=timeout)

    # --- Not implemented ---

    def log_files(self, **kwargs: Any) -> None:
        """Log files. Not yet supported by goodseed."""
        raise NotImplementedError(
            "goodseed does not support log_files() yet"
        )

    def assign_files(self, **kwargs: Any) -> None:
        """Assign files. Not yet supported by goodseed."""
        raise NotImplementedError(
            "goodseed does not support assign_files() yet"
        )

    def log_histograms(self, **kwargs: Any) -> None:
        """Log histograms. Not yet supported by goodseed."""
        raise NotImplementedError(
            "goodseed does not support log_histograms() yet"
        )

    # --- Context manager ---

    def __enter__(self) -> ScaleRun:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._run is not None:
            status = "failed" if exc_type is not None else "finished"
            self._run.close(status=status)
