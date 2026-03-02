"""Neptune v2-compatible API backed by goodseed.

Drop-in replacement for ``import neptune`` that delegates to goodseed's
local storage and sync. Features that goodseed doesn't support yet raise
``NotImplementedError``.

Example::

    import goodseed.neptune as neptune

    run = neptune.init_run(name="baseline", mode="async")
    run["params/lr"] = 0.001
    run["train/loss"].log(0.5, step=1)
    run["sys/tags"].add("production")
    run.stop()
"""

from __future__ import annotations

from typing import Any, Optional

from goodseed.neptune._run import NeptuneRun
from goodseed.neptune._stubs import Model, ModelVersion, Project, Table

__all__ = [
    "init_run",
    "init_model",
    "init_model_version",
    "init_project",
    "Run",
    "Model",
    "ModelVersion",
    "Project",
    "Table",
    "ANONYMOUS_API_TOKEN",
]

Run = NeptuneRun

ANONYMOUS_API_TOKEN = "anonymous"


def init_run(
    *,
    project: Optional[str] = None,
    api_token: Optional[str] = None,
    mode: str = "async",
    with_id: Optional[str] = None,
    name: Optional[str] = None,
    custom_id: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[list[str]] = None,
    flush_period: float = 2,
    proxies: Optional[dict[str, str]] = None,
    capture_hardware_metrics: bool = True,
    capture_stdout: bool = True,
    capture_stderr: bool = True,
    capture_traceback: bool = True,
    git_ref: Any = None,
    **kwargs: Any,
) -> NeptuneRun:
    """Create a new run (Neptune v2 ``init_run`` compatible).

    Maps Neptune v2 parameters to ``goodseed.Run`` equivalents:
    - ``api_token`` → ``api_key``
    - ``mode="async"`` → ``storage="cloud"``
    - ``mode="read-only"`` → ``storage="cloud", read_only=True``
    - ``mode="debug"`` / ``mode="offline"`` → ``storage="local"``
    - ``with_id`` → ``resume_run_id``
    - ``custom_id`` → ``run_id``
    """
    if mode == "async":
        gs_storage = "cloud"
    elif mode == "read-only":
        gs_storage = "cloud"
    else:
        gs_storage = "local"
    read_only = mode == "read-only"

    return NeptuneRun(
        project=project,
        api_key=api_token,
        run_id=custom_id,
        resume_run_id=with_id,
        name=name,
        description=description,
        tags=tags,
        storage=gs_storage,
        read_only=read_only,
        capture_hardware_metrics=capture_hardware_metrics,
        capture_stdout=capture_stdout,
        capture_stderr=capture_stderr,
        capture_traceback=capture_traceback,
        git_ref=git_ref,
    )


def init_model(
    *,
    with_id: Optional[str] = None,
    name: Optional[str] = None,
    key: Optional[str] = None,
    project: Optional[str] = None,
    api_token: Optional[str] = None,
    mode: str = "async",
    **kwargs: Any,
) -> Model:
    """Create or resume a model. Not yet supported by goodseed."""
    raise NotImplementedError("goodseed does not support init_model() yet")


def init_model_version(
    *,
    with_id: Optional[str] = None,
    name: Optional[str] = None,
    model: Optional[str] = None,
    project: Optional[str] = None,
    api_token: Optional[str] = None,
    mode: str = "async",
    **kwargs: Any,
) -> ModelVersion:
    """Create or resume a model version. Not yet supported by goodseed."""
    raise NotImplementedError(
        "goodseed does not support init_model_version() yet"
    )


def init_project(
    *,
    project: Optional[str] = None,
    api_token: Optional[str] = None,
    mode: str = "read-only",
    proxies: Optional[dict[str, str]] = None,
    **kwargs: Any,
) -> Project:
    """Connect to a project for querying. Not yet supported by goodseed."""
    raise NotImplementedError("goodseed does not support init_project() yet")
