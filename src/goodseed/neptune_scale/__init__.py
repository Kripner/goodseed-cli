"""Neptune Scale (v3)-compatible API backed by goodseed.

Drop-in replacement for ``import neptune_scale`` that delegates to
goodseed's local storage and sync. Features that goodseed doesn't
support yet raise ``NotImplementedError``.

Example::

    import goodseed.neptune_scale as neptune_scale

    run = neptune_scale.Run(project="workspace/project", run_id="my-run")
    run.log_metrics({"loss": 0.5}, step=1)
    run.log_configs({"lr": 0.001})
    run.add_tags(["baseline"])
    run.close()
"""

from __future__ import annotations

from typing import Any, Optional

from goodseed.neptune_scale._run import ScaleRun
from goodseed.neptune_scale._types import File, Histogram

__all__ = [
    "Run",
    "File",
    "Histogram",
    "create_project",
    "list_projects",
]

Run = ScaleRun


def create_project(
    name: Optional[str] = None,
    *,
    workspace: Optional[str] = None,
    visibility: str = "private",
    api_token: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Create a project. Not yet supported by goodseed."""
    raise NotImplementedError("goodseed does not support create_project() yet")


def list_projects(
    *,
    api_token: Optional[str] = None,
    **kwargs: Any,
) -> list[Any]:
    """List projects. Not yet supported by goodseed."""
    raise NotImplementedError("goodseed does not support list_projects() yet")
