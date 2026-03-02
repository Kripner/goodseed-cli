"""Goodseed - ML experiment tracker.

Example:
    >>> import goodseed
    >>> run = goodseed.Run(name="My Experiment", tags=["bert"])
    >>> run["learning_rate"] = 0.001
    >>> run["train/loss"].log(0.5, step=1)
    >>> run.close()
"""

from goodseed.run import GitRef, Run, Storage
from goodseed.projects import (
    ensure_project,
    list_projects,
    list_runs,
    list_workspaces,
    me,
)

__version__ = "0.3.0"
__all__ = [
    "Run",
    "GitRef",
    "Storage",
    "list_workspaces",
    "list_projects",
    "ensure_project",
    "list_runs",
    "me",
]
