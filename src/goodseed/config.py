"""Configuration management for Goodseed.

Handles environment variables and path resolution.
"""

import os
from pathlib import Path
from typing import Optional, Union

# Environment variable names
ENV_HOME = "GOODSEED_HOME"
ENV_PROJECT = "GOODSEED_PROJECT"

# Defaults
DEFAULT_HOME = Path.home() / ".goodseed"
DEFAULT_PROJECT = "default"


def get_home(override: Optional[Union[str, Path]] = None) -> Path:
    """Get the Goodseed home directory.

    Args:
        override: If provided, use this path instead of env/default.
    """
    if override:
        return Path(override)
    home = os.environ.get(ENV_HOME)
    if home:
        return Path(home)
    return DEFAULT_HOME


def get_projects_dir(home_override: Optional[Union[str, Path]] = None) -> Path:
    """Get the directory for project run databases."""
    return get_home(home_override) / "projects"


def get_default_project() -> str:
    """Get the default project name."""
    return os.environ.get(ENV_PROJECT, DEFAULT_PROJECT)


def get_run_db_path(
    project: str,
    run_name: str,
    home_override: Optional[Union[str, Path]] = None,
) -> Path:
    """Get the path for a run's SQLite database."""
    return get_projects_dir(home_override) / project / "runs" / f"{run_name}.sqlite"


def ensure_dir(path: Path) -> None:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)
