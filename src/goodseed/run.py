"""Run class for experiment tracking.

The Run object is the main interface for logging experiment data.
All data is written to a local SQLite database that persists permanently.
A local HTTP server (goodseed serve) reads these files for visualization.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

from enum import Enum

from goodseed.config import (
    API_BASE,
    APP_URL,
    ENV_RUN_ID,
    ENV_STORAGE,
    get_api_key,
    get_default_project,
    get_run_db_path,
)
from goodseed.monitoring.manager import MonitoringManager
from goodseed.storage import LocalStorage
from goodseed.sync import SyncProcess, _api_get, _ensure_run, _ensure_run_once, api_get_json
from goodseed.utils import (
    deserialize_value,
    flatten_dict,
    generate_run_id,
    normalize_path,
    serialize_value,
)

import re

Number = Union[int, float]

_MAX_RUN_ID_LEN = 128
_MAX_EXPERIMENT_NAME_LEN = 200
_RUN_ID_RE = re.compile(r'^[a-zA-Z0-9]([a-zA-Z0-9_-]*[a-zA-Z0-9])?$')


def _has_dangerous_experiment_name_chars(value: str) -> bool:
    """Return True if the name contains disallowed control/spoofing chars."""
    for ch in value:
        code = ord(ch)

        # Ban control characters (including NUL and C1 control block).
        if (0x00 <= code <= 0x1F) or (0x7F <= code <= 0x9F):
            return True

        # Ban bidi override/isolate chars to prevent visual spoofing.
        if (0x202A <= code <= 0x202E) or (0x2066 <= code <= 0x2069):
            return True

        # Ban zero-width formatting chars that make labels ambiguous.
        if code in {0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0xFEFF}:
            return True

    return False


def _use_color() -> bool:
    """Check if stderr supports ANSI color codes."""
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


def _goodseed_error(headline: str, detail: str = "") -> RuntimeError:
    """Build a RuntimeError with a formatted [Goodseed] message.

    Adds ANSI bold+red when stderr is a TTY.
    """
    if _use_color():
        head = f"\033[1;31m[Goodseed]\033[0m \033[1m{headline}\033[0m"
    else:
        head = f"[Goodseed] {headline}"
    parts = ["\n", head]
    if detail:
        parts.append(detail)
    parts.append("")  # trailing newline
    return RuntimeError("\n".join(parts))


def _resolve_default_workspace(api_key: str) -> str:
    """Fetch the user's default workspace from the /me endpoint."""
    status, payload = api_get_json("/api/v1/auth/me", api_key=api_key)
    if status == 0:
        raise _goodseed_error(
            "Could not reach the Goodseed backend.",
            "Check your internet connection and try again.\n"
            "If you are behind a proxy, make sure HTTPS_PROXY is set.",
        )
    if status == 200 and isinstance(payload, dict):
        name = payload.get("name") or payload.get("Name")
        if name:
            return str(name)
    raise _goodseed_error(
        "Could not determine your default workspace.",
        "Specify the full project: Run(project='workspace/project')\n"
        "Or check that your API key is valid.",
    )


def _filter_paths_by_type(payload: list[dict[str, Any]], path_type: str) -> list[str]:
    """Extract path names from merged /paths payload by type."""
    return [
        str(item["path"])
        for item in payload
        if isinstance(item, dict) and item.get("type") == path_type and "path" in item
    ]


class Storage(str, Enum):
    """Storage mode for a ``Run``.

    - ``disabled``: No resources created. Writes are silent no-ops, reads raise.
    - ``local``: Local SQLite storage only, no remote sync.
    - ``cloud``: Local storage plus background sync to the remote API.
    """

    DISABLED = "disabled"
    LOCAL = "local"
    CLOUD = "cloud"


def _is_namespace(value: Any) -> bool:
    """Check if value is an argparse.Namespace via duck typing."""
    return type(value).__name__ == "Namespace" and hasattr(value, "__dict__")


class _DisabledGitRefType:
    """Sentinel used to disable Git tracking."""


class GitRef:
    """Git tracking configuration for ``Run``.

    Args:
        repository_path: Path to a Git repository root or a path
            inside that repository.
    """

    DISABLED = _DisabledGitRefType()

    def __init__(self, repository_path: str | Path | None = None):
        self.repository_path = repository_path


def _resolve_db_path(
    run_id: str,
    project: str,
    auto_name: bool,
    log_dir: str | Path | None = None,
    goodseed_home: str | Path | None = None,
) -> tuple:
    """Resolve a unique (run_id, db_path) pair.

    For auto-generated names, appends -2, -3, etc. on collision.
    For explicit names, raises if the file already exists.

    Returns (run_id, db_path).
    """
    if "/" in run_id or "\\" in run_id:
        raise ValueError(f"run_id must not contain path separators: {run_id!r}")

    def _path_for(name: str) -> Path:
        if log_dir:
            return Path(log_dir) / f"{name}.sqlite"
        return get_run_db_path(project, name, goodseed_home)

    db_path = _path_for(run_id)
    if not db_path.exists():
        return run_id, db_path

    if not auto_name:
        raise RuntimeError(
            f"Database already exists: {db_path}\n"
            f"Choose a different run_id or delete the file:\n"
            f"  rm {db_path}"
        )

    base = run_id
    for i in range(2, 1000):
        candidate = f"{base}-{i}"
        db_path = _path_for(candidate)
        if not db_path.exists():
            return candidate, db_path

    raise RuntimeError("Could not find a unique run ID after retries")


def _find_db_path(
    run_id: str,
    project: str,
    log_dir: str | Path | None = None,
    goodseed_home: str | Path | None = None,
) -> Path:
    """Find the DB path for an existing run, or raise."""
    if "/" in run_id or "\\" in run_id:
        raise ValueError(f"run_id must not contain path separators: {run_id!r}")

    if log_dir:
        db_path = Path(log_dir) / f"{run_id}.sqlite"
    else:
        db_path = get_run_db_path(project, run_id, goodseed_home)

    if not db_path.exists():
        raise RuntimeError(
            f"Cannot resume run '{run_id}': database not found at {db_path}"
        )
    return db_path


def _run_git_command(cwd: Path, args: list[str]) -> str | None:
    """Run a git command and return stripped stdout, or None on failure."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if proc.returncode != 0:
        return None
    return proc.stdout.rstrip("\n")


def _resolve_git_repo_root(start_path: str | Path | None) -> Path | None:
    """Resolve a Git repository root from a start path or current working dir."""
    if shutil.which("git") is None:
        return None

    base = Path(start_path).expanduser() if start_path else Path.cwd()
    if base.is_file():
        base = base.parent

    base = base.resolve()
    repo_root = _run_git_command(base, ["rev-parse", "--show-toplevel"])
    if not repo_root:
        return None
    return Path(repo_root)


def _collect_git_configs(git_ref: bool | GitRef | _DisabledGitRefType | None) -> dict[str, Any]:
    """Collect Git metadata and diffs for run configs."""
    if git_ref is False or git_ref is GitRef.DISABLED:
        return {}

    repo_hint: str | Path | None = None
    if isinstance(git_ref, GitRef):
        repo_hint = git_ref.repository_path

    repo_root = _resolve_git_repo_root(repo_hint)
    if repo_root is None:
        return {}

    data: dict[str, Any] = {
        "source_code/git/repository_path": str(repo_root),
    }

    dirty_out = _run_git_command(repo_root, ["status", "--porcelain"])
    if dirty_out is not None:
        data["source_code/git/dirty"] = bool(dirty_out.strip())

    diff = _run_git_command(repo_root, ["diff", "HEAD"])
    if diff is not None:
        data["source_code/diff"] = diff

    commit_id = _run_git_command(repo_root, ["log", "-1", "--pretty=%H"])
    if commit_id:
        data["source_code/git/commit_id"] = commit_id

    commit_message = _run_git_command(repo_root, ["log", "-1", "--pretty=%B"])
    if commit_message:
        data["source_code/git/commit_message"] = commit_message

    commit_author = _run_git_command(repo_root, ["log", "-1", "--pretty=%an"])
    if commit_author:
        data["source_code/git/commit_author"] = commit_author

    commit_date = _run_git_command(repo_root, ["log", "-1", "--pretty=%cI"])
    if commit_date:
        data["source_code/git/commit_date"] = commit_date

    branch = _run_git_command(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    if branch:
        data["source_code/git/current_branch"] = branch

    remotes = _run_git_command(repo_root, ["remote", "-v"])
    if remotes:
        data["source_code/git/remotes"] = remotes

    upstream_name = _run_git_command(
        repo_root,
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
    )
    if upstream_name:
        head_sha = _run_git_command(repo_root, ["rev-parse", "HEAD"])
        upstream_sha = _run_git_command(repo_root, ["rev-parse", "@{u}"])
        if head_sha and upstream_sha and head_sha != upstream_sha:
            upstream_diff = _run_git_command(repo_root, ["diff", "@{u}"])
            if upstream_diff is not None:
                data[f"source_code/diff_upstream_{upstream_sha}"] = upstream_diff

    return data


class _FieldHandler:
    """Handler for bracket-style field access: ``run["key"].log(value)``.

    Returned by ``Run.__getitem__``. Supports ``.log()`` for logging
    metric or string series values, plus ``.add()`` / ``.remove()`` for
    string sets (tags).
    """

    __slots__ = ("_run", "_path")

    def __init__(self, run: Run, path: str):
        self._run = run
        self._path = path

    def log(self, value: Any, *, step: Number) -> None:
        """Log a value to this series.

        Numeric values (int, float) are logged as metrics.
        String values are logged as string series.

        Args:
            value: The value to append.
            step: The step number. Steps can be in any order.
                Logging the same step + path overwrites the previous value.
        """
        self._run._append_to_field(self._path, value, step)

    def add(self, value: str | list[str]) -> None:
        """Add one or more values to a string set field (e.g. tags).

        Args:
            value: A single string or list of strings to add.

        Example::

            run["sys/tags"].add("production")
            run["sys/tags"].add(["v2", "bert"])
        """
        self._run._add_to_string_set(self._path, value)

    def remove(self, value: str | list[str]) -> None:
        """Remove one or more values from a string set field (e.g. tags)."""
        self._run._remove_from_string_set(self._path, value)


class Run:
    """An experiment run for logging metrics and configs.

    Data is written to a local SQLite file that persists after the run closes.
    Use ``goodseed serve`` to visualize runs in the browser.

    Args:
        name: Display name for this run (shown in the UI as ``sys/name``).
        project: Full project name in ``workspace/project`` format
            (defaults to GOODSEED_PROJECT or 'default'). When cloud storage
            is enabled, the project name identifies the workspace and project
            on the server. Locally, the slash maps to the filesystem path.
        run_id: Unique run identifier. Falls back to GOODSEED_RUN_ID env var,
            then auto-generated if not provided.
        resume_run_id: Resume an existing run by its ID. Mutually exclusive
            with run_id.
        description: Free-form text description of the run.
        tags: Initial tags for the run (stored as ``sys/tags``).
        storage: Storage mode. One of ``"disabled"``, ``"local"``, or
            ``"cloud"`` (see ``Storage`` enum). Falls back to
            ``GOODSEED_STORAGE`` env var, then defaults to ``"cloud"``.

            - ``disabled`` — no resources created; writes are silent no-ops;
              reads raise.
            - ``local`` — local SQLite only, no remote sync.
            - ``cloud`` — local SQLite plus background sync to the remote API.
        api_key: API key for cloud storage. Falls back to
            ``GOODSEED_API_KEY``.
        read_only: If True, write methods raise ``RuntimeError``. The
            behavior depends on the storage mode:

            - ``cloud`` — the run is resolved on the remote API and data
              fetching methods (``get_metric_data``, etc.) become available.
              No sync daemon is started.
            - ``local`` — an existing local database is opened for reading.
              Requires ``run_id``.
            - ``disabled`` — both reads and writes raise.
        goodseed_home: Override for GOODSEED_HOME.
        log_dir: Override directory for the run database file.
            If provided, the database is stored at ``log_dir/{run_id}.sqlite``.
            Otherwise uses ``~/.goodseed/projects/{project}/{run_id}.sqlite``.
        capture_hardware_metrics: Log CPU, memory, and GPU metrics.
        capture_stdout: Capture stdout output.
        capture_stderr: Capture stderr output.
        capture_traceback: Capture traceback on unhandled exceptions.
        monitoring_namespace: Custom monitoring namespace path.
            Defaults to ``monitoring/<hash>`` where hash is process-unique.
        git_ref: Git tracking behavior. By default, Git metadata is logged
            automatically for the repository containing the current working
            directory. Pass ``False`` or ``GitRef.DISABLED`` to disable.
            Use ``GitRef(repository_path=...)`` to specify a custom repository.
    """

    def __init__(
        self,
        *,
        project: str | None = None,
        run_id: str | None = None,
        resume_run_id: str | None = None,
        goodseed_home: str | Path | None = None,
        log_dir: str | Path | None = None,
        created_at: str | None = None,
        modified_at: str | None = None,
        capture_hardware_metrics: bool = True,
        capture_stdout: bool = True,
        capture_stderr: bool = True,
        capture_traceback: bool = True,
        monitoring_namespace: str | None = None,
        git_ref: bool | GitRef | _DisabledGitRefType | None = None,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        storage: str | Storage | None = None,
        api_key: str | None = None,
        read_only: bool = False,
        **kwargs: Any,
    ):
        legacy_experiment_name = kwargs.pop("experiment_name", None)
        if legacy_experiment_name is not None:
            if name is not None:
                raise ValueError(
                    "Cannot specify both 'name' and deprecated alias "
                    "'experiment_name'. Use 'name' only."
                )
            warnings.warn(
                "'experiment_name' is deprecated and will be removed in a "
                "future release; use 'name' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            name = legacy_experiment_name

        legacy_run_name = kwargs.pop("run_name", None)
        if legacy_run_name is not None:
            if run_id is not None:
                raise ValueError(
                    "Cannot specify both 'run_id' and deprecated alias "
                    "'run_name'. Use 'run_id' only."
                )
            warnings.warn(
                "'run_name' is deprecated and will be removed in a future "
                "release; use 'run_id' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            run_id = legacy_run_name

        if kwargs:
            unexpected = sorted(kwargs)
            if len(unexpected) == 1:
                raise TypeError(
                    f"Run.__init__() got an unexpected keyword argument "
                    f"'{unexpected[0]}'"
                )
            raise TypeError(
                "Run.__init__() got unexpected keyword arguments: "
                + ", ".join(f"'{name}'" for name in unexpected)
            )

        self.project = project or get_default_project()
        self.name = name
        self._read_only = read_only
        self._lock = threading.RLock()
        self._closed = False
        self._auto_step: dict[str, float] = {}  # per-path auto-increment counters

        if storage is None:
            storage = os.environ.get(ENV_STORAGE, Storage.CLOUD)
        self._storage_mode = Storage(storage)

        if self._storage_mode == Storage.CLOUD and "/" not in self.project:
            resolved_key = api_key or get_api_key()
            if resolved_key:
                workspace = _resolve_default_workspace(resolved_key)
                self.project = f"{workspace}/{self.project}"

        if self._storage_mode == Storage.DISABLED:
            self._init_disabled()
            return

        if read_only:
            if self._storage_mode == Storage.CLOUD:
                self._init_cloud_read_only(api_key, run_id)
            else:
                self._init_local_read_only(run_id, goodseed_home, log_dir)
            return

        if run_id and resume_run_id:
            raise ValueError("Cannot specify both run_id and resume_run_id")

        resuming = resume_run_id is not None
        if resuming:
            self._init_resume(resume_run_id, goodseed_home, log_dir)
        else:
            self._init_new_run(
                run_id, goodseed_home, log_dir, created_at, modified_at,
                name, description, tags, git_ref,
            )

        self._sync_process = None
        self._api_key = None
        self._remote_id = None
        self._sync_url = None
        if self._storage_mode == Storage.CLOUD:
            self._init_remote_sync(api_key)

        self._init_monitoring(
            capture_hardware_metrics, capture_stdout,
            capture_stderr, capture_traceback, monitoring_namespace,
        )
        atexit.register(self._cleanup)

        if resuming:
            print(f"Goodseed run resumed: {self.run_id}")
        else:
            print(f"Goodseed run: {self.run_id}")
        print(f"  Data: {self._db_path}")
        if self._sync_url:
            print(f"  Sync: {self._sync_url}")

    def _init_disabled(self) -> None:
        """Set up a disabled run: no storage, no sync, no monitoring."""
        self.run_id = ""
        self._storage = None
        self._db_path = None
        self._sync_process = None
        self._sync_url = None
        self._monitoring = None
        self._api_key = None
        self._remote_id = None

    def _init_cloud_read_only(
        self,
        api_key: str | None,
        run_id: str | None,
    ) -> None:
        """Set up a read-only run backed by the remote API."""
        resolved_key = api_key or get_api_key()
        if not resolved_key:
            raise _goodseed_error(
                "API key required.",
                "Set GOODSEED_API_KEY or pass api_key= to Run().",
            )
        parts = self.project.split("/", 1)
        if len(parts) != 2:
            raise _goodseed_error(
                f"Invalid project format: {self.project!r}",
                "Use 'workspace/project' format, e.g. Run(project='my-team/my-project').",
            )
        self._api_key = resolved_key
        remote_id, error_msg, _ = _ensure_run_once(
            resolved_key, parts[0], parts[1], run_id or "", self.name,
            log_errors=False,
        )
        if remote_id is None:
            raise _goodseed_error(error_msg)
        self._remote_id = remote_id
        self.run_id = run_id or ""
        self._storage = None
        self._db_path = None
        self._sync_process = None
        self._sync_url = None
        self._monitoring = None

    def _init_local_read_only(
        self,
        run_id: str | None,
        goodseed_home: str | Path | None,
        log_dir: str | Path | None,
    ) -> None:
        """Set up a read-only run from local storage."""
        if not run_id:
            raise RuntimeError(
                "read_only with storage='local' requires a run_id"
            )
        db_path = _find_db_path(run_id, self.project, log_dir, goodseed_home)
        self._storage = LocalStorage(db_path)
        self._db_path = db_path
        self.run_id = run_id
        self._sync_process = None
        self._sync_url = None
        self._monitoring = None
        self._api_key = None
        self._remote_id = None

    def _init_resume(
        self,
        resume_run_id: str,
        goodseed_home: str | Path | None,
        log_dir: str | Path | None,
    ) -> None:
        """Resume an existing run: find DB, verify status, restore auto-steps."""
        db_path = _find_db_path(
            resume_run_id, self.project, log_dir, goodseed_home,
        )
        self._storage = LocalStorage(db_path)
        self._db_path = db_path

        # Check the run is not currently running
        status = self._storage.get_meta("status")
        if status == "running":
            self._storage.close()
            raise RuntimeError(
                f"Cannot resume run '{resume_run_id}': it is still running"
            )

        self.run_id = resume_run_id

        # Mark as running again
        self._storage.set_meta("status", "running")
        self.log_configs({"sys/state": "running"})

    def _init_new_run(
        self,
        run_id: str | None,
        goodseed_home: str | Path | None,
        log_dir: str | Path | None,
        created_at: str | None,
        modified_at: str | None,
        name: str | None,
        description: str | None,
        tags: list[str] | None,
        git_ref: bool | GitRef | _DisabledGitRefType | None,
    ) -> None:
        """Create a brand-new run: resolve ID, create storage, write metadata."""
        effective_id = run_id or os.environ.get(ENV_RUN_ID) or generate_run_id()
        auto_name = run_id is None and not os.environ.get(ENV_RUN_ID)

        # Validate user-provided run IDs for server compatibility.
        if not auto_name:
            if len(effective_id) > _MAX_RUN_ID_LEN:
                raise ValueError(
                    f"run_id must be at most {_MAX_RUN_ID_LEN} characters, "
                    f"got {len(effective_id)}"
                )
            if not _RUN_ID_RE.match(effective_id):
                raise ValueError(
                    f"run_id must contain only letters, digits, hyphens, "
                    f"and underscores, and start/end with a letter or digit: "
                    f"{effective_id!r}"
                )

        if name is not None and len(name) > _MAX_EXPERIMENT_NAME_LEN:
            raise ValueError(
                f"name must be at most {_MAX_EXPERIMENT_NAME_LEN} characters, "
                f"got {len(name)}"
            )
        if name is not None and _has_dangerous_experiment_name_chars(name):
            raise ValueError(
                "name must not contain control, bidirectional override/isolate, "
                "or zero-width formatting characters"
            )

        self.run_id, db_path = _resolve_db_path(
            run_id=effective_id,
            project=self.project,
            auto_name=auto_name,
            log_dir=log_dir,
            goodseed_home=goodseed_home,
        )

        self._storage = LocalStorage(db_path)
        self._db_path = db_path
        self._storage.set_meta("run_id", self.run_id)
        self._storage.set_meta("project", self.project)
        self._storage.set_meta(
            "created_at", created_at or datetime.now(timezone.utc).isoformat()
        )
        if modified_at:
            self._storage.set_meta("modified_at", modified_at)
        self._storage.set_meta("status", "running")
        if self.name:
            self._storage.set_meta("name", self.name)

        created = created_at or datetime.now(timezone.utc).isoformat()
        sys_configs: dict[str, Any] = {
            "sys/id": self.run_id,
            "sys/creation_time": created,
            "sys/state": "running",
        }
        if self.name:
            sys_configs["sys/name"] = self.name
        if description:
            sys_configs["sys/description"] = description
        if tags:
            sys_configs["sys/tags"] = set(tags)
        self.log_configs(sys_configs)

        # Git tracking metadata (best-effort, never fail run creation).
        try:
            git_data = _collect_git_configs(git_ref)
            if git_data:
                self.log_configs(git_data)
        except Exception:
            pass

    def _init_remote_sync(self, api_key: str | None) -> None:
        """Start the background sync process for cloud storage."""
        resolved_key = api_key or get_api_key()
        if not resolved_key:
            raise _goodseed_error(
                "API key required.",
                "Set GOODSEED_API_KEY or pass api_key= to Run().\n"
                "To skip cloud sync, use storage='local'.",
            )
        parts = self.project.split("/", 1)
        if len(parts) != 2:
            raise _goodseed_error(
                f"Invalid project format: {self.project!r}",
                "Use 'workspace/project' format, e.g. Run(project='my-team/my-project').",
            )
        self._api_key = resolved_key
        remote_id, error_msg, _ = _ensure_run_once(
            resolved_key, parts[0], parts[1], self.run_id, self.name,
            log_errors=False,
        )
        if remote_id is None:
            raise _goodseed_error(error_msg)
        self._remote_id = remote_id
        self._sync_process = SyncProcess(
            db_path=self._db_path,
            api_key=resolved_key,
            workspace=parts[0],
            project_name=parts[1],
            run_id=self.run_id,
            experiment_name=self.name,
        )
        self._sync_process.start()
        self._sync_url = API_BASE

    def _init_monitoring(
        self,
        capture_hardware_metrics: bool,
        capture_stdout: bool,
        capture_stderr: bool,
        capture_traceback: bool,
        monitoring_namespace: str | None,
    ) -> None:
        """Create and start the monitoring manager (or set None if disabled)."""
        monitoring_enabled = (
            capture_hardware_metrics or capture_stdout
            or capture_stderr or capture_traceback
        )
        if not monitoring_enabled:
            self._monitoring = None
            return

        self._monitoring = MonitoringManager(
            run_id=self.run_id,
            namespace=monitoring_namespace,
            log_metrics_fn=self._log_metrics_internal,
            log_strings_fn=self._log_strings_internal,
            log_configs_fn=self.log_configs,
            capture_stdout=capture_stdout,
            capture_stderr=capture_stderr,
            capture_hardware_metrics=capture_hardware_metrics,
            capture_traceback=capture_traceback,
        )
        self._monitoring.start()

    def _log_strings_internal(
        self, data: dict[str, str], step: int
    ) -> None:
        """Log string series without checking closed state (for monitoring)."""
        with self._lock:
            if self._closed or self._storage is None:
                return
            ts = int(datetime.now(timezone.utc).timestamp())
            points = []
            for k, v in data.items():
                path = normalize_path(k)
                points.append((path, step, str(v), ts))
            self._storage.log_string_points(points)

    def _log_metrics_internal(
        self, data: dict[str, float], step: int
    ) -> None:
        """Log metrics without checking closed state (for monitoring)."""
        with self._lock:
            if self._closed or self._storage is None:
                return
            ts = int(datetime.now(timezone.utc).timestamp())
            points = []
            for k, v in data.items():
                path = normalize_path(k)
                points.append((path, step, float(v), ts))
            self._storage.log_metric_points(points)

    def log_configs(
        self,
        data: dict[str, Any],
        flatten: bool = True,
    ) -> None:
        """Log configuration values.

        Args:
            data: Dictionary of path -> value mappings.
            flatten: If True, flatten nested dictionaries.
        """
        with self._lock:
            if self._read_only:
                raise RuntimeError("Run is read-only")
            if self._storage_mode == Storage.DISABLED:
                return
            if self._closed:
                raise RuntimeError("Run is closed")

            if flatten:
                flattened: dict[str, Any] = {}
                for k, v in data.items():
                    if isinstance(v, dict):
                        # Flatten nested dicts under their top-level key while
                        # preserving non-dict types (e.g. string_set values).
                        flattened.update(flatten_dict({k: v}, cast_unsupported=True))
                    else:
                        flattened[k] = v
                data = flattened

            serialized = {}
            for k, v in data.items():
                path = normalize_path(k)
                type_tag, value = serialize_value(v)
                serialized[path] = (type_tag, value)

            self._storage.log_configs(serialized)

    def __getitem__(self, key: str) -> _FieldHandler:
        """Access a field for appending series values or managing string sets.

        Example::

            run["train/loss"].log(0.5, step=1)
            run["train/loss"].log(0.3, step=10)
            run["sys/tags"].add(["bert", "production"])
            run["sys/tags"].remove("bert")
        """
        return _FieldHandler(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a config value or namespace of values.

        Scalar values are stored as a single config entry.
        Dictionaries and argparse Namespaces are flattened under the key
        as a namespace prefix.

        Example::

            run["score"] = 0.97
            run["parameters"] = {"lr": 0.001, "batch_size": 32}
            run["parameters"] = argparse.Namespace(lr=0.001)
        """
        if isinstance(value, dict):
            self.log_configs({key: value}, flatten=True)
        elif _is_namespace(value):
            self.log_configs({key: vars(value)}, flatten=True)
        else:
            self.log_configs({key: value})

    def _append_to_field(
        self, key: str, value: Any, step: Number | None = None
    ) -> None:
        """Append a value to a metric or string series field."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self.log_metrics({key: value}, step=step)
        elif isinstance(value, str):
            self.log_strings({key: value}, step=step)
        else:
            raise TypeError(
                f"Unsupported value type for append: {type(value).__name__}. "
                "Use int, float, or str."
            )

    def _add_to_string_set(self, key: str, value: str | list[str]) -> None:
        """Add values to a string set config field (e.g. sys/tags)."""
        with self._lock:
            if self._read_only:
                raise RuntimeError("Run is read-only")
            if self._storage_mode == Storage.DISABLED:
                return
            if self._closed:
                raise RuntimeError("Run is closed")

            path = normalize_path(key)
            new_tags = [value] if isinstance(value, str) else list(value)

            existing = self._storage.get_config(path)
            if existing is None:
                current = set()
            elif existing[0] == "string_set":
                current = set(json.loads(existing[1]))
            else:
                raise TypeError(
                    f"Cannot add string-set values at '{path}': "
                    f"existing type is '{existing[0]}'."
                )

            current.update(new_tags)
            type_tag, serialized = serialize_value(current)
            self._storage.log_configs({path: (type_tag, serialized)})

    def _remove_from_string_set(self, key: str, value: str | list[str]) -> None:
        """Remove values from a string set config field (e.g. sys/tags)."""
        with self._lock:
            if self._read_only:
                raise RuntimeError("Run is read-only")
            if self._storage_mode == Storage.DISABLED:
                return
            if self._closed:
                raise RuntimeError("Run is closed")

            path = normalize_path(key)
            to_remove = [value] if isinstance(value, str) else list(value)

            existing = self._storage.get_config(path)
            if existing is None:
                return
            if existing[0] != "string_set":
                raise TypeError(
                    f"Cannot remove string-set values at '{path}': "
                    f"existing type is '{existing[0]}'."
                )

            current = set(json.loads(existing[1]))
            for tag in to_remove:
                current.discard(tag)
            type_tag, serialized = serialize_value(current)
            self._storage.log_configs({path: (type_tag, serialized)})

    def exists(self, path: str) -> bool:
        """Check if a field exists (config, metric, or string series).

        Args:
            path: The field path to check.

        Returns:
            True if the field exists, False otherwise.
        """
        if self._storage is None:
            raise RuntimeError("Run has no local storage")
        return self._storage.field_exists(normalize_path(path))

    def _fetch_field(self, path: str) -> Any:
        """Fetch the current value of a field from local storage.

        For configs, returns the deserialized value.
        For metric series, returns the last logged value (float).
        For string series, returns the last logged value (str).
        Returns None if the field does not exist.
        """
        if self._storage is None:
            raise RuntimeError("Run has no local storage")

        normed = normalize_path(path)

        # Check configs first
        config = self._storage.get_config(normed)
        if config is not None:
            return deserialize_value(config[0], config[1])

        # Check metric series
        metric = self._storage.get_last_metric_value(normed)
        if metric is not None:
            return metric[1]  # return the y value

        # Check string series
        string_val = self._storage.get_last_string_value(normed)
        if string_val is not None:
            return string_val[1]  # return the string value

        return None

    def _resolve_step(self, path: str, step: Number | None) -> Number:
        """Return the step to use for *path*, advancing the auto counter.

        When *step* is ``None``, returns the next auto-increment value for
        *path* (starting at 0).  When *step* is explicit, it is returned
        as-is but the auto counter is advanced past it so that a subsequent
        ``None`` call won't collide.

        Must be called while holding ``self._lock``.
        """
        if step is None:
            s = self._auto_step.get(path, 0)
            self._auto_step[path] = s + 1
            return s
        # Explicit step: advance counter past it to avoid future collisions.
        prev = self._auto_step.get(path, 0)
        if step + 1 > prev:
            self._auto_step[path] = step + 1
        return step

    def log_metrics(
        self,
        data: dict[str, float],
        step: Number,
    ) -> None:
        """Log metric values at a given step.

        Args:
            data: Dictionary of metric_path -> float_value.
            step: The step number. Steps can be in any order.
                Logging the same step + path overwrites the previous value.
        """
        with self._lock:
            if self._read_only:
                raise RuntimeError("Run is read-only")
            if self._storage_mode == Storage.DISABLED:
                return
            if self._closed:
                raise RuntimeError("Run is closed")

            ts = int(datetime.now(timezone.utc).timestamp())

            points = []
            for k, v in data.items():
                path = normalize_path(k)
                s = self._resolve_step(path, step)
                points.append((path, s, float(v), ts))

            self._storage.log_metric_points(points)

    def log_strings(
        self,
        data: dict[str, str],
        step: Number,
    ) -> None:
        """Log string series values at a given step.

        Args:
            data: Dictionary of series_path -> string_value.
            step: The step number. Steps can be in any order.
                Logging the same step + path overwrites the previous value.
        """
        with self._lock:
            if self._read_only:
                raise RuntimeError("Run is read-only")
            if self._storage_mode == Storage.DISABLED:
                return
            if self._closed:
                raise RuntimeError("Run is closed")

            ts = int(datetime.now(timezone.utc).timestamp())

            points = []
            for k, v in data.items():
                path = normalize_path(k)
                s = self._resolve_step(path, step)
                points.append((path, s, str(v), ts))

            self._storage.log_string_points(points)

    def log_string_series(
        self,
        data: dict[str, str],
        step: Number,
    ) -> None:
        """Deprecated alias for ``log_strings``."""
        warnings.warn(
            "'log_string_series' is deprecated and will be removed in a "
            "future release; use 'log_strings' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.log_strings(data, step=step)

    def _require_remote(self) -> None:
        """Raise if no remote connection is available."""
        if self._remote_id is None:
            raise RuntimeError(
                "No remote connection. Use storage='cloud'."
            )

    def get_metric_paths(self) -> list[str]:
        """Fetch metric paths from the remote API."""
        self._require_remote()
        status, resp = _api_get(
            f"{API_BASE}/api/v1/runs/{self._remote_id}/paths",
            api_key=self._api_key,
        )
        if status != 200:
            raise RuntimeError(f"Failed to fetch metric paths (status {status})")
        payload = json.loads(resp)
        if not isinstance(payload, list):
            raise RuntimeError("Invalid response format for metric paths")
        return _filter_paths_by_type(payload, "metric")

    def get_metric_data(
        self,
        path: str,
        *,
        step_min: int | None = None,
        step_max: int | None = None,
        max_points: int | None = None,
    ) -> dict[str, Any]:
        """Fetch metric data for a path from the remote API.

        Returns dict with keys: ``path``, ``downsampled``,
        ``raw_points`` (list of ``{step, y}``) or ``buckets``.
        """
        self._require_remote()
        params: dict[str, Any] = {"path": path}
        if step_min is not None:
            params["step_min"] = step_min
        if step_max is not None:
            params["step_max"] = step_max
        if max_points is not None:
            params["max_points"] = max_points
        status, resp = _api_get(
            f"{API_BASE}/api/v1/runs/{self._remote_id}/data/metrics",
            api_key=self._api_key,
            params=params,
        )
        if status != 200:
            raise RuntimeError(f"Failed to fetch metric data (status {status})")
        return json.loads(resp)

    def get_string_paths(self) -> list[str]:
        """Fetch string series paths from the remote API."""
        self._require_remote()
        status, resp = _api_get(
            f"{API_BASE}/api/v1/runs/{self._remote_id}/paths",
            api_key=self._api_key,
        )
        if status != 200:
            raise RuntimeError(f"Failed to fetch string paths (status {status})")
        payload = json.loads(resp)
        if not isinstance(payload, list):
            raise RuntimeError("Invalid response format for string paths")
        return _filter_paths_by_type(payload, "string")

    def get_string_data(
        self,
        path: str,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch string data for a path from the remote API.

        Returns list of dicts with keys ``step`` and ``value``.
        """
        self._require_remote()
        params: dict[str, Any] = {"path": path}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        status, resp = _api_get(
            f"{API_BASE}/api/v1/runs/{self._remote_id}/data/strings",
            api_key=self._api_key,
            params=params,
        )
        if status != 200:
            raise RuntimeError(f"Failed to fetch string data (status {status})")
        data = json.loads(resp)
        # Backend returns {points, total}; unwrap for backwards compatibility
        if isinstance(data, dict) and "points" in data:
            return data["points"]
        return data

    def get_configs(self) -> list[dict[str, Any]]:
        """Fetch configs from the remote API.

        Returns list of dicts with keys ``path``, ``type_tag``,
        ``value``, ``updated_at``.
        """
        self._require_remote()
        status, resp = _api_get(
            f"{API_BASE}/api/v1/runs/{self._remote_id}",
            api_key=self._api_key,
        )
        if status != 200:
            raise RuntimeError(f"Failed to fetch run details (status {status})")
        return json.loads(resp).get("configs", [])

    def close(self, status: str = "finished") -> None:
        """Close the run.

        Args:
            status: Run status to set ('finished' or 'failed').

        The WAL is checkpointed so the run is a single .sqlite file.
        No-op for read-only runs.
        """
        if self._read_only or self._storage_mode == Storage.DISABLED:
            return
        with self._lock:
            if self._closed:
                return
            self._closed = True

        if self._monitoring is not None:
            self._monitoring.close()

        self._storage.set_meta("status", status)
        self._storage.set_meta("closed_at", datetime.now(timezone.utc).isoformat())
        now = datetime.now(timezone.utc).isoformat()
        serialized = {}
        for k, v in {"sys/state": status, "sys/closed_time": now}.items():
            type_tag, val = serialize_value(v)
            serialized[k] = (type_tag, val)
        self._storage.log_configs(serialized)

        remaining_unuploaded: int | None = None
        if self._sync_process is not None:
            self._sync_process.close()
            try:
                remaining_unuploaded = self._storage.count_unuploaded()
            except Exception:
                remaining_unuploaded = None
            self._sync_process = None

        if (
            self._storage_mode == Storage.CLOUD
            and remaining_unuploaded is not None
            and remaining_unuploaded > 0
        ):
            print(
                "[Goodseed] Warning: "
                f"{remaining_unuploaded} item(s) are still pending upload. "
                "You can upload remaining data with:\n"
                f"  goodseed upload -p {self.project} -r {self.run_id}",
                file=sys.stderr,
            )

        self._storage.checkpoint_wal()
        self._storage.close()
        print(f"Goodseed run closed: {self.run_id}")

    def sync(self) -> None:
        """Request an immediate sync cycle (non-blocking).

        Data is always persisted to local SQLite immediately. This method
        triggers the background sync process to upload to the remote API
        without waiting for the next scheduled interval.
        """
        if self._sync_process is not None:
            self._sync_process.sync()

    def wait(self, timeout: float | None = None) -> None:
        """Block until all queued data has been uploaded to the remote API.

        Args:
            timeout: Maximum seconds to wait. None means wait indefinitely.
        """
        if self._sync_process is not None:
            self._sync_process.wait(timeout)

    def _cleanup(self) -> None:
        if not self._closed:
            self.close()

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None and self._monitoring and self._monitoring.capture_traceback:
            self._monitoring.log_traceback(exc_type, exc_val, exc_tb)
        status = "failed" if exc_type is not None else "finished"
        self.close(status=status)
