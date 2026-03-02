"""Workspace/project/run helpers for local and remote GoodSeed storage."""

from __future__ import annotations

from typing import Any, Literal
from urllib import parse

from goodseed.config import get_projects_dir
from goodseed.server import _scan_projects, _scan_runs
from goodseed.sync import api_get_json, api_post_json

StorageMode = Literal["auto", "local", "remote"]


def _resolve_storage(storage: StorageMode, api_key: str | None) -> Literal["local", "remote"]:
    if storage == "auto":
        return "remote" if api_key else "local"
    if storage == "remote" and not api_key:
        raise RuntimeError("api_key is required for remote storage mode.")
    return storage


def list_workspaces(
    *,
    storage: StorageMode = "auto",
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """List workspaces for local or remote storage."""
    mode = _resolve_storage(storage, api_key)
    if mode == "remote":
        status, payload = api_get_json("/api/v1/workspaces", api_key=api_key or "")
        if status != 200 or not isinstance(payload, list):
            raise RuntimeError(f"Failed to list workspaces ({status}): {payload}")
        return payload

    projects = _scan_projects(get_projects_dir())
    workspaces = sorted({name["name"].split("/", 1)[0] for name in projects if "/" in name["name"]})
    if not workspaces:
        workspaces = ["default"]
    return [{"id": wid, "role": "owner"} for wid in workspaces]


def me(*, api_key: str) -> dict[str, Any]:
    """Return authenticated user profile from remote server."""
    status, payload = api_get_json("/api/v1/auth/me", api_key=api_key)
    if status != 200 or not isinstance(payload, dict):
        raise RuntimeError(
            f"Failed to fetch profile from /api/v1/auth/me ({status}): {payload}"
        )
    return payload


def list_projects(
    workspace: str,
    *,
    storage: StorageMode = "auto",
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """List projects in a workspace for local or remote storage."""
    mode = _resolve_storage(storage, api_key)
    if mode == "remote":
        workspace_q = parse.quote(workspace, safe="")
        status, payload = api_get_json(
            f"/api/v1/workspaces/{workspace_q}/projects",
            api_key=api_key or "",
        )
        if status != 200 or not isinstance(payload, list):
            raise RuntimeError(f"Failed to list projects ({status}): {payload}")
        return payload

    projects = _scan_projects(get_projects_dir())
    result = []
    for item in projects:
        name = item["name"]
        if "/" not in name:
            if workspace == "default":
                result.append({"name": name})
            continue
        ws, proj = name.split("/", 1)
        if ws == workspace:
            result.append({"name": proj})
    return result


def ensure_project(
    workspace: str,
    project_name: str,
    *,
    storage: StorageMode = "auto",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Ensure a project exists. Returns {'name', 'created'}."""
    mode = _resolve_storage(storage, api_key)
    if mode == "remote":
        existing = list_projects(
            workspace,
            storage="remote",
            api_key=api_key,
        )
        for item in existing:
            name = item.get("name") or item.get("Name")
            if name == project_name:
                return {"name": project_name, "created": False}

        workspace_q = parse.quote(workspace, safe="")
        status, payload = api_post_json(
            f"/api/v1/workspaces/{workspace_q}/projects",
            api_key=api_key or "",
            body={"name": project_name},
        )
        if status != 200:
            raise RuntimeError(f"Failed to create project ({status}): {payload}")
        resolved_name = project_name
        if isinstance(payload, dict):
            resolved_name = payload.get("name") or payload.get("Name") or project_name
        return {"name": str(resolved_name), "created": True}

    projects_dir = get_projects_dir()
    project_dir = projects_dir / workspace / project_name / "runs"
    created = not project_dir.exists()
    project_dir.mkdir(parents=True, exist_ok=True)
    return {"name": project_name, "created": created}


def list_runs(
    workspace: str,
    project_name: str,
    *,
    storage: StorageMode = "auto",
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """List runs in a workspace/project for local or remote storage."""
    mode = _resolve_storage(storage, api_key)
    if mode == "remote":
        workspace_q = parse.quote(workspace, safe="")
        project_q = parse.quote(project_name, safe="")
        status, payload = api_get_json(
            f"/api/v1/workspaces/{workspace_q}/projects/{project_q}/runs",
            api_key=api_key or "",
        )
        if status != 200 or not isinstance(payload, list):
            raise RuntimeError(f"Failed to list runs ({status}): {payload}")
        return payload

    full_project_name = f"{workspace}/{project_name}" if workspace != "default" else project_name
    runs = _scan_runs(get_projects_dir())
    return [{"run_id": run.get("run_id")} for run in runs if run.get("project") == full_project_name]
