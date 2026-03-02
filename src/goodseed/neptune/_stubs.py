"""Stub classes for Neptune v2 features not supported by goodseed.

Model, ModelVersion, Project, and Table all raise ``NotImplementedError``
on every method. They exist so that ``import goodseed.neptune`` exposes
the full Neptune v2 API surface.
"""

from __future__ import annotations

from typing import Any


def _not_implemented(cls_name: str, method: str) -> None:
    raise NotImplementedError(
        f"goodseed does not support {cls_name}.{method}() yet"
    )


class _NeptuneObjectStub:
    """Base stub with methods common to Model, ModelVersion, and Project."""

    _TYPE_NAME: str = "NeptuneObject"

    def stop(self) -> None:
        _not_implemented(self._TYPE_NAME, "stop")

    def sync(self) -> None:
        _not_implemented(self._TYPE_NAME, "sync")

    def wait(self) -> None:
        _not_implemented(self._TYPE_NAME, "wait")

    def exists(self, path: str) -> bool:
        _not_implemented(self._TYPE_NAME, "exists")
        return False  # unreachable

    def get_structure(self) -> dict[str, Any]:
        _not_implemented(self._TYPE_NAME, "get_structure")
        return {}  # unreachable

    def print_structure(self) -> None:
        _not_implemented(self._TYPE_NAME, "print_structure")

    def __getitem__(self, key: str) -> Any:
        _not_implemented(self._TYPE_NAME, "__getitem__")

    def __setitem__(self, key: str, value: Any) -> None:
        _not_implemented(self._TYPE_NAME, "__setitem__")

    def __enter__(self) -> _NeptuneObjectStub:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class Model(_NeptuneObjectStub):
    """Stub for Neptune v2 Model. All methods raise ``NotImplementedError``."""

    _TYPE_NAME = "Model"

    def __init__(self, **kwargs: Any) -> None:
        pass


class ModelVersion(_NeptuneObjectStub):
    """Stub for Neptune v2 ModelVersion."""

    _TYPE_NAME = "ModelVersion"

    def __init__(self, **kwargs: Any) -> None:
        pass

    def change_stage(self, stage: str) -> None:
        """Transition lifecycle stage. Not yet supported."""
        _not_implemented("ModelVersion", "change_stage")


class Project(_NeptuneObjectStub):
    """Stub for Neptune v2 Project."""

    _TYPE_NAME = "Project"

    def __init__(self, **kwargs: Any) -> None:
        pass

    def fetch_runs_table(self, **kwargs: Any) -> Table:
        """Query runs. Not yet supported."""
        _not_implemented("Project", "fetch_runs_table")
        return Table()  # unreachable

    def fetch_models_table(self, **kwargs: Any) -> Table:
        """Query models. Not yet supported."""
        _not_implemented("Project", "fetch_models_table")
        return Table()  # unreachable

    def fetch_model_versions_table(self, **kwargs: Any) -> Table:
        """Query model versions. Not yet supported."""
        _not_implemented("Project", "fetch_model_versions_table")
        return Table()  # unreachable


class Table:
    """Stub for Neptune v2 Table."""

    def __iter__(self) -> Any:
        _not_implemented("Table", "__iter__")

    def to_pandas(self) -> Any:
        """Convert to pandas DataFrame. Not yet supported."""
        _not_implemented("Table", "to_pandas")
