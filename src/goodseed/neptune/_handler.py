"""Neptune v2 dict-like field handler.

Provides the ``run["path"].log()``, ``run["path"].assign()`` etc. interface
that Neptune v2 users expect, delegating to goodseed's Run internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from goodseed.neptune._run import NeptuneRun


def _not_implemented(name: str) -> None:
    raise NotImplementedError(
        f"goodseed does not support Handler.{name}() yet"
    )


class NeptuneHandler:
    """Handler returned by ``NeptuneRun[path]``.

    Supports Neptune v2's method names: ``log``, ``assign``, ``extend``,
    ``fetch``, ``add``, ``add_all``, and chained bracket access
    (``run["a"]["b"]`` builds path ``"a/b"``).
    """

    __slots__ = ("_run", "_path")

    def __init__(self, run: NeptuneRun, path: str) -> None:
        self._run = run
        self._path = path

    def __getitem__(self, key: str) -> NeptuneHandler:
        """Chained bracket access: ``run["a"]["b"]["c"]`` → path ``"a/b/c"``."""
        return NeptuneHandler(self._run, f"{self._path}/{key}")

    # --- Implemented methods ---

    def log(
        self,
        value: Any,
        *,
        step: int | float,
        timestamp: Any = None,
        wait: bool = False,
    ) -> None:
        """Append a value to a series (Neptune v2 ``log``).

        Maps to ``goodseed.Run._append_to_field``. The *timestamp* and
        *wait* parameters are accepted for API compatibility but ignored.
        """
        self._run._append_to_field(self._path, value, step)

    def assign(self, value: Any, *, wait: bool = False) -> None:
        """Assign an atomic value (Neptune v2 ``assign``).

        Maps to ``goodseed.Run.__setitem__``.
        """
        self._run[self._path] = value

    def append(
        self,
        value: Any,
        *,
        step: int | float,
        timestamp: Any = None,
        wait: bool = False,
    ) -> None:
        """Append a single value to a series (Neptune v2 ``append``).

        Maps to ``goodseed.Run._append_to_field``. The *timestamp* and
        *wait* parameters are accepted for API compatibility but ignored.
        """
        self._run._append_to_field(self._path, value, step)

    def extend(
        self,
        values: list[Any],
        *,
        steps: list[int | float],
        timestamps: Any = None,
        wait: bool = False,
    ) -> None:
        """Batch-append values to a series.

        Args:
            values: List of values to append.
            steps: List of step values (same length as values).
            timestamps: Ignored (API compatibility).
            wait: Ignored (API compatibility).
        """
        if len(steps) != len(values):
            raise ValueError(
                f"steps length ({len(steps)}) must match "
                f"values length ({len(values)})"
            )
        for i, v in enumerate(values):
            self._run._append_to_field(self._path, v, steps[i])

    def fetch(self) -> Any:
        """Fetch the current value of this field from local storage."""
        return self._run._fetch_field(self._path)

    def add(self, value: str | list[str]) -> None:
        """Add value(s) to a string set (e.g. tags)."""
        self._run._add_to_string_set(self._path, value)

    def add_all(self, values: list[str]) -> None:
        """Add multiple values to a string set."""
        self._run._add_to_string_set(self._path, values)

    # --- Not implemented ---

    def upload(self, value: Any = None, *, wait: bool = False) -> None:
        """Upload a file. Not yet supported."""
        _not_implemented("upload")

    def upload_files(self, glob_pattern: str = "", *, wait: bool = False) -> None:
        """Upload files by glob pattern. Not yet supported."""
        _not_implemented("upload_files")

    def download(self, destination: str | None = None) -> None:
        """Download a file. Not yet supported."""
        _not_implemented("download")

    def clear(self, *, wait: bool = False) -> None:
        """Clear a field. Not yet supported."""
        _not_implemented("clear")

    def extend_dict(self, value: dict[str, Any]) -> None:
        """Extend a namespace with a dictionary. Not yet supported."""
        _not_implemented("extend_dict")
