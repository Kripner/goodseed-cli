"""Neptune v2 Run adapter.

Wraps ``goodseed.Run`` to expose the Neptune v2 API surface, including
the dict-like ``run["path"].log()`` interface via ``NeptuneHandler``.
"""

from __future__ import annotations

from typing import Any

from goodseed.neptune._handler import NeptuneHandler
from goodseed.run import Run


class NeptuneRun(Run):
    """Neptune v2-compatible Run backed by goodseed storage.

    Adds Neptune v2-specific methods (``sync``, ``wait``, ``exists``,
    ``get_structure``, ``print_structure``) and returns ``NeptuneHandler``
    from ``__getitem__`` instead of goodseed's ``_FieldHandler``.
    """

    def __getitem__(self, key: str) -> NeptuneHandler:
        """Return a ``NeptuneHandler`` for Neptune v2's dict-like access."""
        return NeptuneHandler(self, key)

    def stop(self) -> None:
        """Stop the run (Neptune v2 alias for close)."""
        self.close()

    def get_structure(self) -> dict[str, Any]:
        """Return the full metadata tree. Not yet supported."""
        raise NotImplementedError(
            "goodseed does not support get_structure() yet"
        )

    def print_structure(self) -> None:
        """Print the metadata tree. Not yet supported."""
        raise NotImplementedError(
            "goodseed does not support print_structure() yet"
        )
