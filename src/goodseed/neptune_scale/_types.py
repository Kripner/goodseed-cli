"""Neptune Scale type stubs.

These match Neptune Scale's ``File`` and ``Histogram`` signatures so that
user code referencing them compiles, even though goodseed doesn't yet
support file uploads or histogram logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Union


@dataclass
class File:
    """A file to upload to Neptune.

    Matches the ``neptune_scale.File`` signature. Used with
    ``run.log_files()`` and ``run.assign_files()`` which currently raise
    ``NotImplementedError``.
    """

    source: Union[str, Path, bytes, BinaryIO]
    mime_type: str | None = None
    size: int | None = None
    destination: str | None = None


@dataclass
class Histogram:
    """A histogram to log to Neptune.

    Matches the ``neptune_scale.Histogram`` signature. Used with
    ``run.log_histograms()`` which currently raises ``NotImplementedError``.

    Exactly one of *counts* or *densities* must be provided.
    """

    bin_edges: Any  # ArrayLike
    counts: Any = None  # ArrayLike | None
    densities: Any = None  # ArrayLike | None
