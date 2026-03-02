"""Pytest configuration and shared fixtures for Goodseed tests."""

import functools

import pytest

from goodseed.run import Run

_original_init = Run.__init__


@pytest.fixture(autouse=True)
def _disable_monitoring_by_default(monkeypatch):
    """Disable monitoring daemons in tests unless explicitly enabled.

    This prevents background threads from interfering with test assertions
    and avoids monkey-patching sys.stdout/stderr during pytest output capture.
    Tests that specifically test monitoring should pass the desired
    ``capture_*`` kwargs explicitly.
    """

    @functools.wraps(_original_init)
    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("capture_hardware_metrics", False)
        kwargs.setdefault("capture_stdout", False)
        kwargs.setdefault("capture_stderr", False)
        kwargs.setdefault("capture_traceback", False)
        kwargs.setdefault("storage", "local")
        return _original_init(self, *args, **kwargs)

    monkeypatch.setattr(Run, "__init__", patched_init)
