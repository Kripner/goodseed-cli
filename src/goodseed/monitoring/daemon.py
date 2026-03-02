"""Base daemon thread for periodic monitoring tasks."""

import abc
import threading


class MonitoringDaemon(threading.Thread, abc.ABC):
    """Background thread that calls ``work()`` at a fixed interval.

    The thread is a daemon so it won't prevent the process from exiting.
    Call ``stop()`` for a graceful shutdown (waits up to one interval).
    """

    def __init__(self, interval: float, name: str) -> None:
        super().__init__(daemon=True, name=name)
        self._interval = interval
        self._stop_event = threading.Event()

    def run(self) -> None:
        # Do first work immediately, then wait between iterations
        while True:
            try:
                self.work()
            except Exception:
                pass  # Never crash the daemon
            if self._stop_event.wait(self._interval):
                break

    def stop(self) -> None:
        """Signal the thread to stop after the current iteration."""
        self._stop_event.set()

    @abc.abstractmethod
    def work(self) -> None:
        """Override to perform periodic work."""
        ...
