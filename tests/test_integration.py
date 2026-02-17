"""Integration tests for the local-only workflow.

Tests the full flow: Run() -> log data -> close -> server reads it.
No external dependencies required.
"""

import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from goodseed.run import Run
from goodseed.server import _RequestHandler, _scan_runs, _get_configs, _get_metrics, _get_metric_paths
from goodseed.storage import LocalStorage


@pytest.fixture
def projects_dir(tmp_path):
    """Create a projects directory with test runs."""
    pdir = tmp_path / "projects"
    pdir.mkdir()
    return pdir


def _create_run(goodseed_home, project="default", run_name=None,
                experiment_name="test-exp", configs=None, metrics=None):
    """Helper to create and close a run with data."""
    r = Run(
        experiment_name=experiment_name,
        project=project,
        run_name=run_name,
        goodseed_home=goodseed_home,
    )
    if configs:
        r.log_configs(configs)
    if metrics:
        for step, data in enumerate(metrics):
            r.log_metrics(data, step=step)
    r.close()
    return r


class TestRunToStorage:
    """Test that Run correctly writes data readable by storage."""

    def test_full_run_lifecycle(self, tmp_path):
        r = _create_run(
            tmp_path,
            run_name="lifecycle-test",
            configs={"lr": 0.001, "batch_size": 32},
            metrics=[
                {"loss": 1.0, "acc": 0.1},
                {"loss": 0.5, "acc": 0.5},
                {"loss": 0.1, "acc": 0.9},
            ],
        )

        db_path = tmp_path / "projects" / "default" / "runs" / "lifecycle-test.sqlite"
        assert db_path.exists()

        # No WAL/SHM files after close
        assert not Path(str(db_path) + "-wal").exists()
        assert not Path(str(db_path) + "-shm").exists()

        # Verify data via storage
        s = LocalStorage(db_path)
        assert s.get_meta("status") == "finished"
        assert s.get_meta("closed_at") is not None

        configs = s.get_configs()
        assert configs["lr"] == ("float", "0.001")
        assert configs["batch_size"] == ("int", "32")

        points = s.get_metric_points("loss")
        assert len(points) == 3
        assert points[0]["y"] == 1.0
        assert points[2]["y"] == 0.1

        paths = s.get_metric_paths()
        assert set(paths) == {"acc", "loss"}
        s.close()


class TestServerHelpers:
    """Test server helper functions directly (no HTTP)."""

    def test_scan_runs_empty(self, projects_dir):
        runs = _scan_runs(projects_dir)
        assert runs == []

    def test_scan_runs_finds_runs(self, tmp_path):
        _create_run(tmp_path, run_name="run-a", experiment_name="exp-a")
        _create_run(tmp_path, run_name="run-b", experiment_name="exp-b")

        projects_dir = tmp_path / "projects"
        runs = _scan_runs(projects_dir)
        assert len(runs) == 2
        run_ids = {r["run_id"] for r in runs}
        assert run_ids == {"run-a", "run-b"}

    def test_scan_runs_multiple_projects(self, tmp_path):
        _create_run(tmp_path, project="proj1", run_name="r1")
        _create_run(tmp_path, project="proj2", run_name="r2")

        projects_dir = tmp_path / "projects"
        runs = _scan_runs(projects_dir)
        assert len(runs) == 2
        projects = {r["project"] for r in runs}
        assert projects == {"proj1", "proj2"}

    def test_scan_runs_sorted_newest_first(self, tmp_path):
        _create_run(tmp_path, run_name="old-run", experiment_name="old")
        time.sleep(0.05)  # Ensure different timestamps
        _create_run(tmp_path, run_name="new-run", experiment_name="new")

        runs = _scan_runs(tmp_path / "projects")
        assert runs[0]["run_id"] == "new-run"
        assert runs[1]["run_id"] == "old-run"

    def test_get_configs(self, tmp_path):
        r = _create_run(tmp_path, run_name="cfg-run", configs={"lr": 0.01, "model": "cnn"})
        db_path = tmp_path / "projects" / "default" / "runs" / "cfg-run.sqlite"
        configs = _get_configs(db_path)
        assert configs["lr"] == 0.01
        assert configs["model"] == "cnn"

    def test_get_metrics(self, tmp_path):
        _create_run(
            tmp_path, run_name="met-run",
            metrics=[{"loss": 1.0}, {"loss": 0.5}],
        )
        db_path = tmp_path / "projects" / "default" / "runs" / "met-run.sqlite"

        metrics = _get_metrics(db_path, "loss")
        assert len(metrics) == 2
        assert metrics[0]["value"] == 1.0
        assert metrics[1]["value"] == 0.5
        assert metrics[0]["step"] == 0
        assert metrics[1]["step"] == 1
        assert metrics[0]["path"] == "loss"

    def test_get_metrics_all(self, tmp_path):
        _create_run(
            tmp_path, run_name="all-met",
            metrics=[{"loss": 1.0, "acc": 0.1}],
        )
        db_path = tmp_path / "projects" / "default" / "runs" / "all-met.sqlite"
        metrics = _get_metrics(db_path)
        assert len(metrics) == 2

    def test_get_metric_paths(self, tmp_path):
        _create_run(
            tmp_path, run_name="paths-run",
            metrics=[{"train/loss": 1.0, "train/acc": 0.1, "val/loss": 0.9}],
        )
        db_path = tmp_path / "projects" / "default" / "runs" / "paths-run.sqlite"
        paths = _get_metric_paths(db_path)
        assert set(paths) == {"train/acc", "train/loss", "val/loss"}


class TestServerHTTP:
    """Test the HTTP server with actual HTTP requests."""

    @pytest.fixture
    def server(self, tmp_path):
        """Start a test server on a random port."""
        # Create test data
        _create_run(
            tmp_path, run_name="http-run", experiment_name="http-exp",
            configs={"lr": 0.001},
            metrics=[{"loss": 1.0}, {"loss": 0.5}],
        )

        projects_dir = tmp_path / "projects"
        _RequestHandler.projects_dir = projects_dir

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RequestHandler)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        yield f"http://127.0.0.1:{port}"

        httpd.shutdown()

    def _get(self, url):
        req = Request(url)
        with urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())

    def test_get_runs(self, server):
        data = self._get(f"{server}/api/runs")
        assert "runs" in data
        assert len(data["runs"]) == 1
        assert data["runs"][0]["run_id"] == "http-run"
        assert data["runs"][0]["experiment_name"] == "http-exp"

    def test_get_configs(self, server):
        data = self._get(f"{server}/api/runs/default/http-run/configs")
        assert "configs" in data
        assert data["configs"]["lr"] == 0.001

    def test_get_metrics(self, server):
        data = self._get(f"{server}/api/runs/default/http-run/metrics?path=loss")
        assert "metrics" in data
        assert len(data["metrics"]) == 2
        assert data["metrics"][0]["value"] == 1.0

    def test_get_metrics_all(self, server):
        data = self._get(f"{server}/api/runs/default/http-run/metrics")
        assert len(data["metrics"]) == 2

    def test_get_metric_paths(self, server):
        data = self._get(f"{server}/api/runs/default/http-run/metric-paths")
        assert "paths" in data
        assert data["paths"] == ["loss"]

    def test_run_not_found(self, server):
        from urllib.error import HTTPError
        with pytest.raises(HTTPError) as exc_info:
            self._get(f"{server}/api/runs/default/nonexistent/configs")
        assert exc_info.value.code == 404

    def test_cors_headers(self, server):
        req = Request(f"{server}/api/runs")
        with urlopen(req, timeout=5) as resp:
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_options_preflight(self, server):
        req = Request(f"{server}/api/runs", method="OPTIONS")
        with urlopen(req, timeout=5) as resp:
            assert resp.status == 204
