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
from goodseed.server import _RequestHandler, _scan_runs
from goodseed.storage import (
    LocalStorage,
    downsample_metrics,
    read_configs,
    read_metric_paths,
    read_metrics,
    read_string_series,
)


@pytest.fixture
def projects_dir(tmp_path):
    """Create a projects directory with test runs."""
    pdir = tmp_path / "projects"
    pdir.mkdir()
    return pdir


def _create_run(goodseed_home, project="default", run_id=None,
                name="test-exp", configs=None, metrics=None):
    """Helper to create and close a run with data."""
    r = Run(
        name=name,
        project=project,
        run_id=run_id,
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
            run_id="lifecycle-test",
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
        _create_run(tmp_path, run_id="run-a", name="exp-a")
        _create_run(tmp_path, run_id="run-b", name="exp-b")

        projects_dir = tmp_path / "projects"
        runs = _scan_runs(projects_dir)
        assert len(runs) == 2
        run_ids = {r["run_id"] for r in runs}
        assert run_ids == {"run-a", "run-b"}

    def test_scan_runs_multiple_projects(self, tmp_path):
        _create_run(tmp_path, project="proj1", run_id="r1")
        _create_run(tmp_path, project="proj2", run_id="r2")

        projects_dir = tmp_path / "projects"
        runs = _scan_runs(projects_dir)
        assert len(runs) == 2
        projects = {r["project"] for r in runs}
        assert projects == {"proj1", "proj2"}

    def test_scan_runs_sorted_newest_first(self, tmp_path):
        _create_run(tmp_path, run_id="old-run", name="old")
        time.sleep(0.05)  # Ensure different timestamps
        _create_run(tmp_path, run_id="new-run", name="new")

        runs = _scan_runs(tmp_path / "projects")
        assert runs[0]["run_id"] == "new-run"
        assert runs[1]["run_id"] == "old-run"

    def test_get_configs(self, tmp_path):
        r = _create_run(tmp_path, run_id="cfg-run", configs={"lr": 0.01, "model": "cnn"})
        db_path = tmp_path / "projects" / "default" / "runs" / "cfg-run.sqlite"
        configs = read_configs(db_path)
        assert configs["lr"] == 0.01
        assert configs["model"] == "cnn"

    def test_get_metrics(self, tmp_path):
        _create_run(
            tmp_path, run_id="met-run",
            metrics=[{"loss": 1.0}, {"loss": 0.5}],
        )
        db_path = tmp_path / "projects" / "default" / "runs" / "met-run.sqlite"

        metrics = read_metrics(db_path, "loss")
        assert len(metrics) == 2
        assert metrics[0]["value"] == 1.0
        assert metrics[1]["value"] == 0.5
        assert metrics[0]["step"] == 0
        assert metrics[1]["step"] == 1
        assert metrics[0]["path"] == "loss"

    def test_get_metrics_all(self, tmp_path):
        _create_run(
            tmp_path, run_id="all-met",
            metrics=[{"loss": 1.0, "acc": 0.1}],
        )
        db_path = tmp_path / "projects" / "default" / "runs" / "all-met.sqlite"
        metrics = read_metrics(db_path)
        assert len(metrics) == 2

    def test_get_metric_paths(self, tmp_path):
        _create_run(
            tmp_path, run_id="paths-run",
            metrics=[{"train/loss": 1.0, "train/acc": 0.1, "val/loss": 0.9}],
        )
        db_path = tmp_path / "projects" / "default" / "runs" / "paths-run.sqlite"
        paths = read_metric_paths(db_path)
        assert set(paths) == {"train/acc", "train/loss", "val/loss"}


class TestServerHTTP:
    """Test the HTTP server with actual HTTP requests."""

    @pytest.fixture
    def server(self, tmp_path):
        """Start a test server on a random port."""
        # Create test data
        _create_run(
            tmp_path, run_id="http-run", name="http-exp",
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

    def test_run_list_includes_metric_paths(self, server):
        data = self._get(f"{server}/api/runs")
        run = data["runs"][0]
        assert "metric_paths" in run
        assert "loss" in run["metric_paths"]


class TestServerHTTPExtended:
    """Additional HTTP tests for endpoints not covered by TestServerHTTP."""

    @pytest.fixture
    def server(self, tmp_path):
        """Start a test server with richer data for extended tests."""
        # Run in default project with many metric points + string series
        r = Run(run_id="ext-run", name="ext-exp", project="default", goodseed_home=tmp_path)
        for i in range(50):
            r.log_metrics({"train/loss": 1.0 / (i + 1)}, step=i)
        r["log"].log("epoch 1 started", step=0)
        r["log"].log("epoch 1 finished", step=1)
        r["log"].log("epoch 2 started", step=2)
        r.close()

        # Run in a second project
        _create_run(tmp_path, project="other-proj", run_id="other-run", name="other-exp")

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

    def test_get_projects(self, server):
        data = self._get(f"{server}/api/projects")
        assert "projects" in data
        names = {p["name"] for p in data["projects"]}
        assert "default" in names
        assert "other-proj" in names
        for p in data["projects"]:
            assert "run_count" in p
            assert p["run_count"] >= 1

    def test_runs_project_filter(self, server):
        data = self._get(f"{server}/api/runs?project=default")
        assert len(data["runs"]) == 1
        assert data["runs"][0]["run_id"] == "ext-run"

        data = self._get(f"{server}/api/runs?project=other-proj")
        assert len(data["runs"]) == 1
        assert data["runs"][0]["run_id"] == "other-run"

    def test_downsampled_metrics_via_http(self, server):
        url = f"{server}/api/runs/default/ext-run/metrics?path=train/loss&pointCount=5"
        data = self._get(url)
        assert "points" in data
        assert data["totalCount"] == 50
        assert data["downsampled"] is True
        assert len(data["points"]) <= 10

    def test_downsampled_metrics_with_range(self, server):
        url = (
            f"{server}/api/runs/default/ext-run/metrics"
            f"?path=train/loss&pointCount=100&firstPointIndex=10&lastPointIndex=19"
        )
        data = self._get(url)
        assert data["firstPointIndex"] == 10
        assert data["lastPointIndex"] == 19
        assert len(data["points"]) == 10

    def test_string_series_via_http(self, server):
        data = self._get(f"{server}/api/runs/default/ext-run/string_series?path=log")
        assert "string_series" in data
        assert data["total"] == 3
        assert len(data["string_series"]) == 3
        assert data["string_series"][0]["value"] == "epoch 1 started"

    def test_string_series_with_tail(self, server):
        data = self._get(f"{server}/api/runs/default/ext-run/string_series?path=log&tail=1")
        assert data["total"] == 3
        assert len(data["string_series"]) == 1
        assert data["string_series"][0]["value"] == "epoch 2 started"

    def test_string_series_with_limit_offset(self, server):
        data = self._get(f"{server}/api/runs/default/ext-run/string_series?path=log&limit=1&offset=1")
        assert data["total"] == 3
        assert len(data["string_series"]) == 1
        assert data["string_series"][0]["value"] == "epoch 1 finished"

    def test_pointcount_without_path_returns_400(self, server):
        from urllib.error import HTTPError
        with pytest.raises(HTTPError) as exc_info:
            self._get(f"{server}/api/runs/default/ext-run/metrics?pointCount=10")
        assert exc_info.value.code == 400


class TestMonitoringIntegration:
    """Test monitoring features: stdout/stderr capture, hardware, traceback."""

    def test_stdout_capture(self, tmp_path):
        """Captured stdout lines appear as string series entries."""
        r = Run(
            name="stdout-test",
            goodseed_home=tmp_path,
            run_id="stdout-run",
            capture_stdout=True,
            capture_stderr=False,
            capture_hardware_metrics=False,
            capture_traceback=False,
        )
        print("hello from stdout")
        # Give the daemon time to flush (interval=1s, plus some margin)
        time.sleep(1.5)
        db_path = r._db_path
        r.close()

        s = LocalStorage(db_path)
        paths = s.get_string_series_paths()
        stdout_paths = [p for p in paths if p.endswith("/stdout")]
        assert len(stdout_paths) >= 1, f"Expected stdout path, got: {paths}"

        points = s.get_string_points(stdout_paths[0])
        values = [p["value"] for p in points]
        assert any("hello from stdout" in v for v in values), f"stdout not found in: {values}"
        s.close()

    def test_stderr_capture(self, tmp_path):
        """Captured stderr lines appear as string series entries."""
        import sys as _sys

        r = Run(
            name="stderr-test",
            goodseed_home=tmp_path,
            run_id="stderr-run",
            capture_stdout=False,
            capture_stderr=True,
            capture_hardware_metrics=False,
            capture_traceback=False,
        )
        _sys.stderr.write("error line\n")
        time.sleep(1.5)
        db_path = r._db_path
        r.close()

        s = LocalStorage(db_path)
        paths = s.get_string_series_paths()
        stderr_paths = [p for p in paths if p.endswith("/stderr")]
        assert len(stderr_paths) >= 1, f"Expected stderr path, got: {paths}"

        points = s.get_string_points(stderr_paths[0])
        values = [p["value"] for p in points]
        assert any("error line" in v for v in values), f"stderr not found in: {values}"
        s.close()

    def test_capture_disabled(self, tmp_path):
        """With all monitoring disabled, no monitoring data is created."""
        r = Run(
            name="no-monitor",
            goodseed_home=tmp_path,
            run_id="no-monitor-run",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
        )
        print("this should not be captured")
        db_path = r._db_path
        r.close()

        s = LocalStorage(db_path)
        ss_paths = s.get_string_series_paths()
        monitoring_ss = [p for p in ss_paths if p.startswith("monitoring/")]
        assert monitoring_ss == [], f"Unexpected monitoring string series: {monitoring_ss}"

        m_paths = s.get_metric_paths()
        monitoring_m = [p for p in m_paths if p.startswith("monitoring/")]
        assert monitoring_m == [], f"Unexpected monitoring metrics: {monitoring_m}"

        configs = s.get_configs()
        monitoring_configs = [k for k in configs if k.startswith("monitoring/")]
        assert monitoring_configs == [], f"Unexpected monitoring configs: {monitoring_configs}"
        s.close()

    def test_custom_monitoring_namespace(self, tmp_path):
        """Custom monitoring_namespace replaces the default pattern."""
        r = Run(
            name="custom-ns",
            goodseed_home=tmp_path,
            run_id="custom-ns-run",
            capture_stdout=True,
            capture_stderr=False,
            capture_hardware_metrics=False,
            capture_traceback=False,
            monitoring_namespace="monitoring/custom",
        )
        print("custom namespace test")
        time.sleep(1.5)
        db_path = r._db_path
        r.close()

        s = LocalStorage(db_path)
        paths = s.get_string_series_paths()
        assert "monitoring/custom/stdout" in paths, f"Expected custom path, got: {paths}"
        s.close()

    def test_monitoring_metadata_logged(self, tmp_path):
        """Static metadata (hostname, pid, tid) is logged as configs."""
        r = Run(
            name="meta-test",
            goodseed_home=tmp_path,
            run_id="meta-run",
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=False,
            capture_traceback=True,  # just to enable monitoring namespace
        )
        db_path = r._db_path
        r.close()

        s = LocalStorage(db_path)
        configs = s.get_configs()
        config_keys = list(configs.keys())
        hostname_keys = [k for k in config_keys if k.endswith("/hostname")]
        pid_keys = [k for k in config_keys if k.endswith("/pid")]
        tid_keys = [k for k in config_keys if k.endswith("/tid")]
        assert len(hostname_keys) >= 1, f"No hostname config found in: {config_keys}"
        assert len(pid_keys) >= 1, f"No pid config found in: {config_keys}"
        assert len(tid_keys) >= 1, f"No tid config found in: {config_keys}"
        s.close()

    def test_traceback_captured(self, tmp_path):
        """Traceback is logged when an exception occurs within context manager."""
        db_path = None
        with pytest.raises(ValueError):
            with Run(
                name="traceback-test",
                goodseed_home=tmp_path,
                run_id="tb-run",
                capture_stdout=False,
                capture_stderr=False,
                capture_hardware_metrics=False,
                capture_traceback=True,
            ) as r:
                db_path = r._db_path
                raise ValueError("intentional error")

        assert db_path is not None
        s = LocalStorage(db_path)

        # Check run status
        assert s.get_meta("status") == "failed"

        # Check traceback was logged
        paths = s.get_string_series_paths()
        tb_paths = [p for p in paths if p.endswith("/traceback")]
        assert len(tb_paths) >= 1, f"Expected traceback path, got: {paths}"

        points = s.get_string_points(tb_paths[0])
        values = [p["value"] for p in points]
        assert any("ValueError" in v and "intentional error" in v for v in values), (
            f"Traceback not found in: {values}"
        )
        s.close()

    def test_hardware_metrics_logged(self, tmp_path):
        """Hardware metrics are collected (psutil is a core dependency)."""
        r = Run(
            name="hw-test",
            goodseed_home=tmp_path,
            run_id="hw-run",
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=True,
            capture_traceback=False,
        )
        # Override hardware monitor interval for faster test
        if r._monitoring and r._monitoring._hardware_monitor is not None:
            r._monitoring._hardware_monitor._interval = 0.5
        time.sleep(2)
        db_path = r._db_path
        r.close()

        s = LocalStorage(db_path)
        paths = s.get_metric_paths()
        cpu_paths = [p for p in paths if "/cpu" in p]
        memory_paths = [p for p in paths if "/memory" in p]
        assert len(cpu_paths) >= 1, f"Expected cpu metric, got: {paths}"
        assert len(memory_paths) >= 1, f"Expected memory metric, got: {paths}"

        # Check that values are reasonable
        cpu_points = s.get_metric_points(cpu_paths[0])
        assert len(cpu_points) >= 1
        assert 0 <= cpu_points[0]["y"] <= 100
        s.close()


class TestServerMonitoring:
    """Test that server correctly reports monitoring data."""

    def test_scan_runs_includes_metric_paths(self, tmp_path):
        _create_run(
            tmp_path, run_id="mp-run",
            metrics=[{"train/loss": 1.0, "monitoring/abc/cpu": 50.0}],
        )
        projects_dir = tmp_path / "projects"
        runs = _scan_runs(projects_dir)
        assert len(runs) == 1
        assert "metric_paths" in runs[0]
        assert "train/loss" in runs[0]["metric_paths"]
        assert "monitoring/abc/cpu" in runs[0]["metric_paths"]


class TestDownsampleMetrics:
    """Tests for the downsample_metrics function."""

    def _create_metric_run(self, tmp_path, run_id, metric_path, values):
        """Helper: create a run with a single metric series."""
        r = Run(run_id=run_id, goodseed_home=tmp_path)
        for step, val in enumerate(values):
            r.log_metrics({metric_path: val}, step=step)
        r.close()
        return tmp_path / "projects" / "default" / "runs" / f"{run_id}.sqlite"

    def test_no_data(self, tmp_path):
        r = Run(run_id="empty-ds", goodseed_home=tmp_path)
        r.close()
        db_path = tmp_path / "projects" / "default" / "runs" / "empty-ds.sqlite"
        result = downsample_metrics(db_path, "nonexistent", point_count=10)
        assert result["points"] == []
        assert result["totalCount"] == 0
        assert result["downsampled"] is False

    def test_fewer_points_than_requested(self, tmp_path):
        db_path = self._create_metric_run(tmp_path, "few-pts", "loss", [0.9, 0.8, 0.7])
        result = downsample_metrics(db_path, "loss", point_count=100)
        assert len(result["points"]) == 3
        assert result["totalCount"] == 3
        assert result["downsampled"] is False
        assert result["points"][0]["value"] == 0.9
        assert result["points"][2]["value"] == 0.7

    def test_downsamples_when_too_many_points(self, tmp_path):
        values = [float(i) for i in range(100)]
        db_path = self._create_metric_run(tmp_path, "many-pts", "loss", values)
        result = downsample_metrics(db_path, "loss", point_count=10)
        assert result["totalCount"] == 100
        assert result["downsampled"] is True
        # Should produce roughly point_count buckets + bookends
        assert len(result["points"]) <= 15

    def test_bookend_points(self, tmp_path):
        values = [float(i) for i in range(64)]
        db_path = self._create_metric_run(tmp_path, "bookend", "loss", values)
        result = downsample_metrics(db_path, "loss", point_count=4)
        points = result["points"]
        # First and last raw points should be exact
        assert points[0]["step"] == 0
        assert points[-1]["step"] == 63

    def test_first_last_point_index(self, tmp_path):
        values = [float(i) for i in range(50)]
        db_path = self._create_metric_run(tmp_path, "slice", "loss", values)
        result = downsample_metrics(
            db_path, "loss", point_count=100,
            first_point_index=10, last_point_index=19,
        )
        assert result["totalCount"] == 50
        assert result["firstPointIndex"] == 10
        assert result["lastPointIndex"] == 19
        assert len(result["points"]) == 10

    def test_inverted_range_returns_empty(self, tmp_path):
        values = [float(i) for i in range(10)]
        db_path = self._create_metric_run(tmp_path, "invert", "loss", values)
        result = downsample_metrics(
            db_path, "loss", point_count=100,
            first_point_index=8, last_point_index=2,
        )
        assert result["points"] == []

    def test_min_max_in_buckets(self, tmp_path):
        # Create data with known min/max within each bucket
        values = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0, 40.0, 4.0]
        db_path = self._create_metric_run(tmp_path, "minmax", "loss", values)
        result = downsample_metrics(db_path, "loss", point_count=2)
        assert result["downsampled"] is True
        for point in result["points"]:
            if point["downsampled"]:
                assert point["minY"] <= point["value"] <= point["maxY"]

    def test_single_point(self, tmp_path):
        db_path = self._create_metric_run(tmp_path, "single", "loss", [0.5])
        result = downsample_metrics(db_path, "loss", point_count=100)
        assert result["totalCount"] == 1
        assert len(result["points"]) == 1
        assert result["points"][0]["value"] == 0.5
        assert result["downsampled"] is False


class TestGetStringSeries:
    """Tests for the read_string_series function."""

    def _create_string_run(self, tmp_path, run_id, path, values):
        """Helper: create a run with a string series."""
        r = Run(run_id=run_id, goodseed_home=tmp_path)
        for step, val in enumerate(values):
            r[path].log(val, step=step)
        r.close()
        return tmp_path / "projects" / "default" / "runs" / f"{run_id}.sqlite"

    def test_basic_read(self, tmp_path):
        db_path = self._create_string_run(
            tmp_path, "ss-basic", "log", ["a", "b", "c"]
        )
        result = read_string_series(db_path, "log")
        assert result["total"] == 3
        assert len(result["points"]) == 3
        assert [p["value"] for p in result["points"]] == ["a", "b", "c"]

    def test_limit_offset(self, tmp_path):
        db_path = self._create_string_run(
            tmp_path, "ss-page", "log", ["a", "b", "c", "d", "e"]
        )
        result = read_string_series(db_path, "log", limit=2, offset=1)
        assert result["total"] == 5
        assert len(result["points"]) == 2
        assert [p["value"] for p in result["points"]] == ["b", "c"]

    def test_tail(self, tmp_path):
        db_path = self._create_string_run(
            tmp_path, "ss-tail", "log", ["a", "b", "c", "d", "e"]
        )
        result = read_string_series(db_path, "log", tail=2)
        assert result["total"] == 5
        assert len(result["points"]) == 2
        assert [p["value"] for p in result["points"]] == ["d", "e"]

    def test_no_path_returns_all(self, tmp_path):
        r = Run(run_id="ss-all", goodseed_home=tmp_path)
        r["stdout"].log("hello", step=0)
        r["stderr"].log("oops", step=0)
        r.close()
        db_path = tmp_path / "projects" / "default" / "runs" / "ss-all.sqlite"
        result = read_string_series(db_path)
        assert result["total"] == 2
        paths = {p["path"] for p in result["points"]}
        assert paths == {"stderr", "stdout"}

    def test_empty_series(self, tmp_path):
        r = Run(run_id="ss-empty", goodseed_home=tmp_path)
        r.close()
        db_path = tmp_path / "projects" / "default" / "runs" / "ss-empty.sqlite"
        result = read_string_series(db_path, "nonexistent")
        assert result["total"] == 0
        assert result["points"] == []


class TestTrashRestore:
    """Tests for trash/restore HTTP endpoints."""

    @pytest.fixture
    def server(self, tmp_path):
        """Start a test server with a run."""
        _create_run(
            tmp_path, run_id="trash-run", name="trash-exp",
            configs={"lr": 0.001},
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

    def _post(self, url):
        req = Request(url, method="POST", data=b"")
        with urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())

    def test_trash_and_restore(self, server):
        # Starts not trashed
        data = self._get(f"{server}/api/runs")
        assert data["runs"][0]["trashed"] is False

        # Trash it
        result = self._post(f"{server}/api/runs/default/trash-run/trash")
        assert result["ok"] is True

        # Now shows as trashed
        data = self._get(f"{server}/api/runs")
        assert data["runs"][0]["trashed"] is True

        # Restore it
        result = self._post(f"{server}/api/runs/default/trash-run/trash/restore")
        assert result["ok"] is True

        # Back to not trashed
        data = self._get(f"{server}/api/runs")
        assert data["runs"][0]["trashed"] is False

    def test_trash_nonexistent_run(self, server):
        from urllib.error import HTTPError
        with pytest.raises(HTTPError) as exc_info:
            self._post(f"{server}/api/runs/default/nonexistent/trash")
        assert exc_info.value.code == 404
