"""End-to-end integration tests for remote sync.

These tests require a running backend and valid credentials.
They are skipped when the required environment variables are not set:

    GOODSEED_API_KEY          — API key for the remote API
    GOODSEED_TEST_PROJECT     — Project in "workspace/project" format

Optional:

    GOODSEED_API_URL — Override the API base URL (e.g. http://localhost:9090)
"""

import os

import pytest

from goodseed.run import Run
from goodseed.storage import LocalStorage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _print_api_base():
    """Log which API server the tests are targeting."""
    from goodseed.config import API_BASE
    print(f"\n  API_BASE = {API_BASE}")


@pytest.fixture
def api_key():
    key = os.environ.get("GOODSEED_API_KEY")
    if not key:
        pytest.skip("GOODSEED_API_KEY not set")
    return key


@pytest.fixture
def test_project():
    project = os.environ.get("GOODSEED_TEST_PROJECT")
    if not project:
        pytest.skip("GOODSEED_TEST_PROJECT not set")
    if "/" not in project:
        pytest.fail("GOODSEED_TEST_PROJECT must be in 'workspace/project' format")
    return project


def _db_path_for(tmp_path, test_project, run_id):
    return tmp_path / "projects" / test_project / "runs" / f"{run_id}.sqlite"


def _create_test_run(tmp_path, test_project, api_key=None, storage="local",
                     name="integration-test"):
    """Create a run, log standard test data, close it, return (run_id, db_path)."""
    run = Run(
        project=test_project,
        name=name,
        goodseed_home=tmp_path,
        api_key=api_key,
        storage=storage,
        capture_hardware_metrics=False,
        capture_stdout=False,
        capture_stderr=False,
        capture_traceback=False,
    )
    run_id = run.run_id

    # Configs
    run["learning_rate"] = 0.001
    run["batch_size"] = 32
    run["model"] = "test-mlp"

    # Metric series (10 steps)
    for step in range(10):
        run["train/loss"].log(1.0 - step * 0.1, step=step)
        run["train/acc"].log(step * 0.1, step=step)

    # String series
    run["notes"].log("started training", step=0)
    run["notes"].log("finished training", step=1)

    if storage == "cloud":
        run.sync()
        run.wait(timeout=10.0)

    run.close()

    db_path = _db_path_for(tmp_path, test_project, run_id)
    return run_id, db_path


# ---------------------------------------------------------------------------
# Local-only tests (no credentials needed)
# ---------------------------------------------------------------------------

class TestLocalLogging:
    """Create a run with storage='local', read it back via LocalStorage."""

    def test_local_run_lifecycle(self, tmp_path):
        project = "test-workspace/test-project"
        run = Run(
            project=project,
            name="local-test",
            goodseed_home=tmp_path,
            storage="local",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
        )
        run_id = run.run_id

        run["learning_rate"] = 0.001
        run["batch_size"] = 32
        run["model"] = "test-mlp"

        for step in range(10):
            run["train/loss"].log(1.0 - step * 0.1, step=step)
            run["train/acc"].log(step * 0.1, step=step)

        run["notes"].log("started training", step=0)
        run["notes"].log("finished training", step=1)

        run.close()

        # Read back via LocalStorage
        db_path = _db_path_for(tmp_path, project, run_id)
        assert db_path.exists(), f"DB not found at {db_path}"
        storage = LocalStorage(db_path)

        # Verify metric paths
        metric_paths = storage.get_metric_paths()
        assert "train/loss" in metric_paths
        assert "train/acc" in metric_paths

        # Verify metric data
        loss_points = storage.get_metric_points("train/loss")
        assert len(loss_points) == 10
        steps = sorted(p["step"] for p in loss_points)
        assert steps[0] == 0
        assert steps[-1] == 9

        acc_points = storage.get_metric_points("train/acc")
        assert len(acc_points) == 10

        # Verify string data
        string_paths = storage.get_string_series_paths()
        assert "notes" in string_paths

        notes = storage.get_string_points("notes")
        note_map = {int(p["step"]): p["value"] for p in notes}
        assert note_map[0] == "started training"
        assert note_map[1] == "finished training"

        # Verify configs
        configs = storage.get_configs()
        assert configs["learning_rate"][1] == "0.001"
        assert configs["batch_size"][1] == "32"
        assert configs["model"][1] == "test-mlp"

        storage.close()

    def test_local_read_only(self, tmp_path):
        """Open a local run in read-only mode via Run()."""
        project = "test-workspace/test-project"
        run = Run(
            project=project,
            name="local-ro-test",
            goodseed_home=tmp_path,
            storage="local",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
        )
        run_id = run.run_id
        run["train/loss"].log(1.0, step=0)
        run["train/loss"].log(0.5, step=1)
        run.close()

        # Re-open as read-only
        ro = Run(
            project=project,
            run_id=run_id,
            goodseed_home=tmp_path,
            storage="local",
            read_only=True,
        )

        # Read-only guards
        with pytest.raises(RuntimeError, match="read-only"):
            ro["train/loss"].log(999.0, step=99)
        with pytest.raises(RuntimeError, match="read-only"):
            ro["new_config"] = "value"

        # Data is accessible through internal storage
        assert ro._storage is not None
        assert len(ro._storage.get_metric_points("train/loss")) == 2

        ro.close()


# ---------------------------------------------------------------------------
# Cloud tests (require credentials)
# ---------------------------------------------------------------------------

class TestFullRemoteSync:
    """Create a run, log data, upload, then read it back from the server."""

    def test_full_remote_sync(self, tmp_path, api_key, test_project):
        run_id, db_path = _create_test_run(
            tmp_path, test_project, api_key=api_key, storage="cloud",
        )

        # Verify local DB was also written
        assert db_path.exists(), f"Local DB not found at {db_path}"
        local = LocalStorage(db_path)
        assert len(local.get_metric_points("train/loss")) == 10
        local.close()

        # Open cloud read-only run and fetch data back
        ro = Run(
            project=test_project,
            run_id=run_id,
            api_key=api_key,
            storage="cloud",
            read_only=True,
        )

        # Verify metric paths
        metric_paths = ro.get_metric_paths()
        assert "train/loss" in metric_paths
        assert "train/acc" in metric_paths

        # Verify metric data
        loss_data = ro.get_metric_data("train/loss")
        assert loss_data["path"] == "train/loss"
        raw_points = loss_data["raw_points"]
        assert len(raw_points) == 10
        steps = sorted(p["step"] for p in raw_points)
        assert steps[0] == 0
        assert steps[-1] == 9

        acc_data = ro.get_metric_data("train/acc")
        assert len(acc_data["raw_points"]) == 10

        # Verify string paths
        string_paths = ro.get_string_paths()
        assert "notes" in string_paths

        # Verify string data
        notes = ro.get_string_data("notes")
        values = {int(d["step"]): d["value"] for d in notes}
        assert values[0] == "started training"
        assert values[1] == "finished training"

        # Verify configs
        configs = ro.get_configs()
        config_map = {c["path"]: c for c in configs}
        assert config_map["learning_rate"]["value"] == "0.001"
        assert config_map["batch_size"]["value"] == "32"
        assert config_map["model"]["value"] == "test-mlp"

        # Verify read-only guards
        with pytest.raises(RuntimeError, match="read-only"):
            ro["train/loss"].log(999.0, step=99)

        with pytest.raises(RuntimeError, match="read-only"):
            ro["new_config"] = "value"

        ro.close()


class TestUploadCommand:
    """Test the manual upload path: local-only run → upload_run → verify."""

    def test_upload_then_read(self, tmp_path, api_key, test_project):
        # 1. Create run with storage="local" (local only)
        run = Run(
            project=test_project,
            goodseed_home=tmp_path,
            storage="local",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
        )
        run_id = run.run_id
        run["train/loss"].log(1.0, step=0)
        run["train/loss"].log(0.5, step=1)
        run.close()

        # 2. Verify data locally before upload
        db_path = _db_path_for(tmp_path, test_project, run_id)
        local = LocalStorage(db_path)
        assert len(local.get_metric_points("train/loss")) == 2
        local.close()

        # 3. Upload via upload_run()
        from goodseed.sync import upload_run

        total = upload_run(db_path, api_key)
        assert total > 0

        # 4. Verify on server via cloud read-only run
        ro = Run(
            project=test_project,
            run_id=run_id,
            api_key=api_key,
            storage="cloud",
            read_only=True,
        )
        metric_paths = ro.get_metric_paths()
        assert "train/loss" in metric_paths

        loss_data = ro.get_metric_data("train/loss")
        assert len(loss_data["raw_points"]) == 2
