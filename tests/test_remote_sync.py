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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFullRemoteSync:
    """Create a run, log data, upload, then read it back from the server."""

    def test_full_remote_sync(self, tmp_path, api_key, test_project):
        # 1. Create run with remote sync enabled
        run = Run(
            project=test_project,
            name="integration-test",
            goodseed_home=tmp_path,
            api_key=api_key,
            storage="cloud",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
        )
        run_id = run.run_id

        # 2. Log configs
        run["learning_rate"] = 0.001
        run["batch_size"] = 32
        run["model"] = "test-mlp"

        # 3. Log metric series (10 steps)
        for step in range(10):
            loss = 1.0 - step * 0.1
            acc = step * 0.1
            run["train/loss"].log(loss, step=step)
            run["train/acc"].log(acc, step=step)

        # 4. Log string series
        run["notes"].log("started training", step=0)
        run["notes"].log("finished training", step=1)

        # 5. Close — blocks until all data uploaded
        run.close()

        # 6. Open read-only run and fetch data back
        ro = Run(
            project=test_project,
            run_id=run_id,
            api_key=api_key,
            read_only=True,
        )

        # 7. Verify metric paths
        metric_paths = ro.get_metric_paths()
        assert "train/loss" in metric_paths
        assert "train/acc" in metric_paths

        # 8. Verify metric data
        loss_data = ro.get_metric_data("train/loss")
        assert loss_data["path"] == "train/loss"
        raw_points = loss_data["raw_points"]
        assert len(raw_points) == 10
        steps = sorted(p["step"] for p in raw_points)
        assert steps[0] == 0
        assert steps[-1] == 9

        acc_data = ro.get_metric_data("train/acc")
        assert len(acc_data["raw_points"]) == 10

        # 9. Verify string paths
        string_paths = ro.get_string_paths()
        assert "notes" in string_paths

        # 10. Verify string data
        notes = ro.get_string_data("notes")
        values = {int(d["step"]): d["value"] for d in notes}
        assert values[0] == "started training"
        assert values[1] == "finished training"

        # 11. Verify configs
        configs = ro.get_configs()
        config_map = {c["path"]: c for c in configs}
        assert config_map["learning_rate"]["value"] == "0.001"
        assert config_map["batch_size"]["value"] == "32"
        assert config_map["model"]["value"] == "test-mlp"

        # 12. Verify read-only guards
        with pytest.raises(RuntimeError, match="read-only"):
            ro["train/loss"].log(999.0, step=99)

        with pytest.raises(RuntimeError, match="read-only"):
            ro["new_config"] = "value"

        # close() is a no-op for read-only runs
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

        # 2. Upload via upload_run()
        from goodseed.sync import upload_run

        db_path = (
            tmp_path / "projects" / test_project / "runs" / f"{run_id}.sqlite"
        )
        total = upload_run(db_path, api_key)
        assert total > 0

        # 3. Verify on server via read-only run
        ro = Run(
            project=test_project,
            run_id=run_id,
            api_key=api_key,
            read_only=True,
        )
        metric_paths = ro.get_metric_paths()
        assert "train/loss" in metric_paths

        loss_data = ro.get_metric_data("train/loss")
        assert len(loss_data["raw_points"]) == 2
