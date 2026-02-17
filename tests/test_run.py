"""Unit tests for Run class.

The Run class stores data locally as SQLite files. We use
goodseed_home=tmp_path to isolate DB files per test.
"""

from pathlib import Path

import pytest

from goodseed.run import Run
from goodseed.storage import LocalStorage


@pytest.fixture
def run(tmp_path):
    """Create a Run instance."""
    r = Run(
        experiment_name="test-experiment",
        project="test-project",
        goodseed_home=tmp_path,
    )
    yield r
    if not r._closed:
        r.close()


class TestRunInit:
    def test_creates_run(self, tmp_path):
        r = Run(
            experiment_name="exp1",
            goodseed_home=tmp_path,
        )
        assert r.run_name is not None
        assert r.experiment_name == "exp1"
        r.close()

    def test_custom_run_name(self, tmp_path):
        r = Run(
            experiment_name="exp1",
            run_name="my-run",
            goodseed_home=tmp_path,
        )
        assert r.run_name == "my-run"
        r.close()

    def test_default_project(self, tmp_path):
        r = Run(goodseed_home=tmp_path)
        assert r.project == "default"
        r.close()

    def test_stores_metadata(self, tmp_path):
        r = Run(
            experiment_name="exp1",
            project="proj1",
            goodseed_home=tmp_path,
        )
        assert r._storage.get_meta("project") == "proj1"
        assert r._storage.get_meta("experiment_name") == "exp1"
        assert r._storage.get_meta("run_name") == r.run_name
        assert r._storage.get_meta("created_at") is not None
        assert r._storage.get_meta("status") == "running"
        r.close()

    def test_db_path_default(self, tmp_path):
        r = Run(
            project="myproj",
            run_name="myrun",
            goodseed_home=tmp_path,
        )
        expected = tmp_path / "projects" / "myproj" / "runs" / "myrun.sqlite"
        assert r._db_path == expected
        assert expected.exists()
        r.close()

    def test_log_dir_override(self, tmp_path):
        log_dir = tmp_path / "custom_logs"
        log_dir.mkdir()
        r = Run(
            run_name="custom-run",
            log_dir=log_dir,
        )
        expected = log_dir / "custom-run.sqlite"
        assert r._db_path == expected
        assert expected.exists()
        r.close()

    def test_existing_db_raises(self, tmp_path):
        """If DB already exists, raise with instructions."""
        db_path = tmp_path / "projects" / "default" / "runs" / "stale-run.sqlite"
        db_path.parent.mkdir(parents=True)
        storage = LocalStorage(db_path)
        storage.close()

        with pytest.raises(RuntimeError, match="already exists"):
            Run(run_name="stale-run", goodseed_home=tmp_path)


class TestLogConfigs:
    def test_log_configs(self, run):
        run.log_configs({"learning_rate": 0.001, "batch_size": 32})
        configs = run._storage.get_configs()
        assert "learning_rate" in configs
        assert "batch_size" in configs

    def test_log_configs_types(self, run):
        run.log_configs({
            "flag": True,
            "name": "test",
            "count": 42,
            "rate": 3.14,
        })
        configs = run._storage.get_configs()
        assert configs["flag"] == ("bool", "true")
        assert configs["name"] == ("str", "test")
        assert configs["count"] == ("int", "42")
        assert configs["rate"] == ("float", "3.14")

    def test_log_configs_none(self, run):
        run.log_configs({"empty": None})
        configs = run._storage.get_configs()
        assert configs["empty"] == ("null", "")

    def test_log_configs_flatten(self, run):
        run.log_configs(
            {"model": {"hidden_size": 256, "layers": 4}},
            flatten=True,
        )
        configs = run._storage.get_configs()
        assert "model/hidden_size" in configs
        assert "model/layers" in configs

    def test_log_configs_overwrite(self, run):
        run.log_configs({"lr": 0.001})
        run.log_configs({"lr": 0.01})
        configs = run._storage.get_configs()
        assert configs["lr"] == ("float", "0.01")


class TestLogMetrics:
    def test_log_metrics(self, run):
        run.log_metrics({"loss": 0.9}, step=1)
        run.log_metrics({"loss": 0.8}, step=2)
        points = run._storage.get_metric_points("loss")
        assert len(points) == 2
        assert points[0]["y"] == 0.9
        assert points[1]["y"] == 0.8

    def test_log_multiple_metrics(self, run):
        run.log_metrics({"loss": 0.9, "acc": 0.1}, step=1)
        loss_points = run._storage.get_metric_points("loss")
        acc_points = run._storage.get_metric_points("acc")
        assert len(loss_points) == 1
        assert len(acc_points) == 1

    def test_step_is_int(self, run):
        run.log_metrics({"loss": 0.5}, step=1.5)
        points = run._storage.get_metric_points("loss")
        assert points[0]["step"] == 1


class TestClose:
    def test_close(self, run):
        run.close()
        assert run._closed is True

    def test_close_sets_status(self, tmp_path):
        r = Run(goodseed_home=tmp_path, run_name="status-test")
        db_path = r._db_path
        r.close()
        # Re-open the DB to check status
        s = LocalStorage(db_path)
        assert s.get_meta("status") == "finished"
        assert s.get_meta("closed_at") is not None
        s.close()

    def test_close_checkpoints_wal(self, tmp_path):
        r = Run(goodseed_home=tmp_path, run_name="wal-test")
        r.log_metrics({"loss": 0.5}, step=1)
        db_path = r._db_path
        r.close()
        # After close, only the .sqlite file should exist
        assert db_path.exists()
        assert not Path(str(db_path) + "-wal").exists()

    def test_double_close(self, run):
        run.close()
        run.close()  # Should not raise

    def test_log_after_close(self, run):
        run.close()
        with pytest.raises(RuntimeError, match="closed"):
            run.log_metrics({"loss": 0.5}, step=1)

    def test_config_after_close(self, run):
        run.close()
        with pytest.raises(RuntimeError, match="closed"):
            run.log_configs({"lr": 0.001})

    def test_close_with_failed_status(self, tmp_path):
        r = Run(goodseed_home=tmp_path, run_name="fail-test")
        db_path = r._db_path
        r.close(status="failed")
        s = LocalStorage(db_path)
        assert s.get_meta("status") == "failed"
        s.close()


class TestContextManager:
    def test_context_manager(self, tmp_path):
        with Run(
            experiment_name="ctx",
            goodseed_home=tmp_path,
        ) as r:
            r.log_metrics({"loss": 0.5}, step=1)
        assert r._closed is True

    def test_context_manager_sets_finished(self, tmp_path):
        with Run(goodseed_home=tmp_path, run_name="ctx-ok") as r:
            db_path = r._db_path
        s = LocalStorage(db_path)
        assert s.get_meta("status") == "finished"
        s.close()

    def test_context_manager_with_error(self, tmp_path):
        with pytest.raises(ValueError):
            with Run(
                experiment_name="err",
                goodseed_home=tmp_path,
            ) as r:
                r.log_metrics({"loss": 0.5}, step=1)
                raise ValueError("test error")
        assert r._closed is True

    def test_context_manager_error_sets_failed(self, tmp_path):
        with pytest.raises(ValueError):
            with Run(goodseed_home=tmp_path, run_name="ctx-fail") as r:
                db_path = r._db_path
                raise ValueError("boom")
        s = LocalStorage(db_path)
        assert s.get_meta("status") == "failed"
        s.close()


class TestDataPersistence:
    def test_data_readable_after_close(self, tmp_path):
        """Data should be readable from the .sqlite file after close."""
        with Run(
            experiment_name="persist",
            project="testproj",
            run_name="persist-run",
            goodseed_home=tmp_path,
        ) as r:
            r.log_configs({"lr": 0.001, "model": "resnet"})
            r.log_metrics({"loss": 0.9}, step=0)
            r.log_metrics({"loss": 0.5}, step=1)
            db_path = r._db_path

        # Re-open the DB file and verify data
        s = LocalStorage(db_path)
        configs = s.get_configs()
        assert configs["lr"] == ("float", "0.001")
        assert configs["model"] == ("str", "resnet")

        points = s.get_metric_points("loss")
        assert len(points) == 2
        assert points[0]["y"] == 0.9
        assert points[1]["y"] == 0.5

        paths = s.get_metric_paths()
        assert paths == ["loss"]
        s.close()
