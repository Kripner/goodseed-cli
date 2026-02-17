"""Unit tests for LocalStorage."""

import pytest
from pathlib import Path

from goodseed.storage import LocalStorage


@pytest.fixture
def storage(tmp_path):
    """Create a fresh LocalStorage instance."""
    db_path = tmp_path / "test.db"
    s = LocalStorage(db_path)
    yield s
    s.close()


class TestMeta:
    def test_set_and_get_meta(self, storage):
        storage.set_meta("run_name", "test-run")
        assert storage.get_meta("run_name") == "test-run"

    def test_get_missing_meta(self, storage):
        assert storage.get_meta("nonexistent") is None

    def test_overwrite_meta(self, storage):
        storage.set_meta("key", "v1")
        storage.set_meta("key", "v2")
        assert storage.get_meta("key") == "v2"


class TestConfigs:
    def test_log_and_get_configs(self, storage):
        storage.log_configs({
            "lr": ("float", "0.001"),
            "optimizer": ("str", "adam"),
        })
        configs = storage.get_configs()
        assert configs["lr"] == ("float", "0.001")
        assert configs["optimizer"] == ("str", "adam")

    def test_overwrite_config(self, storage):
        storage.log_configs({"lr": ("float", "0.001")})
        storage.log_configs({"lr": ("float", "0.01")})
        configs = storage.get_configs()
        assert configs["lr"] == ("float", "0.01")

    def test_empty_configs(self, storage):
        configs = storage.get_configs()
        assert configs == {}


class TestMetricPoints:
    def test_log_and_get_points(self, storage):
        points = [
            ("loss", 1, 0.9, 1735689600),
            ("loss", 2, 0.8, 1735689601),
            ("loss", 3, 0.7, 1735689602),
        ]
        storage.log_metric_points(points)
        result = storage.get_metric_points("loss")
        assert len(result) == 3
        assert result[0]["y"] == 0.9
        assert result[2]["y"] == 0.7

    def test_get_points_by_path(self, storage):
        points = [
            ("loss", 1, 0.9, 1735689600),
            ("acc", 1, 0.1, 1735689600),
        ]
        storage.log_metric_points(points)
        loss_points = storage.get_metric_points("loss")
        assert len(loss_points) == 1
        acc_points = storage.get_metric_points("acc")
        assert len(acc_points) == 1

    def test_get_all_points(self, storage):
        points = [
            ("loss", 1, 0.9, 1735689600),
            ("acc", 1, 0.1, 1735689600),
        ]
        storage.log_metric_points(points)
        all_points = storage.get_metric_points()
        assert len(all_points) == 2

    def test_get_metric_paths(self, storage):
        points = [
            ("loss", 1, 0.9, 1735689600),
            ("acc", 1, 0.1, 1735689600),
            ("loss", 2, 0.8, 1735689601),
        ]
        storage.log_metric_points(points)
        paths = storage.get_metric_paths()
        assert paths == ["acc", "loss"]

    def test_points_ordered_by_step(self, storage):
        points = [
            ("loss", 3, 0.7, 1735689602),
            ("loss", 1, 0.9, 1735689600),
            ("loss", 2, 0.8, 1735689601),
        ]
        storage.log_metric_points(points)
        result = storage.get_metric_points("loss")
        steps = [p["step"] for p in result]
        assert steps == [1, 2, 3]


class TestCheckpointWal:
    def test_checkpoint_wal(self, tmp_path):
        db_path = tmp_path / "wal_test.db"
        s = LocalStorage(db_path)
        s.set_meta("key", "value")
        s.checkpoint_wal()
        # After checkpoint, WAL should be empty/gone
        wal_path = Path(str(db_path) + "-wal")
        assert not wal_path.exists() or wal_path.stat().st_size == 0
        s.close()

    def test_checkpoint_then_close_single_file(self, tmp_path):
        db_path = tmp_path / "single.db"
        s = LocalStorage(db_path)
        s.set_meta("key", "value")
        s.log_configs({"lr": ("float", "0.001")})
        s.checkpoint_wal()
        s.close()
        # Only the main DB file should remain
        assert db_path.exists()
        assert not Path(str(db_path) + "-wal").exists()
        assert not Path(str(db_path) + "-shm").exists()


class TestLifecycle:
    def test_close_then_error(self, storage):
        storage.close()
        with pytest.raises(RuntimeError, match="closed"):
            storage.set_meta("key", "value")

    def test_context_manager(self, tmp_path):
        db_path = tmp_path / "ctx.db"
        with LocalStorage(db_path) as s:
            s.set_meta("key", "value")
            assert s.get_meta("key") == "value"

    def test_double_close(self, storage):
        storage.close()
        storage.close()  # Should not raise

    def test_db_file_created(self, tmp_path):
        db_path = tmp_path / "sub" / "test.db"
        s = LocalStorage(db_path)
        assert db_path.exists()
        s.close()

    def test_delete_db_file(self, tmp_path):
        db_path = tmp_path / "del.db"
        s = LocalStorage(db_path)
        s.set_meta("key", "value")
        assert db_path.exists()

        s.delete_db_file()
        assert not db_path.exists()

    def test_delete_db_file_cleans_wal(self, tmp_path):
        db_path = tmp_path / "wal.db"
        s = LocalStorage(db_path)
        s.set_meta("key", "value")
        s.close()

        # WAL files may or may not exist depending on checkpointing,
        # but delete_db_file should handle both cases
        s2 = LocalStorage(db_path)
        s2.delete_db_file()
        assert not db_path.exists()
        assert not Path(str(db_path) + "-wal").exists()
        assert not Path(str(db_path) + "-shm").exists()
