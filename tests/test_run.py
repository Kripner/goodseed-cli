"""Unit tests for Run class.

The Run class stores data locally as SQLite files. We use
goodseed_home=tmp_path to isolate DB files per test.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from goodseed.run import GitRef, Run
from goodseed.storage import LocalStorage


@pytest.fixture
def run(tmp_path):
    """Create a Run instance."""
    r = Run(
        name="test-experiment",
        project="test-project",
        goodseed_home=tmp_path,
    )
    yield r
    if not r._closed:
        r.close()


class TestRunInit:
    def test_creates_run(self, tmp_path):
        r = Run(
            name="exp1",
            goodseed_home=tmp_path,
        )
        assert r.run_id is not None
        assert r.name == "exp1"
        r.close()

    def test_custom_run_id(self, tmp_path):
        r = Run(
            name="exp1",
            run_id="my-run",
            goodseed_home=tmp_path,
        )
        assert r.run_id == "my-run"
        r.close()

    def test_default_project(self, tmp_path):
        r = Run(goodseed_home=tmp_path)
        assert r.project == "default"
        r.close()

    def test_stores_metadata(self, tmp_path):
        r = Run(
            name="exp1",
            project="proj1",
            goodseed_home=tmp_path,
        )
        assert r._storage.get_meta("project") == "proj1"
        assert r._storage.get_meta("name") == "exp1"
        assert r._storage.get_meta("run_id") == r.run_id
        assert r._storage.get_meta("created_at") is not None
        assert r._storage.get_meta("status") == "running"
        r.close()

    def test_db_path_default(self, tmp_path):
        r = Run(
            project="myproj",
            run_id="myrun",
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
            run_id="custom-run",
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
            Run(run_id="stale-run", goodseed_home=tmp_path)

    def test_run_id_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GOODSEED_RUN_ID", "env-run-id")
        r = Run(goodseed_home=tmp_path)
        assert r.run_id == "env-run-id"
        r.close()

    def test_run_id_overrides_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GOODSEED_RUN_ID", "env-run-id")
        r = Run(run_id="explicit-id", goodseed_home=tmp_path)
        assert r.run_id == "explicit-id"
        r.close()

    def test_run_id_and_resume_run_id_exclusive(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot specify both"):
            Run(run_id="a", resume_run_id="b", goodseed_home=tmp_path)

    def test_run_constructor(self, tmp_path):
        r = Run(name="via-constructor", goodseed_home=tmp_path)
        assert r.name == "via-constructor"
        assert isinstance(r, Run)
        r.close()

    def test_constructor_rejects_positional_args(self):
        with pytest.raises(TypeError):
            Run("default")

    def test_constructor_accepts_experiment_name_alias(self, tmp_path):
        with pytest.warns(DeprecationWarning, match="experiment_name"):
            r = Run(experiment_name="legacy-name", goodseed_home=tmp_path)
        assert r.name == "legacy-name"
        r.close()

    def test_constructor_accepts_run_name_alias(self, tmp_path):
        with pytest.warns(DeprecationWarning, match="run_name"):
            r = Run(run_name="legacy-run", goodseed_home=tmp_path)
        assert r.run_id == "legacy-run"
        r.close()

    def test_constructor_rejects_name_and_experiment_name_together(self, tmp_path):
        with pytest.raises(ValueError, match="name"):
            Run(name="new", experiment_name="legacy", goodseed_home=tmp_path)

    def test_constructor_rejects_run_id_and_run_name_together(self, tmp_path):
        with pytest.raises(ValueError, match="run_id"):
            Run(run_id="new", run_name="legacy", goodseed_home=tmp_path)

    def test_constructor_rejects_unknown_kwarg(self, tmp_path):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            Run(goodseed_home=tmp_path, not_a_real_kwarg=True)


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

    def test_float_step(self, run):
        run.log_metrics({"loss": 0.5}, step=1.5)
        points = run._storage.get_metric_points("loss")
        assert points[0]["step"] == 1.5

    def test_step_is_required(self, run):
        with pytest.raises(TypeError):
            run.log_metrics({"loss": 0.9})


class TestClose:
    def test_close(self, run):
        run.close()
        assert run._closed is True

    def test_close_sets_status(self, tmp_path):
        r = Run(goodseed_home=tmp_path, run_id="status-test")
        db_path = r._db_path
        r.close()
        # Re-open the DB to check status
        s = LocalStorage(db_path)
        assert s.get_meta("status") == "finished"
        assert s.get_meta("closed_at") is not None
        s.close()

    def test_close_checkpoints_wal(self, tmp_path):
        r = Run(goodseed_home=tmp_path, run_id="wal-test")
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
        r = Run(goodseed_home=tmp_path, run_id="fail-test")
        db_path = r._db_path
        r.close(status="failed")
        s = LocalStorage(db_path)
        assert s.get_meta("status") == "failed"
        s.close()

    def test_close_marks_run_closed(self, tmp_path):
        r = Run(goodseed_home=tmp_path, run_id="close-test")
        r.close()
        assert r._closed is True


class TestContextManager:
    def test_context_manager(self, tmp_path):
        with Run(
            name="ctx",
            goodseed_home=tmp_path,
        ) as r:
            r.log_metrics({"loss": 0.5}, step=1)
        assert r._closed is True

    def test_context_manager_sets_finished(self, tmp_path):
        with Run(goodseed_home=tmp_path, run_id="ctx-ok") as r:
            db_path = r._db_path
        s = LocalStorage(db_path)
        assert s.get_meta("status") == "finished"
        s.close()

    def test_context_manager_with_error(self, tmp_path):
        with pytest.raises(ValueError):
            with Run(
                name="err",
                goodseed_home=tmp_path,
            ) as r:
                r.log_metrics({"loss": 0.5}, step=1)
                raise ValueError("test error")
        assert r._closed is True

    def test_context_manager_error_sets_failed(self, tmp_path):
        with pytest.raises(ValueError):
            with Run(goodseed_home=tmp_path, run_id="ctx-fail") as r:
                db_path = r._db_path
                raise ValueError("boom")
        s = LocalStorage(db_path)
        assert s.get_meta("status") == "failed"
        s.close()


class TestDataPersistence:
    def test_data_readable_after_close(self, tmp_path):
        """Data should be readable from the .sqlite file after close."""
        with Run(
            name="persist",
            project="testproj",
            run_id="persist-run",
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


def _git(*args, cwd):
    """Run git command in tests."""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def git_repo(tmp_path):
    """Create a git repository with one commit."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")

    repo = tmp_path / "repo"
    repo.mkdir()

    _git("init", "-b", "main", cwd=repo)
    _git("config", "user.name", "Goodseed Test", cwd=repo)
    _git("config", "user.email", "test@goodseed.local", cwd=repo)
    _git("remote", "add", "origin", "https://example.com/org/repo.git", cwd=repo)

    tracked = repo / "tracked.txt"
    tracked.write_text("line-1\n")
    _git("add", "tracked.txt", cwd=repo)
    _git("commit", "-m", "initial commit", cwd=repo)

    return repo


class TestGitTracking:
    def test_git_info_logged_by_default(self, tmp_path, monkeypatch, git_repo):
        monkeypatch.chdir(git_repo)
        run = Run(goodseed_home=tmp_path, run_id="git-default")
        db_path = run._db_path
        run.close()

        storage = LocalStorage(db_path)
        configs = storage.get_configs()
        storage.close()

        assert configs["source_code/git/repository_path"] == ("str", str(git_repo))
        assert configs["source_code/git/dirty"] == ("bool", "false")
        assert configs["source_code/git/commit_message"] == ("str", "initial commit")
        assert "source_code/git/commit_id" in configs
        assert "source_code/git/commit_author" in configs
        assert "source_code/git/commit_date" in configs
        assert configs["source_code/git/current_branch"] == ("str", "main")
        assert "origin" in configs["source_code/git/remotes"][1]
        assert configs["source_code/diff"] == ("str", "")

    def test_git_tracking_can_be_disabled(self, tmp_path, monkeypatch, git_repo):
        monkeypatch.chdir(git_repo)
        run = Run(goodseed_home=tmp_path, run_id="git-disabled", git_ref=False)
        db_path = run._db_path
        run.close()

        storage = LocalStorage(db_path)
        configs = storage.get_configs()
        storage.close()
        assert not any(k.startswith("source_code/") for k in configs)

    def test_git_tracking_can_be_disabled_with_constant(self, tmp_path, monkeypatch, git_repo):
        monkeypatch.chdir(git_repo)
        run = Run(
            goodseed_home=tmp_path,
            run_id="git-disabled-constant",
            git_ref=GitRef.DISABLED,
        )
        db_path = run._db_path
        run.close()

        storage = LocalStorage(db_path)
        configs = storage.get_configs()
        storage.close()
        assert not any(k.startswith("source_code/") for k in configs)

    def test_git_tracking_supports_custom_repository_path(self, tmp_path, monkeypatch, git_repo):
        monkeypatch.chdir(tmp_path)
        run = Run(
            goodseed_home=tmp_path,
            run_id="git-custom-path",
            git_ref=GitRef(repository_path=git_repo),
        )
        db_path = run._db_path
        run.close()

        storage = LocalStorage(db_path)
        configs = storage.get_configs()
        storage.close()
        assert configs["source_code/git/repository_path"] == ("str", str(git_repo))

    def test_git_tracking_logs_dirty_state_and_diff(self, tmp_path, monkeypatch, git_repo):
        monkeypatch.chdir(git_repo)
        tracked = git_repo / "tracked.txt"
        tracked.write_text("line-1\nline-2\n")

        run = Run(goodseed_home=tmp_path, run_id="git-dirty")
        db_path = run._db_path
        run.close()

        storage = LocalStorage(db_path)
        configs = storage.get_configs()
        storage.close()
        assert configs["source_code/git/dirty"] == ("bool", "true")
        assert "+line-2" in configs["source_code/diff"][1]

    def test_git_tracking_logs_upstream_diff_when_branch_is_ahead(
        self, tmp_path, monkeypatch, git_repo
    ):
        remote = tmp_path / "origin.git"
        _git("init", "--bare", str(remote), cwd=tmp_path)

        _git("remote", "remove", "origin", cwd=git_repo)
        _git("remote", "add", "origin", str(remote), cwd=git_repo)
        _git("push", "-u", "origin", "main", cwd=git_repo)

        tracked = git_repo / "tracked.txt"
        tracked.write_text("line-1\nline-ahead\n")
        _git("add", "tracked.txt", cwd=git_repo)
        _git("commit", "-m", "ahead commit", cwd=git_repo)

        monkeypatch.chdir(git_repo)
        run = Run(goodseed_home=tmp_path, run_id="git-upstream")
        db_path = run._db_path
        run.close()

        storage = LocalStorage(db_path)
        configs = storage.get_configs()
        storage.close()

        upstream_keys = [
            k for k in configs.keys() if k.startswith("source_code/diff_upstream_")
        ]
        assert len(upstream_keys) == 1


class TestNeptuneStyleAPI:
    """Tests for run["key"] = value and run["key"].log(value) syntax."""

    def test_setitem_logs_config(self, run):
        run["score"] = 0.97
        configs = run._storage.get_configs()
        assert configs["score"] == ("float", "0.97")

    def test_setitem_nested_path(self, run):
        run["test/acc"] = 0.95
        configs = run._storage.get_configs()
        assert configs["test/acc"] == ("float", "0.95")

    def test_setitem_string(self, run):
        run["model_name"] = "resnet50"
        configs = run._storage.get_configs()
        assert configs["model_name"] == ("str", "resnet50")

    def test_setitem_overwrite(self, run):
        run["lr"] = 0.001
        run["lr"] = 0.01
        configs = run._storage.get_configs()
        assert configs["lr"] == ("float", "0.01")

    def test_log_metric(self, run):
        run["train/loss"].log(0.9, step=0)
        run["train/loss"].log(0.8, step=1)
        run["train/loss"].log(0.7, step=2)
        points = run._storage.get_metric_points("train/loss")
        assert len(points) == 3
        assert [p["y"] for p in points] == [0.9, 0.8, 0.7]
        assert [p["step"] for p in points] == [0, 1, 2]

    def test_log_with_explicit_step(self, run):
        run["metric"].log(1.0, step=0)
        run["metric"].log(2.0, step=10)
        run["metric"].log(3.0, step=20)
        points = run._storage.get_metric_points("metric")
        assert [p["step"] for p in points] == [0, 10, 20]
        assert [p["y"] for p in points] == [1.0, 2.0, 3.0]

    def test_log_step_is_required(self, run):
        with pytest.raises(TypeError):
            run["metric"].log(1.0)

    def test_log_float_step(self, run):
        run["metric"].log(1.0, step=0.5)
        run["metric"].log(2.0, step=1.5)
        points = run._storage.get_metric_points("metric")
        assert [p["step"] for p in points] == [0.5, 1.5]

    def test_log_string_series(self, run):
        run["generated_text"].log("hello", step=0)
        run["generated_text"].log("world", step=1)
        points = run._storage.get_string_points("generated_text")
        assert len(points) == 2
        assert [p["value"] for p in points] == ["hello", "world"]
        assert [p["step"] for p in points] == [0, 1]

    def test_log_string_with_step(self, run):
        run["log"].log("start", step=0)
        run["log"].log("end", step=100)
        points = run._storage.get_string_points("log")
        assert [p["step"] for p in points] == [0, 100]

    def test_log_string_series_alias(self, run):
        with pytest.warns(DeprecationWarning, match="log_string_series"):
            run.log_string_series({"generated_text": "hello"}, step=0)
        points = run._storage.get_string_points("generated_text")
        assert len(points) == 1
        assert points[0]["value"] == "hello"
        assert points[0]["step"] == 0

    def test_log_int_value(self, run):
        run["count"].log(1, step=0)
        run["count"].log(2, step=1)
        points = run._storage.get_metric_points("count")
        assert [p["y"] for p in points] == [1.0, 2.0]

    def test_log_rejects_bool(self, run):
        with pytest.raises(TypeError, match="Unsupported"):
            run["flag"].log(True, step=0)

    def test_log_rejects_unsupported_type(self, run):
        with pytest.raises(TypeError, match="Unsupported"):
            run["data"].log([1, 2, 3], step=0)

    def test_multiple_paths_with_explicit_steps(self, run):
        run["loss"].log(0.9, step=0)
        run["loss"].log(0.8, step=1)
        run["acc"].log(0.1, step=0)
        loss = run._storage.get_metric_points("loss")
        acc = run._storage.get_metric_points("acc")
        assert [p["step"] for p in loss] == [0, 1]
        assert [p["step"] for p in acc] == [0]

    def test_mixing_apis(self, run):
        """Batch log_metrics and bracket API can coexist."""
        run.log_metrics({"loss": 0.9}, step=5)
        run["loss"].log(0.8, step=6)
        points = run._storage.get_metric_points("loss")
        assert [p["step"] for p in points] == [5, 6]

    def test_setitem_after_close_raises(self, run):
        run.close()
        with pytest.raises(RuntimeError, match="closed"):
            run["score"] = 0.5

    def test_log_after_close_raises(self, run):
        run.close()
        with pytest.raises(RuntimeError, match="closed"):
            run["loss"].log(0.5, step=0)

    def test_setitem_dict_flattens(self, run):
        run["parameters"] = {"lr": 0.001, "batch_size": 32}
        configs = run._storage.get_configs()
        assert configs["parameters/lr"] == ("float", "0.001")
        assert configs["parameters/batch_size"] == ("int", "32")

    def test_setitem_nested_dict(self, run):
        run["parameters"] = {"train": {"max_epochs": 10}}
        configs = run._storage.get_configs()
        assert configs["parameters/train/max_epochs"] == ("int", "10")

    def test_setitem_argparse_namespace(self, run):
        import argparse
        args = argparse.Namespace(lr=0.01, batch=32, activation="ReLU")
        run["parameters"] = args
        configs = run._storage.get_configs()
        assert configs["parameters/lr"] == ("float", "0.01")
        assert configs["parameters/batch"] == ("int", "32")
        assert configs["parameters/activation"] == ("str", "ReLU")

    def test_log_configs_nested_default_flattens(self, run):
        run.log_configs({
            "parameters": {
                "epoch_nr": 5,
                "batch_size": 32,
            },
        })
        configs = run._storage.get_configs()
        assert configs["parameters/epoch_nr"] == ("int", "5")
        assert configs["parameters/batch_size"] == ("int", "32")

    def test_log_configs_mixed_default_flattens(self, run):
        run.log_configs({"score": 0.97, "params": {"lr": 0.01}})
        configs = run._storage.get_configs()
        assert configs["score"] == ("float", "0.97")
        assert configs["params/lr"] == ("float", "0.01")


class TestSysNamespace:
    """Tests for the sys/ namespace (Neptune-style run metadata)."""

    def test_sys_id_logged(self, tmp_path):
        r = Run(run_id="my-run", goodseed_home=tmp_path)
        configs = r._storage.get_configs()
        assert configs["sys/id"] == ("str", "my-run")
        r.close()

    def test_sys_creation_time(self, tmp_path):
        r = Run(goodseed_home=tmp_path)
        configs = r._storage.get_configs()
        assert "sys/creation_time" in configs
        assert configs["sys/creation_time"][0] == "str"
        r.close()

    def test_sys_state_running(self, tmp_path):
        r = Run(goodseed_home=tmp_path)
        configs = r._storage.get_configs()
        assert configs["sys/state"] == ("str", "running")
        r.close()

    def test_sys_state_updated_on_close(self, tmp_path):
        r = Run(goodseed_home=tmp_path, run_id="state-test")
        db_path = r._db_path
        r.close()
        s = LocalStorage(db_path)
        configs = s.get_configs()
        assert configs["sys/state"] == ("str", "finished")
        assert "sys/closed_time" in configs
        s.close()

    def test_sys_state_failed(self, tmp_path):
        r = Run(goodseed_home=tmp_path, run_id="fail-state")
        db_path = r._db_path
        r.close(status="failed")
        s = LocalStorage(db_path)
        configs = s.get_configs()
        assert configs["sys/state"] == ("str", "failed")
        s.close()

    def test_sys_name(self, tmp_path):
        r = Run(name="my-experiment", goodseed_home=tmp_path)
        configs = r._storage.get_configs()
        assert configs["sys/name"] == ("str", "my-experiment")
        r.close()

    def test_sys_name_absent_when_not_provided(self, tmp_path):
        r = Run(goodseed_home=tmp_path)
        configs = r._storage.get_configs()
        assert "sys/name" not in configs
        r.close()

    def test_description(self, tmp_path):
        r = Run(
            description="NN trained on MNIST with high LR",
            goodseed_home=tmp_path,
        )
        configs = r._storage.get_configs()
        assert configs["sys/description"] == ("str", "NN trained on MNIST with high LR")
        r.close()

    def test_description_absent_when_not_provided(self, tmp_path):
        r = Run(goodseed_home=tmp_path)
        configs = r._storage.get_configs()
        assert "sys/description" not in configs
        r.close()

    def test_description_editable_via_setitem(self, run):
        run["sys/description"] = "updated description"
        configs = run._storage.get_configs()
        assert configs["sys/description"] == ("str", "updated description")

    def test_name_editable_via_setitem(self, run):
        run["sys/name"] = "new-name"
        configs = run._storage.get_configs()
        assert configs["sys/name"] == ("str", "new-name")


class TestTags:
    """Tests for tags support (sys/tags string set)."""

    def test_tags_at_init(self, tmp_path):
        r = Run(
            tags=["bert", "production"],
            goodseed_home=tmp_path,
        )
        configs = r._storage.get_configs()
        assert configs["sys/tags"][0] == "string_set"
        tags = json.loads(configs["sys/tags"][1])
        assert set(tags) == {"bert", "production"}
        r.close()

    def test_tags_add_single(self, run):
        run["sys/tags"].add("v1")
        configs = run._storage.get_configs()
        tags = json.loads(configs["sys/tags"][1])
        assert "v1" in tags

    def test_tags_add_list(self, run):
        run["sys/tags"].add(["v1", "bert"])
        configs = run._storage.get_configs()
        tags = json.loads(configs["sys/tags"][1])
        assert set(tags) == {"v1", "bert"}

    def test_tags_add_merges(self, tmp_path):
        r = Run(tags=["bert"], goodseed_home=tmp_path)
        r["sys/tags"].add("production")
        r["sys/tags"].add(["v2", "bert"])  # duplicate bert
        configs = r._storage.get_configs()
        tags = json.loads(configs["sys/tags"][1])
        assert set(tags) == {"bert", "production", "v2"}
        r.close()

    def test_tags_remove_single(self, tmp_path):
        r = Run(tags=["bert", "production"], goodseed_home=tmp_path)
        r["sys/tags"].remove("production")
        configs = r._storage.get_configs()
        tags = json.loads(configs["sys/tags"][1])
        assert set(tags) == {"bert"}
        r.close()

    def test_tags_remove_list(self, tmp_path):
        r = Run(tags=["bert", "production", "v2"], goodseed_home=tmp_path)
        r["sys/tags"].remove(["production", "missing"])
        configs = r._storage.get_configs()
        tags = json.loads(configs["sys/tags"][1])
        assert set(tags) == {"bert", "v2"}
        r.close()

    def test_tags_add_after_close_raises(self, run):
        run.close()
        with pytest.raises(RuntimeError, match="closed"):
            run["sys/tags"].add("tag")

    def test_tags_remove_after_close_raises(self, run):
        run.close()
        with pytest.raises(RuntimeError, match="closed"):
            run["sys/tags"].remove("tag")

    def test_tags_add_existing_non_set_raises(self, run):
        run["sys/tags"] = "production"
        with pytest.raises(TypeError, match="existing type"):
            run["sys/tags"].add("v2")

    def test_tags_remove_existing_non_set_raises(self, run):
        run["sys/tags"] = "production"
        with pytest.raises(TypeError, match="existing type"):
            run["sys/tags"].remove("v2")

    def test_tags_remove_missing_is_noop(self, run):
        run["sys/tags"].remove("missing")
        configs = run._storage.get_configs()
        assert "sys/tags" not in configs

    def test_no_tags_means_no_sys_tags(self, tmp_path):
        r = Run(goodseed_home=tmp_path)
        configs = r._storage.get_configs()
        assert "sys/tags" not in configs
        r.close()


class TestResumeRun:
    """Tests for resuming an existing run."""

    def test_resume_basic(self, tmp_path):
        # Create and close a run
        r1 = Run(run_id="resume-me", goodseed_home=tmp_path)
        r1.log_metrics({"loss": 0.9}, step=0)
        r1.log_metrics({"loss": 0.8}, step=1)
        r1["score"] = 0.5
        db_path = r1._db_path
        r1.close()

        # Resume it
        r2 = Run(resume_run_id="resume-me", goodseed_home=tmp_path)
        assert r2.run_id == "resume-me"
        assert r2._db_path == db_path

        # Can log new data
        r2.log_metrics({"loss": 0.7}, step=2)
        r2["score"] = 0.8

        points = r2._storage.get_metric_points("loss")
        assert len(points) == 3

        configs = r2._storage.get_configs()
        assert configs["score"] == ("float", "0.8")
        r2.close()

    def test_resume_continues_with_explicit_steps(self, tmp_path):
        r1 = Run(run_id="step-resume", goodseed_home=tmp_path)
        r1["loss"].log(0.9, step=0)
        r1["loss"].log(0.8, step=1)
        r1["loss"].log(0.7, step=2)
        r1.close()

        r2 = Run(resume_run_id="step-resume", goodseed_home=tmp_path)
        r2["loss"].log(0.6, step=3)
        r2["loss"].log(0.5, step=4)

        points = r2._storage.get_metric_points("loss")
        steps = [p["step"] for p in points]
        assert steps == [0, 1, 2, 3, 4]
        r2.close()

    def test_resume_running_raises(self, tmp_path):
        r1 = Run(run_id="still-running", goodseed_home=tmp_path)
        # Don't close r1, it's still running

        with pytest.raises(RuntimeError, match="still running"):
            Run(resume_run_id="still-running", goodseed_home=tmp_path)

        r1.close()

    def test_resume_nonexistent_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="not found"):
            Run(resume_run_id="does-not-exist", goodseed_home=tmp_path)

    def test_resume_sets_state_running(self, tmp_path):
        r1 = Run(run_id="state-resume", goodseed_home=tmp_path)
        r1.close()

        r2 = Run(resume_run_id="state-resume", goodseed_home=tmp_path)
        configs = r2._storage.get_configs()
        assert configs["sys/state"] == ("str", "running")
        assert r2._storage.get_meta("status") == "running"
        r2.close()

    def test_resume_with_log_dir(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        r1 = Run(run_id="logdir-run", log_dir=log_dir)
        r1.log_metrics({"acc": 0.5}, step=0)
        r1.close()

        r2 = Run(resume_run_id="logdir-run", log_dir=log_dir)
        r2.log_metrics({"acc": 0.9}, step=1)
        points = r2._storage.get_metric_points("acc")
        assert len(points) == 2
        r2.close()

    def test_resume_string_series_with_explicit_steps(self, tmp_path):
        r1 = Run(run_id="str-resume", goodseed_home=tmp_path)
        r1["log"].log("line1", step=0)
        r1["log"].log("line2", step=1)
        r1.close()

        r2 = Run(resume_run_id="str-resume", goodseed_home=tmp_path)
        r2["log"].log("line3", step=2)
        points = r2._storage.get_string_points("log")
        steps = [p["step"] for p in points]
        assert steps == [0, 1, 2]
        r2.close()

    def test_resume_failed_run(self, tmp_path):
        r1 = Run(run_id="failed-run", goodseed_home=tmp_path)
        r1.close(status="failed")

        # Should be able to resume a failed run
        r2 = Run(resume_run_id="failed-run", goodseed_home=tmp_path)
        r2["recovery_metric"].log(1.0, step=0)
        r2.close()

    def test_resume_adds_new_fields(self, tmp_path):
        r1 = Run(run_id="new-fields", goodseed_home=tmp_path)
        r1["train/loss"].log(0.9, step=0)
        r1.close()

        r2 = Run(resume_run_id="new-fields", goodseed_home=tmp_path)
        r2["eval/loss"].log(0.3, step=0)
        r2["f1_score"] = 0.85

        configs = r2._storage.get_configs()
        assert configs["f1_score"] == ("float", "0.85")
        eval_points = r2._storage.get_metric_points("eval/loss")
        assert len(eval_points) == 1
        r2.close()
