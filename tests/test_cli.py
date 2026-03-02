"""Unit tests for CLI commands."""

import pytest

from goodseed.cli import main
from goodseed.run import Run


class TestCmdList:
    def test_list_empty(self, tmp_path, capsys):
        main(["list", str(tmp_path)])
        output = capsys.readouterr().out
        assert "No projects found" in output

    def test_list_projects(self, tmp_path, capsys):
        with Run(run_id="cli-run", goodseed_home=tmp_path):
            pass

        projects_dir = tmp_path / "projects"
        main(["list", str(projects_dir)])
        output = capsys.readouterr().out
        assert "default" in output
        assert "1 project(s)" in output

    def test_list_runs_with_project(self, tmp_path, capsys):
        with Run(
            name="cli-exp",
            run_id="cli-run",
            goodseed_home=tmp_path,
        ) as r:
            r.log_configs({"lr": 0.001})

        projects_dir = tmp_path / "projects"
        main(["list", str(projects_dir), "--project", "default"])
        output = capsys.readouterr().out
        assert "cli-run" in output
        assert "cli-exp" in output
        assert "1 run(s)" in output

    def test_list_multiple_runs(self, tmp_path, capsys):
        with Run(run_id="run-a", goodseed_home=tmp_path):
            pass
        with Run(run_id="run-b", goodseed_home=tmp_path):
            pass

        projects_dir = tmp_path / "projects"
        main(["list", str(projects_dir), "-p", "default"])
        output = capsys.readouterr().out
        assert "run-a" in output
        assert "run-b" in output
        assert "2 run(s)" in output

    def test_list_nonexistent_dir(self, tmp_path, capsys):
        main(["list", str(tmp_path / "nonexistent")])
        output = capsys.readouterr().out
        assert "does not exist" in output


class TestCmdUpload:
    def test_upload_specific_run(self, tmp_path, monkeypatch, capsys):
        with Run(run_id="upload-one", goodseed_home=tmp_path):
            pass

        monkeypatch.setenv("GOODSEED_HOME", str(tmp_path))
        monkeypatch.setenv("GOODSEED_API_KEY", "test-key")

        called_paths = []

        def fake_upload_run(db_path, api_key):
            called_paths.append((db_path, api_key))
            return 7

        monkeypatch.setattr("goodseed.cli.upload_run", fake_upload_run)

        rc = main(["upload", "-r", "upload-one", "-p", "default"])
        output = capsys.readouterr().out

        assert rc == 0
        assert len(called_paths) == 1
        assert called_paths[0][0].name == "upload-one.sqlite"
        assert called_paths[0][1] == "test-key"
        assert "Uploading run 'upload-one'" in output
        assert "Done. Uploaded 7 item(s)." in output

    def test_upload_all_runs_when_run_id_omitted(self, tmp_path, monkeypatch, capsys):
        with Run(run_id="upload-a", goodseed_home=tmp_path):
            pass
        with Run(run_id="upload-b", goodseed_home=tmp_path):
            pass

        monkeypatch.setenv("GOODSEED_HOME", str(tmp_path))
        monkeypatch.setenv("GOODSEED_API_KEY", "test-key")

        called_names = []

        def fake_upload_run(db_path, api_key):
            called_names.append(db_path.stem)
            assert api_key == "test-key"
            return 3

        monkeypatch.setattr("goodseed.cli.upload_run", fake_upload_run)

        rc = main(["upload", "-p", "default"])
        output = capsys.readouterr().out

        assert rc == 0
        assert called_names == ["upload-a", "upload-b"]
        assert "Uploading run 'upload-a'" in output
        assert "Uploading run 'upload-b'" in output
        assert "Done. Uploaded 6 item(s)." in output

    def test_upload_all_runs_empty_project(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("GOODSEED_HOME", str(tmp_path))
        monkeypatch.setenv("GOODSEED_API_KEY", "test-key")

        rc = main(["upload", "-p", "default"])
        output = capsys.readouterr().out

        assert rc == 0
        assert "No runs found in project 'default'." in output


class TestCreateParser:
    def test_default_command(self):
        """When no command given, args.command should be None (defaults to serve)."""
        from goodseed.cli import create_parser
        parser = create_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_serve_with_port(self):
        from goodseed.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(["serve", "--port", "9999"])
        assert args.command == "serve"
        assert args.port == 9999

    def test_list_with_dir(self):
        from goodseed.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(["list", "/some/path"])
        assert args.command == "list"
        assert args.dir == "/some/path"

    def test_upload_without_run_id(self):
        from goodseed.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(["upload", "-p", "default"])
        assert args.command == "upload"
        assert args.run_id is None

    def test_upload_with_run_id_flag(self):
        from goodseed.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(["upload", "-p", "default", "--run-id", "abc123"])
        assert args.command == "upload"
        assert args.run_id == "abc123"
