# GoodSeed

ML experiment tracker. Logs metrics and configs to local SQLite files, serves them via a built-in HTTP server, and visualizes them in the browser.

Full documentation at [goodseed.ai/docs](https://goodseed.ai/docs/).

## Install

```bash
pip install goodseed
```

Python 3.9+ required.

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

Log metrics and configs from a training script:

```python
import goodseed

run = goodseed.Run(name="my-experiment", tags=["bert"])

# Log configs (single values)
run["learning_rate"] = 0.001
run["batch_size"] = 32

# Log metrics (series of values)
for epoch in range(100):
    loss = train_step()
    run["train/loss"].log(loss, step=epoch)

run.close()
```

Custom step values:

```python
run["metric"].log(value=acc, step=i)
```

`step` is required for series logging via `run["path"].log(...)`.

Batch logging methods are also available:

```python
run.log_configs({"learning_rate": 0.001, "batch_size": 32})
run.log_metrics({"loss": loss, "acc": acc}, step=step)
```

Your data is saved to a local SQLite file. You can also use `with goodseed.Run(...) as run:` to close the run automatically.

### Storage Modes

The `storage` parameter controls where data is stored:

- **`"cloud"`** (default) — local SQLite plus background sync to the remote API.
- **`"local"`** — local SQLite only, no remote sync.
- **`"disabled"`** — no storage; all writes are silent no-ops.

Cloud storage syncs data in the background while your training runs. Set the `GOODSEED_API_KEY` environment variable and use `workspace/project` format for the project name:

```python
import goodseed

run = goodseed.Run(project="my-workspace/my-project", name="experiment-1")
run["train/loss"].log(0.5, step=0)
run.close()  # blocks until all data is uploaded
```

For local-only storage (no remote sync):

```python
run = goodseed.Run(storage="local")
```

You can also set the mode via the `GOODSEED_STORAGE` environment variable.

### Read Data from Server

```python
run = goodseed.Run(project="my-workspace/my-project", run_id="bold-falcon", read_only=True)
data = run.get_metric_data("train/loss")
configs = run.get_configs()
```

### Resume a run

```python
run = goodseed.Run(resume_run_id="bold-falcon")
run["train/loss"].log(0.3, step=123)
run["eval/f1"] = 0.85
run.close()
```

### Monitoring

By default, goodseed automatically captures:

- **stdout / stderr** — every `print()` and warning is logged
- **Tracebacks** — captured on unhandled exceptions (status set to `failed`)
- **CPU & memory** — via `psutil` (installed with goodseed)
- **GPU** — NVIDIA (via `nvidia-smi`) and AMD (via `rocm-smi`), no pip dependency

Disable any of these with:

```python
run = goodseed.Run(
    capture_stdout=False,
    capture_stderr=False,
    capture_hardware_metrics=False,
    capture_traceback=False,
)
```

Then view your runs:

```bash
goodseed serve
```

Open the printed link in your browser to see your runs, metrics, and configs.

## Coming from Neptune?

You can export your data from [neptune.ai](https://neptune.ai) and import it into GoodSeed using [neptune-exporter](https://github.com/neptune-ai/neptune-exporter). See the [migration guide](https://docs.neptune.ai/transition_hub/migration/to_goodseed) for details.

## Configuration

| Variable | Description |
|----------|-------------|
| `GOODSEED_HOME` | Data directory (default: `~/.goodseed`) |
| `GOODSEED_PROJECT` | Default project name (default: `default`) |
| `GOODSEED_RUN_ID` | Default run ID (overridden by `run_id` argument) |
| `GOODSEED_API_KEY` | API key for cloud storage |
| `GOODSEED_STORAGE` | Storage mode: `disabled`, `local`, or `cloud` (default: `cloud`) |

## Git Tracking

By default, `Run()` auto-tracks Git metadata from the current repository:

- dirty state
- `source_code/diff` (index vs `HEAD`)
- last commit message, ID, author, and date
- current branch
- remotes
- `source_code/diff_upstream_<sha>` when `HEAD` differs from the remote tracking branch

Specify a custom repository path:

```python
import goodseed

run = goodseed.Run(git_ref=goodseed.GitRef(repository_path="/path/to/repo"))
```

Disable Git tracking:

```python
import goodseed

run = goodseed.Run(git_ref=False)
# or: run = goodseed.Run(git_ref=goodseed.GitRef.DISABLED)
```

## CLI

```bash
goodseed                   # Start the server (default command)
goodseed serve [dir]       # Start the server, optionally from a specific directory
goodseed serve --port 9000 # Use a custom port
goodseed list              # List projects
goodseed list -p default   # List runs in a project
goodseed upload -p <workspace/project> --run-id <run_id>  # Upload one run
goodseed upload -p <workspace/project>                    # Upload all runs
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Run the upstream Neptune exporter compatibility E2E test (disabled by default):

```bash
GOODSEED_RUN_NEPTUNE_EXPORTER_E2E=1 pytest tests/test_neptune_exporter_e2e.py -v
```

Optional:
- set `NEPTUNE_EXPORTER_DIR` to use an existing local clone
- by default, the test uses `archive/neptune-exporter` from this workspace

See [DOCS.md](DOCS.md) for architecture details and API reference.

## Beta Notice

GoodSeed is currently in beta. We may introduce breaking changes as we iterate on the product.

In particular, the local SQLite schema and parts of the Python/CLI interface may change in future releases. Depending on the change, upgrading could require migrating or recreating existing local GoodSeed data files (stored under `~/.goodseed` by default).

Feedback is very welcome while we stabilize the API. Please open issues with bug reports or feature requests.