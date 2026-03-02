# Goodseed - Architecture & API Reference

## Architecture

### Storage

Each run is a single SQLite file at `~/.goodseed/projects/<project>/runs/<run_id>.sqlite`.

SQLite WAL mode allows the training process to write while the server reads concurrently. On `close()`, the WAL is checkpointed so the result is one file.

Schema:

```sql
run_meta(key TEXT PRIMARY KEY, value TEXT)
configs(path TEXT PRIMARY KEY, type_tag TEXT, value TEXT, updated_at TEXT,
        uploaded INTEGER NOT NULL DEFAULT 0)
metric_series(id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL UNIQUE)
metric_points(series_id INTEGER, step REAL, y REAL, ts INTEGER,
              uploaded INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (series_id, step))
string_series(id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL UNIQUE)
string_points(series_id INTEGER, step REAL, value TEXT, ts INTEGER,
              uploaded INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (series_id, step))
```

The `uploaded` column tracks whether each row has been synced to the remote API. The background sync process reads rows with `uploaded=0`, uploads them, and marks them `uploaded=1` using optimistic concurrency (data values in the WHERE clause prevent marking rows overwritten since they were read).

### Server

`goodseed serve` starts a local HTTP server (`http.server.ThreadingHTTPServer`, no dependencies) that scans the projects directory and serves run data as JSON.

Endpoints:

```
GET /api/projects                                -- list all projects
GET /api/runs                                    -- list all runs
GET /api/runs?project=<name>                     -- list runs filtered by project
GET /api/runs/<project>/<run_id>/configs          -- config key-value pairs
GET /api/runs/<project>/<run_id>/metrics          -- all metric points
GET /api/runs/<project>/<run_id>/metrics?path=loss  -- filtered by path
GET /api/runs/<project>/<run_id>/metric-paths     -- list of metric names
GET /api/runs/<project>/<run_id>/string_series    -- all string series points
GET /api/runs/<project>/<run_id>/string_series?path=<path>           -- filtered by path
GET /api/runs/<project>/<run_id>/string_series?limit=50&offset=0     -- paginated
GET /api/runs/<project>/<run_id>/string_series?tail=20               -- last N entries
```

CORS is enabled (`Access-Control-Allow-Origin: *`) so the frontend at goodseed.ai can connect.

### Monitoring

Goodseed automatically captures system metrics and console output in a background thread, following Neptune's monitoring namespace convention.

**Namespace**: `monitoring/<8-char-hash>/` where the hash is derived from `hostname:pid:tid`. This ensures each process gets its own namespace.

**Console capture** (`ConsoleCaptureDaemon`, 1s interval):
- Wraps `sys.stdout` / `sys.stderr` with `StreamWithMemory` which buffers writes and forwards to the original stream
- Every second, drains buffered lines and logs them as string series at `monitoring/<hash>/stdout` and `monitoring/<hash>/stderr`
- Handles `\r` (carriage return) for progress bars — overwrites the current line

**Hardware metrics** (`HardwareMonitorDaemon`, 10s interval):
- CPU and memory utilisation via `psutil` (optional dependency)
- NVIDIA GPU via `nvidia-smi --query-gpu=... --format=csv` (ships with driver, no pip dependency)
- AMD GPU via `rocm-smi --showuse --showmemuse --showpower --json` (ships with driver, no pip dependency)
- Metrics: `cpu`, `memory`, `gpu`, `gpu_memory`, `gpu_power` (multi-GPU: `gpu_0`, `gpu_1`, ...)

**Traceback capture**: On exception inside a `with Run(...) as run:` block, the formatted traceback is logged as a string series at `monitoring/<hash>/traceback` and status is set to `failed`.

**Static metadata**: Logged as configs at `monitoring/<hash>/hostname`, `monitoring/<hash>/pid`, `monitoring/<hash>/tid`.

### Run Status

Runs have a `status` field in `run_meta`: `running` (default), `finished`, or `failed`. Status is set when `close()` is called. The context manager sets `failed` on exception, `finished` otherwise.

### Data Types

Configs are stored with type tags:

| Python Type | type_tag | Stored as |
|-------------|----------|-----------|
| `bool` | `"bool"` | `"true"` / `"false"` |
| `int` | `"int"` | `"42"` |
| `float` | `"float"` | `"3.14"` |
| `str` | `"str"` | `"hello"` |
| `datetime` | `"datetime"` | ISO 8601 string |
| `None` | `"null"` | `""` |
| `set` | `"string_set"` | JSON array `["a","b"]` |

Metrics are always `float`.

## API Reference

### `goodseed.Run`

```python
Run(
    name: str | None = None,            # display name (sys/name)
    description: str | None = None,     # free-form text (sys/description)
    tags: list[str] | None = None,      # tags (sys/tags)
    project: str | None = None,         # "workspace/project" format; default: GOODSEED_PROJECT or "default"
    run_id: str | None = None,          # unique ID; falls back to GOODSEED_RUN_ID env, then auto-generated
    resume_run_id: str | None = None,   # resume an existing run (mutually exclusive with run_id)
    storage: str | Storage | None = None,  # "disabled", "local", or "cloud"; env GOODSEED_STORAGE; default "cloud"
    api_key: str | None = None,         # API key; falls back to GOODSEED_API_KEY env
    read_only: bool = False,            # write methods raise; read behavior depends on storage mode
    goodseed_home: str | Path | None = None,  # override for ~/.goodseed
    log_dir: str | Path | None = None,  # override directory for the .sqlite file
    # Monitoring options (all default to True):
    capture_stdout: bool = True,        # capture print() output
    capture_stderr: bool = True,        # capture stderr output
    capture_hardware_metrics: bool = True,  # CPU, memory, GPU
    capture_traceback: bool = True,     # log traceback on exception
    monitoring_namespace: str | None = None,  # override "monitoring/<hash>"
)
```

#### Automatic namespaces

Each run automatically populates three namespaces:

| Namespace | Contents |
|-----------|----------|
| `sys/` | Run metadata: `id`, `name`, `description`, `tags`, `creation_time`, `state` |
| `monitoring/` | Hardware metrics (CPU, GPU), stdout/stderr streams, tracebacks |
| `source_code/` | Git info, diffs |

The `sys/state` field is updated to `finished` or `failed` when the run closes.

#### `run["key"] = value`

Set a config value, or assign a dictionary/namespace of values under a prefix.

```python
# Scalar values
run["score"] = 0.97
run["model_name"] = "resnet50"

# Dictionary (flattened under key as namespace)
run["parameters"] = {"lr": 0.001, "batch_size": 32}
# Stores: parameters/lr = 0.001, parameters/batch_size = 32

# Nested dictionary
run["parameters"] = {"train": {"max_epochs": 10}}
# Stores: parameters/train/max_epochs = 10

# argparse.Namespace
run["parameters"] = argparse.Namespace(lr=0.01, batch=32)
# Stores: parameters/lr = 0.01, parameters/batch = 32

# Edit sys/ fields
run["sys/name"] = "new-name"
run["sys/description"] = "updated description"
```

#### `run["key"].log(value, *, step)`

Log a value to a series. Numeric values (int, float) create metric series. String values create string series.

```python
# Log metrics
run["train/loss"].log(0.9, step=0)
run["train/loss"].log(0.8, step=1)

# Log with custom step
run["metric"].log(value=acc, step=i)

# Log string series
run["generated_text"].log("hello world", step=0)
```

`step` is required.

#### `run["sys/tags"].add(value)`

Add tags to the run. Accepts a single string or a list of strings.

```python
run["sys/tags"].add("production")
run["sys/tags"].add(["v2", "bert"])
```

#### `run.log_configs(data, flatten=True)`

Log configuration key-value pairs (batch method).

```python
run.log_configs({"learning_rate": 0.001, "optimizer": "adam"})

# Flatten nested dicts
run.log_configs({"model": {"hidden": 256, "layers": 4}}, flatten=True)
# Stores: "model/hidden" = 256, "model/layers" = 4
```

#### `run.log_metrics(data, step)`

Log metric values at a given step (batch method).

```python
run.log_metrics({"loss": 0.5, "accuracy": 0.85}, step=100)
```

- `step` can be `int` or `float`
- Same step + path overwrites the previous value
- Values are coerced to `float`

#### `run.log_strings(data, step)`

Log string series values at a given step (batch method).

```python
run.log_strings({"output": "Generated text here..."}, step=100)
```

- `step` can be `int` or `float`
- Same step + path overwrites the previous value
- Values are coerced to `str`

#### `run.close(status='finished')`

Close the run. When remote sync is enabled, `close()` blocks until all remaining data is uploaded. Then checkpoints the WAL and closes the database connection.

#### Resuming a run

```python
run = goodseed.Run(resume_run_id="bold-falcon")

# Continue logging
run["train/loss"].log(0.3, step=123)
run["eval/f1"] = 0.85
run.close()
```

- The run must not be currently running (status must be `finished` or `failed`)
- Auto-step state is restored from the existing data
- `run_id` and `resume_run_id` are mutually exclusive

#### Context Manager

```python
with goodseed.Run(name="exp") as run:
    run.log_metrics({"loss": 0.5}, step=1)
# status='finished' on normal exit, 'failed' on exception
```

## Storage Modes

The ``storage`` parameter (or ``GOODSEED_STORAGE`` env var) controls where data is stored:

| Mode | Local SQLite | Remote Sync | Remote Reads |
|------|-------------|-------------|--------------|
| ``cloud`` (default) | Yes | Yes | Yes |
| ``local`` | Yes | No | No |
| ``disabled`` | No | No | No |

When ``read_only=True``:

| Mode | Writes | Local Reads | Remote Reads |
|------|--------|-------------|--------------|
| ``cloud`` | Raise | No | Yes |
| ``local`` | Raise | Yes | No |
| ``disabled`` | Raise | Raise | Raise |

## Cloud Sync

When ``storage="cloud"`` (the default), a separate OS process runs in the background to upload data to the remote API. The SQLite database in WAL mode serves as the durable queue and IPC mechanism.

- **Sync process**: spawned via `multiprocessing` with `"spawn"` context. Reads unuploaded rows, sends them to the API, marks them uploaded on confirmed success.
- **Close behavior**: `run.close()` signals the sync process to drain all remaining data and blocks until upload is complete.
- **Orphan protection**: the sync process monitors the parent PID. If the parent dies (e.g. SIGKILL), it drains remaining data and exits.
- **Manual upload**: if sync was disabled or interrupted, use `goodseed upload -p <project> [--run-id <run_id>]` to upload remaining data.

Requires `GOODSEED_API_KEY` environment variable or `api_key` parameter.

## Data Fetching

Read data back from the remote API using ``storage="cloud"`` with ``read_only=True``. Read local data using ``storage="local"`` with ``read_only=True``.

```python
# Open a read-only handle to an existing run on the server
run = goodseed.Run(project="ws/proj", run_id="bold-falcon", api_key="gsk_...", read_only=True)

# Fetch metric paths and data
paths = run.get_metric_paths()              # ["train/loss", "train/acc"]
data = run.get_metric_data("train/loss")    # {path, downsampled, raw_points: [{step, y}]}

# Fetch string series
spaths = run.get_string_paths()             # ["notes", ...]
sdata = run.get_string_data("notes")        # [{step, value}]

# Fetch configs
configs = run.get_configs()                 # [{path, type_tag, value, updated_at}]
```

#### `run.get_metric_paths() -> list[str]`

Returns list of metric path strings available on the remote.

#### `run.get_metric_data(path, *, step_min=None, step_max=None, max_points=None) -> dict`

Returns dict with keys: `path`, `downsampled`, `raw_points` (list of `{step, y}`) or `buckets` (when downsampled).

#### `run.get_string_paths() -> list[str]`

Returns list of string series path strings available on the remote.

#### `run.get_string_data(path, *, limit=None, offset=None) -> list[dict]`

Returns list of dicts with keys `step` and `value`.

#### `run.get_configs() -> list[dict]`

Returns list of dicts with keys `path`, `type_tag`, `value`, `updated_at`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOODSEED_HOME` | `~/.goodseed` | Base directory for all data |
| `GOODSEED_PROJECT` | `default` | Default project name (`workspace/project` format for cloud storage) |
| `GOODSEED_RUN_ID` | — | Default run ID (overridden by `run_id` argument) |
| `GOODSEED_API_KEY` | — | API key for cloud storage |
| `GOODSEED_STORAGE` | `cloud` | Storage mode: `disabled`, `local`, or `cloud` |

## CLI Reference

### `goodseed serve [dir] [--port PORT]`

Start the local HTTP server. `dir` defaults to `~/.goodseed/projects`, port defaults to 8765.

### `goodseed list [dir] [--project NAME]`

List projects, or runs within a specific project when `--project` / `-p` is provided.

### `goodseed upload -p <project> [--run-id RUN_ID] [--api-key KEY]`

Upload unuploaded data from local run databases to the remote API. When `--run-id` is provided, uploads only that run; when omitted, uploads all runs in the project. Runs synchronously in the foreground. Useful for uploading data from runs where ``storage="local"`` was used, or where sync was interrupted.

## File Layout

```
goodseed/
  src/goodseed/
    __init__.py       # exports Run, GitRef, Storage
    run.py            # Run class
    storage.py        # LocalStorage (SQLite read/write)
    server.py         # HTTP server and read-only query functions
    config.py         # Environment config and path helpers
    utils.py          # Name generation, serialization, flattening
    cli.py            # CLI entry point
    sync.py           # Background sync process and upload_run()
    _sync_legacy.py   # Legacy Supabase sync (not used in current workflow)
    monitoring/
      __init__.py
      daemon.py             # MonitoringDaemon base class (background thread)
      console_capture.py    # StreamWithMemory, ConsoleCaptureDaemon
      hardware.py           # HardwareMonitorDaemon (CPU, memory, NVIDIA/AMD GPU)
  tests/
    conftest.py             # disables monitoring by default in tests
    test_storage.py
    test_run.py
    test_utils.py
    test_integration.py     # full workflow + HTTP server + monitoring tests
    test_cli.py
    test_neptune_proxy.py
  examples/
    mlp.py                  # PyTorch MLP on synthetic data (requires torch, sklearn)
```
