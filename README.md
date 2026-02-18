# GoodSeed

ML experiment tracker. Logs metrics and configs to local SQLite files, serves them via a built-in HTTP server, and visualizes them in the browser.

Full documentation at [goodseed.ai/docs](https://goodseed.ai/docs/).

## Install

```bash
pip install goodseed
```

Python 3.9+ required. No runtime dependencies.

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

Log metrics and configs from a training script:

```python
import goodseed

run = goodseed.Run(experiment_name="my-experiment")
run.log_configs({"learning_rate": 0.001, "batch_size": 32})

for step in range(100):
    loss = train_step()
    run.log_metrics({"loss": loss}, step=step)

run.close()
```

Your data is saved to a local SQLite file. You can also use `with goodseed.Run(...) as run:` to close the run automatically.

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

## CLI

```bash
goodseed                   # Start the server (default command)
goodseed serve [dir]       # Start the server, optionally from a specific directory
goodseed serve --port 9000 # Use a custom port
goodseed list              # List projects
goodseed list -p default   # List runs in a project
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

See [DOCS.md](DOCS.md) for architecture details and API reference.
