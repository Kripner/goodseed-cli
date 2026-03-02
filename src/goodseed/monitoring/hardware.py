"""Hardware metrics collection (CPU, memory, GPU).

Collects system metrics at a configurable interval and logs them
as metric series.

- psutil for CPU and memory utilisation
- nvidia-smi for NVIDIA GPU utilisation, memory, and power draw
- rocm-smi for AMD GPU utilisation, memory, and power draw

GPU metrics use CLI tools that ship with their respective drivers,
so no pip dependencies are needed.  If a tool is missing or fails,
the corresponding metrics are silently skipped.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable

import psutil  # type: ignore[import-untyped]

from goodseed.monitoring.daemon import MonitoringDaemon

_NVIDIA_SMI: str | None = shutil.which("nvidia-smi")

_NVIDIA_QUERY = ",".join([
    "utilization.gpu",     # %
    "memory.used",         # MiB
    "memory.total",        # MiB
    "power.draw",          # W
])


def _collect_nvidia_metrics() -> list[dict[str, float]]:
    """Return per-GPU metrics from nvidia-smi, or [] on failure."""
    if _NVIDIA_SMI is None:
        return []
    try:
        out = subprocess.run(
            [_NVIDIA_SMI, "--query-gpu=" + _NVIDIA_QUERY,
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return []
    except Exception:
        return []

    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpu_util = float(parts[0])
            mem_used = float(parts[1])
            mem_total = float(parts[2])
            power = float(parts[3])
        except (ValueError, IndexError):
            continue
        m: dict[str, float] = {"gpu": gpu_util}
        if mem_total > 0:
            m["gpu_memory"] = round(mem_used / mem_total * 100, 1)
        m["gpu_power"] = round(power, 1)
        gpus.append(m)
    return gpus


_ROCM_SMI: str | None = shutil.which("rocm-smi")


def _collect_amd_metrics() -> list[dict[str, float]]:
    """Return per-GPU metrics from rocm-smi, or [] on failure."""
    if _ROCM_SMI is None:
        return []
    try:
        out = subprocess.run(
            [_ROCM_SMI, "--showuse", "--showmemuse", "--showpower", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return []
        data = json.loads(out.stdout)
    except Exception:
        return []

    gpus = []
    for _card_id, card in sorted(data.items()):
        if not isinstance(card, dict):
            continue
        m: dict[str, float] = {}
        # GPU utilisation %
        for key in ("GPU use (%)", "GPU Usage (%)", "GPU use"):
            if key in card:
                try:
                    m["gpu"] = float(str(card[key]).rstrip("%"))
                except (ValueError, TypeError):
                    pass
                break
        # Memory utilisation %
        for key in ("GPU Memory Allocated (VRAM%)", "GPU memory use (%)",
                     "VRAM Usage (%)", "GPU memory use"):
            if key in card:
                try:
                    m["gpu_memory"] = float(str(card[key]).rstrip("%"))
                except (ValueError, TypeError):
                    pass
                break
        # Power draw (watts)
        for key in ("Current Socket Graphics Package Power (W)",
                     "Average Graphics Package Power (W)"):
            if key in card:
                try:
                    m["gpu_power"] = round(float(card[key]), 1)
                except (ValueError, TypeError):
                    pass
                break
        if m:
            gpus.append(m)
    return gpus



def _flatten_gpu_metrics(gpus: list[dict[str, float]]) -> dict[str, float]:
    """Flatten per-GPU metric dicts, adding _N suffixes for multi-GPU."""
    if not gpus:
        return {}
    if len(gpus) == 1:
        return gpus[0]
    flat: dict[str, float] = {}
    for i, gpu in enumerate(gpus):
        for k, v in gpu.items():
            flat[f"{k}_{i}"] = v
    return flat


class HardwareMonitorDaemon(MonitoringDaemon):
    """Periodically collects hardware metrics and logs them."""

    def __init__(
        self,
        *,
        namespace: str,
        log_fn: Callable[[dict[str, float], int], None],
        interval: float = 10.0,
    ) -> None:
        super().__init__(interval=interval, name="goodseed-hardware-monitor")
        self._namespace = namespace
        self._log_fn = log_fn
        self._step = 0

        # Prime the first cpu_percent call (returns 0.0 on first call)
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    def work(self) -> None:
        metrics: dict[str, float] = {}

        # CPU %
        try:
            metrics["cpu"] = psutil.cpu_percent(interval=None)
        except Exception:
            pass

        # Memory %
        try:
            metrics["memory"] = psutil.virtual_memory().percent
        except Exception:
            pass

        # GPU metrics (try NVIDIA first, then AMD)
        gpus = _collect_nvidia_metrics()
        if not gpus:
            gpus = _collect_amd_metrics()
        metrics.update(_flatten_gpu_metrics(gpus))

        if metrics:
            prefixed = {f"{self._namespace}/{k}": v for k, v in metrics.items()}
            self._log_fn(prefixed, self._step)
            self._step += 1

    def close(self) -> None:
        self.stop()
