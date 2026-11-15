"""Bounded subprocess execution and read-only hardware inventory."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def execute(argv, log, timeout):
    with Path(log).open("w") as output:
        process = subprocess.Popen(
            argv, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f"command exited {code}; see {log}")


def gpu_count():
    # Torch respects CUDA_VISIBLE_DEVICES (including UUIDs) and container visibility.
    try:
        import torch
    except ImportError:
        return 0
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def inventory():
    result = {
        "platform": platform.platform(),
        "python": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": {},
        "commands": {},
    }
    for package in ("torch", "tensorrt_llm"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    for name, argv in {
        "gpu": [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total",
            "--format=csv",
        ],
        "topology": ["nvidia-smi", "topo", "-m"],
        "nvlink": ["nvidia-smi", "nvlink", "--status"],
    }.items():
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=20)
            result["commands"][name] = {
                "argv": argv,
                "returncode": p.returncode,
                "stdout": p.stdout,
                "stderr": p.stderr,
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["commands"][name] = {"argv": argv, "error": str(exc)}
    return result
