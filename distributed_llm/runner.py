"""Launch complete experiments; retain provenance, failures and unavailable sizes."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import json
import shutil
import subprocess

from .config import validate_dataset
from .metrics import inference_report, nccl_rows
from .runtime import execute, gpu_count, inventory, write_json

COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "alltoall")


def run_inference(config, output):
    validate_dataset(config.dataset)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "kind": "trtllm",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "dataset_sha256": hashlib.sha256(Path(config.dataset).read_bytes()).hexdigest(),
        "inventory": inventory(),
        "runs": [],
    }
    available = gpu_count()
    for tp in config.world_sizes:
        directory = output / f"tp{tp}"
        directory.mkdir()
        row = {
            "gpus": tp,
            "status": "running",
            "command": config.command(tp, directory),
        }
        manifest["runs"].append(row)
        write_json(output / "manifest.json", manifest)
        if tp > available:
            row.update(
                status="skipped",
                reason=f"requires {tp} CUDA GPUs; visible usable GPUs: {available}",
            )
        else:
            try:
                if not shutil.which("trtllm-bench"):
                    raise RuntimeError("trtllm-bench is not installed")
                execute(
                    row["command"], directory / "stdout.log", config.timeout_seconds
                )
                row["metrics"] = inference_report(
                    json.loads((directory / "raw-report.json").read_text()),
                    json.loads((directory / "requests.json").read_text()),
                    tp,
                )
                row["status"] = "completed"
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                RuntimeError,
                subprocess.TimeoutExpired,
            ) as exc:
                row.update(status="failed", reason=str(exc))
        write_json(output / "manifest.json", manifest)
    render_report(manifest, output / "REPORT.md")
    return manifest


def run_nccl(binary_dir, output, sizes=(1, 2, 4, 8), timeout=600):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "kind": "nccl-tests",
        "inventory": inventory(),
        "runs": [],
    }
    available = gpu_count()
    for gpus in sizes:
        for collective in COLLECTIVES:
            log = output / f"{collective}-{gpus}.log"
            command = [
                str(Path(binary_dir).resolve() / (collective + "_perf")),
                "-b",
                "8",
                "-e",
                "128M",
                "-f",
                "2",
                "-g",
                str(gpus),
                "-w",
                "5",
                "-n",
                "20",
                "-c",
                "1",
                "-d",
                "float",
            ]
            binary = Path(command[0])
            digest = (
                hashlib.sha256(binary.read_bytes()).hexdigest()
                if binary.is_file()
                else None
            )
            row = {
                "gpus": gpus,
                "collective": collective,
                "command": command,
                "binary_sha256": digest,
                "status": "running",
            }
            manifest["runs"].append(row)
            write_json(output / "manifest.json", manifest)
            if gpus > available:
                row.update(
                    status="skipped", reason=f"only {available} usable CUDA GPUs"
                )
            else:
                try:
                    execute(command, log, timeout)
                    row["measurements"] = nccl_rows(log.read_text(), collective, gpus)
                    row["status"] = "completed"
                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                    subprocess.TimeoutExpired,
                ) as exc:
                    row.update(status="failed", reason=str(exc))
            write_json(output / "manifest.json", manifest)
    return manifest


def render_report(manifest, path):
    rows = manifest["runs"]
    baseline = next(
        (
            r["metrics"]["output_tokens_per_second"]
            for r in rows
            if r["gpus"] == 1 and r["status"] == "completed"
        ),
        None,
    )
    lines = [
        "# Tensor-parallel inference results",
        "",
        "Only completed GPU runs contain measurements. ITL is a per-request average, not individual token gaps.",
        "",
        "| GPUs | Status | Output tok/s | tok/s/GPU | TTFT P50 ms | Average ITL P50 ms | Speedup | Efficiency |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value):
        return "—" if value is None else f"{value:.3f}"

    for row in rows:
        if row["status"] != "completed":
            lines.append(f"| {row['gpus']} | {row['status']} | — | — | — | — | — | — |")
            continue
        m = row["metrics"]
        speed = m["output_tokens_per_second"] / baseline if baseline else None
        values = [
            m["output_tokens_per_second"],
            m["output_tokens_per_second_per_gpu"],
            m["ttft"]["p50_ms"],
            m["request_average_itl"]["p50_ms"],
            speed,
            speed / row["gpus"] if speed is not None else None,
        ]
        lines.append(
            f"| {row['gpus']} | completed | " + " | ".join(map(fmt, values)) + " |"
        )
    lines.extend(
        [
            "",
            "Full commands, dataset hash, versions, hardware inventory, and failure/skip reasons are in manifest.json.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n")
