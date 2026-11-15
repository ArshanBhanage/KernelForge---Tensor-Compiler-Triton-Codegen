"""Validated single-node tensor-parallel experiment configuration."""

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class Config:
    model: str
    dataset: str
    world_sizes: tuple = (1, 2, 4, 8)
    warmup: int = 10
    concurrency: int = 16
    timeout_seconds: int = 3600

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError(
                "model must be a nonempty Hugging Face ID or checkpoint path"
            )
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("dataset is required")
        if not self.world_sizes or len(set(self.world_sizes)) != len(self.world_sizes):
            raise ValueError("world sizes must be nonempty and unique")
        if any(type(n) is not int or n not in (1, 2, 4, 8) for n in self.world_sizes):
            raise ValueError("supported GPU counts: 1, 2, 4, 8")
        for name in ("concurrency", "timeout_seconds"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        if type(self.warmup) is not int or self.warmup < 0:
            raise ValueError("warmup must be a nonnegative integer")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if "world_sizes" in data:
            data["world_sizes"] = tuple(data["world_sizes"])
        return cls(**data)

    def command(self, tp, directory):
        if tp not in self.world_sizes:
            raise ValueError("TP size is not in the experiment")
        directory = Path(directory)
        return [
            "trtllm-bench",
            "--model",
            self.model,
            "throughput",
            "--backend",
            "pytorch",
            "--dataset",
            str(Path(self.dataset).resolve()),
            "--tp",
            str(tp),
            "--pp",
            "1",
            "--streaming",
            "--concurrency",
            str(self.concurrency),
            "--warmup",
            str(self.warmup),
            "--report_json",
            str(directory / "raw-report.json"),
            "--request_json",
            str(directory / "requests.json"),
        ]


def validate_dataset(path):
    seen = set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = row.get("task_id")
        if type(task) is not int or task in seen:
            raise ValueError("dataset task_id must be a unique integer")
        seen.add(task)
        if type(row.get("output_tokens")) is not int or row["output_tokens"] <= 0:
            raise ValueError("output_tokens must be positive")
        if ("prompt" in row) == ("input_ids" in row):
            raise ValueError("exactly one of prompt/input_ids is required")
        if "prompt" in row and (
            not isinstance(row["prompt"], str) or not row["prompt"]
        ):
            raise ValueError("prompt must be nonempty text")
        if "input_ids" in row:
            ids = row["input_ids"]
            if (
                not isinstance(ids, list)
                or not ids
                or any(type(x) is not int or x < 0 for x in ids)
            ):
                raise ValueError("input_ids must be nonempty nonnegative token IDs")
    if not seen:
        raise ValueError("dataset is empty")
    return len(seen)
