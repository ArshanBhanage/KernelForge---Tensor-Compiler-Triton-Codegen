"""CPU contract/integration tests; fixtures are synthetic, never GPU measurements."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from distributed_llm.config import Config, validate_dataset
from distributed_llm.metrics import inference_report, nccl_rows
from distributed_llm.runner import run_inference, render_report
from distributed_llm.runtime import execute


def fixture():
    return (
        {
            "world_info": {"world_size": 2, "tp_size": 2, "pp_size": 1},
            "request_info": {"num_requests": 2},
            "performance": {"system_output_throughput_tok_s": 400},
        },
        [
            {
                "id": 0,
                "start_timestamp": 1000000,
                "first_token_timestamp": 11000000,
                "end_timestamp": 31000000,
                "output_tokens": 3,
                "error_tokens": 0,
            },
            {
                "id": 1,
                "start_timestamp": 1000000,
                "first_token_timestamp": 21000000,
                "end_timestamp": 21000000,
                "output_tokens": 1,
                "error_tokens": 0,
            },
        ],
    )


class MetricsTests(unittest.TestCase):
    def test_units_and_per_gpu_denominator(self):
        raw, requests = fixture()
        metrics = inference_report(raw, requests, 2)
        self.assertEqual(metrics["output_tokens_per_second_per_gpu"], 200)
        self.assertEqual(metrics["ttft"]["p50_ms"], 15)
        self.assertEqual(metrics["request_average_itl"]["mean_ms"], 10)
        self.assertEqual(metrics["request_average_itl"]["samples"], 1)

    def test_bad_reports_rejected(self):
        for change in ("mapping", "count", "timestamps", "errors", "nan", "missing"):
            raw, requests = fixture()
            if change == "mapping":
                raw["world_info"]["world_size"] = 4
            if change == "count":
                raw["request_info"]["num_requests"] = 3
            if change == "timestamps":
                requests[0]["first_token_timestamp"] = -1
            if change == "errors":
                requests[0]["error_tokens"] = 1
            if change == "nan":
                raw["performance"]["system_output_throughput_tok_s"] = float("nan")
            if change == "missing":
                del raw["performance"]
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                inference_report(raw, requests, 2)

    def test_nccl_both_modes_and_scientific_notation(self):
        rows = nccl_rows(
            "# header\n1024 256 float sum -1 1.25 8.1e-1 1.2 0 2.0 0.5 0.75 0\n",
            "all_reduce",
            4,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["latency_us"], 1.25)
        self.assertEqual(rows[0]["algbw_gbps"], 0.81)
        self.assertEqual(rows[1]["mode"], "in_place")

    def test_nccl_failure_or_unknown_format(self):
        for text in ("", "1024 256 float", "1024 256 float sum -1 1 1 1 1 1 1 1 0"):
            with self.assertRaises(ValueError):
                nccl_rows(text, "all_reduce", 2)


class ConfigTests(unittest.TestCase):
    def test_scaling_plan(self):
        config = Config("model name", "data name.jsonl")
        for size in (1, 2, 4, 8):
            command = config.command(size, Path("out") / str(size))
            self.assertEqual(command[command.index("--tp") + 1], str(size))
            self.assertIn("--streaming", command)
            self.assertEqual(command[2], "model name")  # shell-free argv

    def test_invalid_config(self):
        for sizes in ((0,), (3,), (1, 1), (), (True,)):
            with self.assertRaises(ValueError):
                Config("m", "d", sizes)
        with self.assertRaises(ValueError):
            Config("m", "d", concurrency=0)

    def test_dataset_validation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            row = {"task_id": 1, "prompt": "test", "output_tokens": 2}
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(validate_dataset(path), 1)
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaises(ValueError):
                validate_dataset(path)


class RunnerTests(unittest.TestCase):
    def test_completed_skipped_and_failed_runs(self):
        with tempfile.TemporaryDirectory() as d:
            dataset = Path(d) / "data.jsonl"
            dataset.write_text('{"task_id":0,"prompt":"x","output_tokens":3}\n')

            def fake_run(command, log, timeout):
                tp = int(command[command.index("--tp") + 1])
                if tp == 1:
                    raise RuntimeError("synthetic execution failure")
                raw, requests = fixture()
                Path(command[command.index("--report_json") + 1]).write_text(
                    json.dumps(raw)
                )
                Path(command[command.index("--request_json") + 1]).write_text(
                    json.dumps(requests)
                )

            with patch("distributed_llm.runner.gpu_count", return_value=2), patch(
                "distributed_llm.runner.inventory", return_value={}
            ), patch("distributed_llm.runner.shutil.which", return_value="fake"), patch(
                "distributed_llm.runner.execute", side_effect=fake_run
            ):
                manifest = run_inference(Config("m", str(dataset)), Path(d) / "run")
            self.assertEqual(
                [r["status"] for r in manifest["runs"]],
                ["failed", "completed", "skipped", "skipped"],
            )
            self.assertNotIn("metrics", manifest["runs"][2])
            self.assertTrue((Path(d) / "run/REPORT.md").exists())
            with self.assertRaises(FileExistsError):
                run_inference(Config("m", str(dataset)), Path(d) / "run")

    def test_all_skipped_have_no_performance_claims(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "report.md"
            render_report({"runs": [{"gpus": 8, "status": "skipped"}]}, out)
            self.assertIn("| 8 | skipped | —", out.read_text())

    def test_process_failure_and_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log"
            with self.assertRaises(RuntimeError):
                execute([sys.executable, "-c", "raise SystemExit(3)"], log, 5)
            with self.assertRaises(subprocess.TimeoutExpired):
                execute(
                    [sys.executable, "-c", "import time; time.sleep(10)"], log, 0.05
                )


if __name__ == "__main__":
    unittest.main()
