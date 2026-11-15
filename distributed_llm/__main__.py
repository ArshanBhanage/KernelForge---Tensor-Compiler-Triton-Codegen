"""CPU-safe command planning, GPU experiment execution and report generation."""

import argparse
import json
from pathlib import Path
from .config import Config
from .runner import run_inference, run_nccl, render_report
from .runtime import inventory, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("plan", "bench"):
        s = sub.add_parser(name)
        s.add_argument("--config", default="configs/scaling.json")
        s.add_argument("--output", default="results/inference")
    s = sub.add_parser("topology")
    s.add_argument("--output", default="topology.json")
    s = sub.add_parser("nccl")
    s.add_argument("--binary-dir", required=True)
    s.add_argument("--output", default="results/nccl")
    s = sub.add_parser("report")
    s.add_argument("manifest")
    s.add_argument("--output", default="REPORT.md")
    args = p.parse_args()
    try:
        if args.action == "plan":
            config = Config.load(args.config)
            print(
                json.dumps(
                    {
                        "execution": "plan_only",
                        "commands": [
                            config.command(n, Path(args.output).resolve() / f"tp{n}")
                            for n in config.world_sizes
                        ],
                    },
                    indent=2,
                )
            )
        elif args.action == "topology":
            write_json(args.output, inventory())
        elif args.action == "report":
            render_report(json.loads(Path(args.manifest).read_text()), args.output)
        else:
            result = (
                run_inference(Config.load(args.config), args.output)
                if args.action == "bench"
                else run_nccl(args.binary_dir, args.output)
            )
            print(
                json.dumps(
                    [
                        {k: r[k] for k in ("gpus", "status", "reason") if k in r}
                        for r in result["runs"]
                    ],
                    indent=2,
                )
            )
            if any(r["status"] == "failed" for r in result["runs"]):
                return 1
            if not any(r["status"] == "completed" for r in result["runs"]):
                return 2
        return 0
    except (ValueError, TypeError, OSError, KeyError) as exc:
        p.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
