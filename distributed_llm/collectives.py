"""torchrun NCCL correctness and CUDA stream overlap experiments (single node).

The communication stream hands work to ProcessGroupNCCL's internal stream.
Work.wait establishes the dependency back to the caller stream. CUDA events then
join both branches before timing the combined workload. No GPU numbers are
produced on CPU-only hosts.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--elements", type=int, default=1048576, help="FP32 elements per rank chunk"
    )
    p.add_argument("--matrix-size", type=int, default=1024)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--output", default="results/overlap")
    args = p.parse_args()
    if min(args.elements, args.matrix_size, args.iterations) <= 0 or args.warmup < 0:
        p.error("sizes/iterations must be positive and warmup nonnegative")
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available() or not dist.is_nccl_available():
        p.error("requires a CUDA-enabled PyTorch installation and NCCL")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (1, 2, 4, 8) or args.matrix_size % world:
        p.error("world size must be 1/2/4/8 and divide matrix-size")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    rank = dist.get_rank()
    try:
        directory = Path(args.output)
        directory.mkdir(parents=True, exist_ok=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(1234)
        compute_stream = torch.cuda.Stream()
        comm_stream = torch.cuda.Stream()
        # Stable allocations stay alive until every stream is synchronized.
        a = torch.randn(args.matrix_size, args.matrix_size, device="cuda")
        b = torch.randn_like(a)
        product = torch.empty_like(a)
        torch.cuda.synchronize()
        # Row-parallel linear layer: each rank computes a partition's contribution.
        width = args.matrix_size // world
        start = rank * width
        partial = (
            a[:, start : start + width].contiguous()
            @ b[start : start + width, :].contiguous()
        )
        dist.all_reduce(partial)
        torch.testing.assert_close(partial, a @ b, rtol=2e-4, atol=2e-3)
        results = []
        n = args.elements
        for op in ("all_reduce", "all_gather", "reduce_scatter", "all_to_all"):
            source = torch.full(
                (n if op in ("all_reduce", "all_gather") else n * world,),
                float(rank + 1),
                device="cuda",
            )
            if op == "all_to_all":
                # Encode BOTH source and destination rank to validate routing.
                source = (
                    rank * world
                    + torch.arange(world, device="cuda", dtype=torch.float32)
                ).repeat_interleave(n)
            destination = torch.empty(
                n * world if op in ("all_gather", "all_to_all") else n, device="cuda"
            )

            def launch():
                if op == "all_reduce":
                    destination.copy_(source)
                    return dist.all_reduce(destination, async_op=True)
                if op == "all_gather":
                    return dist.all_gather_into_tensor(
                        destination, source, async_op=True
                    )
                if op == "reduce_scatter":
                    return dist.reduce_scatter_tensor(
                        destination, source, async_op=True
                    )
                return dist.all_to_all_single(destination, source, async_op=True)

            torch.cuda.synchronize()
            with torch.cuda.stream(comm_stream):
                launch().wait()
            torch.cuda.synchronize()
            if op in ("all_reduce", "reduce_scatter"):
                expected = torch.full_like(destination, world * (world + 1) / 2)
            elif op == "all_to_all":
                expected = (
                    torch.arange(world, device="cuda", dtype=torch.float32) * world
                    + rank
                ).repeat_interleave(n)
            else:
                expected = torch.arange(
                    1, world + 1, device="cuda", dtype=torch.float32
                ).repeat_interleave(n)
            torch.testing.assert_close(destination, expected, rtol=0, atol=0)
            for mode in ("serial", "overlap"):
                samples = []
                for i in range(args.warmup + args.iterations):
                    dist.barrier()
                    torch.cuda.synchronize()
                    begin = torch.cuda.Event(enable_timing=True)
                    compute_done = torch.cuda.Event()
                    comm_done = torch.cuda.Event()
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    compute_stream.wait_event(begin)
                    comm_stream.wait_event(begin)
                    with torch.cuda.stream(compute_stream):
                        torch.mm(a, b, out=product)
                        compute_done.record()
                    with torch.cuda.stream(comm_stream):
                        if mode == "serial":
                            comm_stream.wait_event(compute_done)
                        work = launch()
                        work.wait()  # NCCL stream -> caller stream ordering
                        comm_done.record()
                    current = torch.cuda.current_stream()
                    current.wait_event(compute_done)
                    current.wait_event(comm_done)
                    end.record()
                    end.synchronize()
                    elapsed = torch.tensor(begin.elapsed_time(end), device="cuda")
                    # Critical-path latency is the slowest rank, not rank zero alone.
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                    if i >= args.warmup:
                        samples.append(elapsed.item())
                results.append(
                    {
                        "collective": op,
                        "mode": mode,
                        "max_rank_p50_ms": statistics.median(samples),
                        "max_rank_mean_ms": statistics.mean(samples),
                        "samples_ms": samples,
                    }
                )
        torch.cuda.synchronize()
        torch.testing.assert_close(product, a @ b)
        if rank == 0:
            metadata = {
                "world_size": world,
                "elements_per_chunk": n,
                "matrix_size": args.matrix_size,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(),
                "gpu": torch.cuda.get_device_name(),
                "partition_correctness": "passed",
                "collective_correctness": "passed",
                "results": results,
            }
            (directory / "overlap.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
            print(json.dumps(metadata, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
