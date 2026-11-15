# Distributed LLM Inference — Multi-GPU Communications

TensorRT-LLM inference and benchmark orchestration for **1/2/4/8-GPU tensor parallelism**, with NCCL collective correctness checks, CUDA stream overlap experiments, topology capture, and reproducible reports. The original [KernelForge tensor compiler](docs/KERNELFORGE.md) remains available in `include/`, `src/`, and `python/`.

**Validation status:** CPU tests pass. This development host has no CUDA GPU, so TensorRT-LLM generation, NCCL measurements and GPU scaling have **not** been executed here. The repository contains runnable experiments, not claimed GPU performance results. See [validation](docs/DISTRIBUTED_VALIDATION.md).

## What is implemented

| Component | Behavior |
|---|---|
| `distributed_llm.infer` | Loads a model through TensorRT-LLM, sets TP/PP, generates text, and shuts down its workers |
| `distributed_llm bench` | Sweeps TP 1/2/4/8 with streaming `trtllm-bench`; stores logs, raw reports, request records and provenance |
| `distributed_llm.collectives` | One process per GPU; verifies all-reduce, all-gather, reduce-scatter and all-to-all; compares serial and overlapping compute/communication |
| `distributed_llm nccl` | Runs the four upstream `nccl-tests` binaries and parses latency, algorithm bandwidth and bus bandwidth |
| `distributed_llm topology` | Records GPU identities, PCI bus IDs, driver versions, the topology matrix, and NVLink status |
| Reporting | Output tokens/sec/GPU, TTFT, average ITL, 1-GPU-relative speedup and scaling efficiency |

TensorRT-LLM owns model partitioning, rank spawning, runtime collectives and model-specific kernels. The explicit NCCL stream experiments are a separate research harness using PyTorch's NCCL backend; they do not replace or modify TensorRT-LLM's internal scheduler. Current TensorRT-LLM uses its PyTorch backend, selected explicitly by this runner.

## CPU-only quick start

Python 3.9+ is sufficient for planning, parsing and contract tests. No model is downloaded by these commands.

```sh
python3 -m unittest discover -s tests -p 'test_distributed.py' -v
python3 -m distributed_llm plan --config configs/scaling.json
```

The existing C++ compiler and all CPU tests can be checked together:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Real GPU setup

Use a Linux NVIDIA GPU environment with a TensorRT-LLM-supported driver/GPU, a compatible CUDA-enabled PyTorch build, and `trtllm-bench` on PATH. Follow NVIDIA's [installation guide](https://nvidia.github.io/TensorRT-LLM/installation/linux.html) or use its supported container. TensorRT-LLM and CUDA dependency versions must match your hardware; this project does not install or upgrade your GPU stack. Record the exact container digest for comparisons.

The CLI/report adapter follows NVIDIA's current [benchmark interface](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-bench.html) and [upstream reporting schema](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/bench/dataclasses/reporting.py). Check `trtllm-bench throughput --help` in your chosen environment. A different schema fails explicitly and keeps the raw artifacts. GPU runtime compatibility is not yet certified against a specific release.

```sh
# Run ONCE. TensorRT-LLM starts and coordinates its own TP worker ranks.
CUDA_VISIBLE_DEVICES=0,1 python3 -m distributed_llm.infer \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp 2 \
  --prompt 'Explain tensor parallelism.' --output generation.json

python3 -m distributed_llm topology --output topology.json
python3 -m distributed_llm bench --config configs/scaling.json --output results/scaling-01
```

Edit `configs/scaling.json` for the model, JSONL dataset, warmup, concurrency and GPU counts. Model weights must fit the selected devices, and the architecture must support the requested TP size. Model access/download requirements remain those of the model publisher. `CUDA_VISIBLE_DEVICES` restricts the usable devices; the runner never requests more GPUs than PyTorch can see. A fitting model is not guaranteed by device count alone.

Each output directory must be new, protecting existing measurements. Unavailable sizes are `skipped`; nonzero exits, timeouts, missing reports or invalid metrics are `failed`. Exit code 0 means at least one completed run and no failures, 1 means a failure, and 2 means all runs were skipped. A skipped run has no invented timing values.

The included eight prompts are a smoke workload. For useful throughput comparisons, prepare a larger fixed-length dataset with the same model tokenizer, keep concurrency and input/output lengths constant, and repeat runs:

```sh
trtllm-bench --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 prepare-dataset \
  --output workload.jsonl --random-seed 42 token-norm-dist \
  --input-mean 128 --input-stdev 0 --output-mean 128 --output-stdev 0 --num-requests 1000
# Point dataset in configs/scaling.json to workload.jsonl before benchmarking.
```

## NCCL collective measurements

Build [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests) with the CUDA/NCCL libraries for your environment. Record the source commit; the runner records binary hashes. This harness is single-node and uses one process controlling `-g N` GPUs; MPI is not required for these tests.

```sh
# In your nccl-tests checkout:
make -j CUDA_HOME=/usr/local/cuda NCCL_HOME=/path/to/nccl
# Back in this repository:
python3 -m distributed_llm nccl --binary-dir /path/to/nccl-tests/build --output results/nccl-01
```

The runner exercises `all_reduce_perf`, `all_gather_perf`, `reduce_scatter_perf`, and `alltoall_perf`, with 5 warmups, 20 timed iterations, FP32, correctness checking, and an 8-byte to 128-MiB geometric size sweep. Parsed out-of-place and supported in-place results retain `latency_us`, `algbw_gbps` and `busbw_gbps` (GB/s, decimal). Bandwidth comes directly from nccl-tests; bus bandwidth is a collective-specific normalized metric, not a measured PCIe/NVLink link rate. See [NVIDIA's definitions](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md).

## Stream scheduling and tensor partitions

```sh
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  -m distributed_llm.collectives --elements 1048576 --matrix-size 1024 \
  --warmup 5 --iterations 30 --output results/overlap-2gpu
```

Every rank first computes its input-dimension shard of a matrix multiplication. An all-reduce reconstructs the full result and compares it with an unsharded reference. Full reference matrices are intentionally retained for this correctness experiment; this is not a memory-saving model loader.

The harness then validates each collective numerically, including source/destination routing for all-to-all. It benchmarks an independent FP32 GEMM alongside communication:

- **Serial:** the communication stream waits for the GEMM completion event.
- **Overlap:** compute and communication may progress independently after a shared start event.
- Both paths join with CUDA events before reading elapsed time. NCCL `Work.wait()` connects its internal stream to the caller stream. Per-iteration measurements use the slowest rank, excluding warmups.

The combined timings include dispatch gaps and, for all-reduce, a destination reset copy. They are not pure collective latency or evidence that TensorRT-LLM inference overlaps by the same amount. Shared SM/bandwidth pressure can make overlap slower. Use Nsight Systems around the `torchrun` command to inspect the actual schedule.

## Results and methodology

A scaling directory contains `manifest.json`, `REPORT.md`, and `tpN/{stdout.log,raw-report.json,requests.json}` for executed sizes. Reports include:

- Output throughput divided by TP × PP ranks (PP is fixed at 1).
- TTFT from request start to first output token.
- Per-request average ITL: `(last token time − first token time) / (output tokens − 1)`. One-token outputs are excluded. P50/P95 aggregate these request averages, not individual token intervals.
- Speedup relative to a completed 1-GPU run and efficiency `speedup / GPU count`. Both remain unavailable without a 1-GPU baseline.

Run directories retain the dataset SHA-256, configuration, commands, software versions, raw topology commands and skip/failure reasons. TTFT/ITL use upstream nanosecond request timestamps converted to milliseconds. No hardware measurements are included in this checkout. See [architecture and limits](docs/DISTRIBUTED_ARCHITECTURE.md).
