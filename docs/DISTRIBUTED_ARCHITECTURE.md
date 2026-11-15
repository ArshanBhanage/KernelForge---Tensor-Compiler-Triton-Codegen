# Architecture and experiment boundaries

```text
configuration + workload
          |
          v
single-node experiment runner -----> GPU/PCIe/NVLink inventory
          |
          +--> trtllm-bench --> TensorRT-LLM workers (TP ranks)
          |                         |
          |                         +--> model partitions + runtime collectives
          |                         +--> raw report + streaming request records
          |
          +--> nccl-tests binaries --> latency / algbw / busbw
          |
          +--> status manifest --> normalized JSON + Markdown scaling report

separate torchrun experiment
   rank-local matrix shard --> NCCL all-reduce --> full-reference validation
   independent GEMM stream + collective stream --> join --> slowest-rank timing
```

## Ownership and rank mapping

`infer.py` is launched once; `tensor_parallel_size=N` and `pipeline_parallel_size=1` delegate partitioning and worker execution to the TensorRT-LLM LLM API. `trtllm-bench` has the same ownership. Do not wrap either in torchrun/mpirun or you may start multiple independent model instances.

`collectives.py` instead uses torchrun's rank environment. Each local rank selects its own visible GPU before initializing NCCL. All ranks execute collectives in identical order; operation inputs remain alive until both CUDA streams finish. There is one process group with a 180-second watchdog. On worker failure, torchrun is responsible for terminating peer processes.

## Reproducibility

Keep model revision, precision, workload, concurrency, warmup, clock/power settings, GPU model, interconnect and runtime versions fixed when comparing sizes. The tool records versions but does not change GPU clocks or power settings. GPU topology is read-only; NVLink command failures are retained rather than interpreted as proof of no NVLink.

Topology output can describe all physical GPUs even when CUDA_VISIBLE_DEVICES restricts the experiment. The manifest records that mask. GPU runtime count comes from PyTorch, so container/MIG visibility is honored; capacity and topology suitability remain the operator's responsibility.

The nccl-tests adapter expects the conventional 13-column table and requests float explicitly. Unsupported table layouts fail parsing, preserving the raw log. Missing in-place values marked N/A or '-' are omitted. A failed correctness counter rejects the run.

## Scope

Single-node TP only. No multi-node launch, expert/pipeline scaling, custom TensorRT plugins, engine conversion, model-specific sharding reimplementation, production serving API, or automatic GPU provisioning. The explicit collective/stream harness is independent of TensorRT-LLM; it does not claim to instrument its internals. TensorRT-LLM runtime use, NCCL performance and GPU timing still require real hardware validation.

Sources: [TensorRT-LLM distributed generation](https://nvidia.github.io/TensorRT-LLM/examples/llm_inference_distributed.html), [PyTorch distributed CUDA semantics](https://docs.pytorch.org/docs/stable/distributed.html), [NCCL tests](https://github.com/NVIDIA/nccl-tests).
