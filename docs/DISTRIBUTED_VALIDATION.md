# Distributed inference validation

Validation date: 2026-11-14. Development environment: macOS arm64, Python 3.9, PyTorch 2.8.0 with CUDA unavailable.

## Executed locally

- Ten Python tests covering 1/2/4/8-GPU command generation, dataset validation, throughput denominators, nanosecond-to-millisecond conversion, one-token ITL exclusion, malformed/missing measurements, NCCL table parsing and correctness failures, mixed completed/failed/skipped execution, overwrite protection, subprocess failure and timeout termination.
- Python syntax compilation and CLI help/plan smoke checks.
- A real CPU-host invocation of the scaling runner: all four GPU counts were skipped with `visible usable GPUs: 0`; exit status was 2. The resulting report contained no timing or throughput values.
- The original compiler's C++ tests and generated-Python arithmetic tests remain part of CTest, alongside the new Python suite.

Parser fixtures and mocked subprocess results in tests are synthetic contract data, not measurements. No synthetic performance fixture is published as a benchmark result.

## Not executed

TensorRT-LLM model loading/generation, CUDA stream scheduling, GPU collective correctness, nccl-tests measurements, real PCIe/NVLink inventory, and 1/2/4/8-GPU scaling require an NVIDIA GPU host. They remain unvalidated at runtime here. No GPU speedup, latency, bandwidth, or tokens/sec claim is made.

## API contract reference

The implementation was checked against NVIDIA's current CLI documentation and the reporting/request-record source at TensorRT-LLM commit `47375913f155bcb50fb8c8da57f9c8b9520c38f3`. This identifies the reviewed interface, not a tested binary dependency or a guarantee that every TensorRT-LLM release is compatible.

Before citing résumé performance claims, execute the documented GPU commands, retain the source/container versions and raw artifacts, inspect correctness results, and report the actual hardware and workload. In particular, “measured NCCL latency” and “inspected NVLink topology” are not yet supported by measurements from this development session.
