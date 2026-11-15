# KernelForge — Tensor IR and Triton Code Generation

A small C++20 compiler library for contiguous FP32 elementwise tensor graphs. It validates a typed IR, resolves dependencies, eliminates dead nodes, and fuses the live graph into one Triton kernel. Python tools check emitted code and optionally execute it on a GPU.

## Build and run

Requires CMake 3.20+, C++20, and Python 3. Core builds do not require Triton.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

```sh
./build/kforge > generated.py
./build/kforge --benchmark
```

The CLI compiles `relu((a + b) * b)` plus an intentionally dead multiply. For custom graphs, use the public C++ `Graph`, `Node`, `Type`, `compile`, and `evaluate` APIs in `include/compiler.hpp`; the CLI is an example frontend, not a Python parser.

## Compiler pipeline

1. Validate nonempty positive shapes, FP32 dtype, operation arity, operand references, and exact shape/type agreement. All nodes, including dead nodes, must be valid.
2. Traverse dependencies with cycle detection. Forward references are supported.
3. Keep only nodes reachable from the single output (dead-code elimination).
4. Emit loads, SSA-like elementwise temporaries, and one masked output store in a single `@triton.jit` kernel. Shared subexpressions are emitted once.

Supported operations: input, add, multiply, ReLU. Inputs are ordered by dependency traversal and named `p<node_id>`. The CPU evaluator's inputs are indexed by node ID. Generated kernels require contiguous FP32 inputs of the validated shape, `N` equal to the shape product, and a valid power-of-two block size. The example Python launcher supplies these correctly.

No broadcasting, reductions, matrix multiplication, strides, autograd, autotuning, register-pressure heuristic, or arbitrary Python tracing. Fusion is suitable for small same-shape graphs; large graphs may spill registers. NaN behavior of the CPU reference is not specified to match GPU ReLU.

## Memory model and benchmark

For 1,024 elements and three live elementwise operations, unfused operation-output materialization is `3 × 1024 × 4 = 12,288` bytes. Fusion writes one final output (`4,096` bytes): **66.7% fewer modeled materialized bytes, counting the final output**. Temporary-only materialization drops from 8,192 to zero. This does not measure DRAM traffic, allocator usage, or register spills.

`--benchmark` warms up 100 times, then times 10,000 individual CPU calls to validation + dependency analysis + source generation using a monotonic clock. It reports P50/P95 microseconds and a source-size checksum. Graph construction, Triton JIT, GPU execution and process startup are excluded. See [measured results](VALIDATION.md). The résumé's 2.19 µs value is not assumed or hardcoded.

## Optional real GPU test

Install compatible PyTorch and Triton in your CUDA environment, then:

```sh
python3 python/run_gpu.py
```

This compares actual generated-kernel outputs with PyTorch for sizes 1, 127, 1024, and 1025, including masked tails. GPU execution was not available on the development Mac. The default test uses a minimal Python Triton test double to validate emitted arithmetic and masks; it does not prove Triton compilation or GPU correctness.

Code generation follows the [official Triton vector-add API](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html). GitHub Actions tests the C++ compiler and generated-source semantics on Linux and macOS.
