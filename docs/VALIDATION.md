# Validation

Measured on 2026-11-12, macOS arm64, AppleClang 17, CMake Release build.

Core build and CTest passed locally. Test checks remain active in Release builds.

Compiler benchmark (one local run; timings vary by host/load):

```json
{"iterations":10000,"p50_us":1.083,"p95_us":1.125,"unfused_materialization_bytes":12288,"fused_materialization_bytes":4096,"source_bytes_checksum":3850000}
```

C++ semantic/error-path tests and generated Python arithmetic/masked-tail checks passed. Real Triton GPU execution remains untested on this host.
