# Benchmarks

Every run should record model revision, quantization, runtime/backend, input size, device/OS, warm-up policy, p50/p95 latency, frame age, dropped frames, memory peak, temperature, battery change, output validity, and product-quality result.

A model loading successfully is a `loaded-only` result. A `verified` result requires the stated quality and performance threshold on the target device.

## Inference performance

[`inference-perf/`](inference-perf/) holds the reproducible protocol and JSON schema (`inference-perf/v1`) for the on-device inference research line: segmented end-to-end latency, W1/W2/W3 workloads, execution-provider/quantization/reuse/thermal sweeps, and the promotion rules from `not-verified` to `loaded-only` / `verified`. The runnable Kotlin/JVM skeleton lives in [`../scripts/inf-perf-bench`](../scripts/inf-perf-bench/); raw reports stay in gitignored `reports/local/`.
