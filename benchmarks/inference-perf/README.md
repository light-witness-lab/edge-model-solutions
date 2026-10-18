# Inference performance benchmark

A reproducible protocol for the on-device inference research line. It measures the **whole composition decision path**, not a single model's nominal latency:

```text
frame sample → resize/normalize → vision encode → prompt embed
            → decision forward (zero-token) or VLM prefill+decode
            → gate → result age
```

This is an engineering plan plus a runnable harness skeleton; it is **not** a set of performance claims. Every file produced here starts at `not-verified` until a named device and workload produce numbers. Emulator timings under software rendering stay labeled as such and never predict phone silicon.

## What we measure

Segment latency (p50/p95/min/max/mean, warm-up excluded):

| Segment | Meaning | Why it matters |
| --- | --- | --- |
| `vision_encode` | Vision tower over the contact sheet / frame | The dominant "prefill" cost for a multimodal choice; the prime candidate for prefix/feature reuse |
| `prompt_embed` | Tokenization + embedding of the fixed choice question | Mostly constant; should be cacheable |
| `decision_or_prefill` | d1 marker forward, or VLM prompt prefill | The reasoning cost per request |
| `decode_per_token` | VLM token decode (zero for d1) | Small here, but central to the future 120 Hz question |
| `end_to_end_round` | Full gated round including both permutation orders | The value the 30 s budget and cooldown actually see |

Plus, per run: `prefix_reuse_ms` (cold full-prefill vs shared vision/prefix), throughput (rounds/s, decode tok/s, preview fps), resources (memory peak, CPU%, dropped frames, p95 frame age), and output validity (gate returned/unstable/no-result/timeout counts, margin p50). Preview rate, model inference rate, and rendered-result age are always recorded separately.

## Workloads

- **W1 – decision head (d1, zero generated tokens).** The 20 deterministic 512×512 contact sheets already frozen in [`scripts/d1-prompt-eval`](../../scripts/d1-prompt-eval/) (P1/P2 bias probes + 18 discrimination sheets), plus a slot for the independent labelled photo set (the only accepted aesthetic evidence, still pending). Vision features are computed once per sheet and reused, mirroring the shipping gate.
- **W2 – VLM choice (SmolVLM 256M/500M).** The same sheets through vision→embed→decoder, both a single ordering and the guarded second ordering, so UNSTABLE rates and real wall-clock are captured together.
- **W3 – continuous scan (future).** A frame stream at a target cadence (today 1–3 rounds/s, north star 120 Hz analysis) that sweeps the camera; measures freshness under a moving scene signature (edge-triggered rounds), backpressure, and thermal falloff over minutes rather than a single cold run.

## Variable matrix

Hold the model and fixture fixed, then sweep one axis at a time:

- execution provider / backend: CPU (XNNPACK where available) → NNAPI → vendor QNN/GPU/NPU;
- thread count and affinity: 1 / 2 / 4 / big-cores / all, with explicit affinity where the OS allows;
- input resolution: analysis 288×384 vs 384×288 (orientation) and contact-sheet 384² / 512²;
- quantization: q4 MatMulNBits vs Int8 vs fp16 where the runtime supports each;
- reuse: vision/prefix recomputed every round vs shared across the permutation pair (and, later, across near-identical frames);
- thermal: cold first round vs sustained 3/5/10-minute run, recording start/end thermal state and clock throttling.

## Device tiers

| Tier | Role | Status |
| --- | --- | --- |
| Desktop JVM + ORT 1.22 | Deterministic functional/regression baseline; parity across harness changes | available (`scripts/d1-prompt-eval`) |
| Android 35 ARM64 AVD | Import, session creation, frame conversion, schema gate, degraded UI, budget behavior | `loaded-only`; software-render timing is unrepresentative |
| Honor Vs3 (and other real Android) | The only tier that answers latency, NPU, thermal and battery | `not-verified` until measured on device |

## Protocol rules

1. Pin model id, immutable revision, quantization and runtime version; record them in every JSON.
2. Warm up (default 3 iters), then measure (default 20 iters); never report the warm-up round as performance.
3. Single concurrency unless the run explicitly tests the prefill/decode split; fixed thread count per run.
4. Same bytes in → same result on deterministic CPU; a re-run should reproduce per-sheet probabilities.
5. Report p50 **and** p95, plus a clearly labeled single-run timing when only one run exists.
6. Record memory peak, battery delta and thermal state for any phone run; capture sustained (multi-minute) behavior separately from cold.
7. Gate outcomes travel with timing — a fast round that returns UNSTABLE is not a success.
8. Raw reports go to `reports/local/` (gitignored); only normalized, reproducible records are committed.

## Reproduce

The harness is a Kotlin/JVM Gradle project under [`scripts/inf-perf-bench`](../../scripts/inf-perf-bench/) using the same pinned ONNX Runtime 1.22 as the eval harness.

```sh
# Self-check: exercise the timing/percentile/JSON plumbing with a synthetic workload
cd scripts/inf-perf-bench
./gradlew run --args="--selfcheck /tmp/inf-perf-sample.json"

# Real run (requires a pinned model directory; segments with no bound graph stay null)
./gradlew run --args="--model ../../models/d1-omni-600m --workload W1 --threads 4 --out /tmp/w1.json"
```

The self-check output is marked `"workload": "synthetic-selfcheck"` and proves the measurement plumbing only; it is never a model result. The model-path run reuses the pinned graph loader and the deterministic fixtures; device-side (Android) runs are driven from the Street Snap app's existing instrumentation and normalized back into [`result-template.json`](result-template.json).

## How a result is promoted

A run is evidence about the named device/runtime only. It moves to `loaded-only` when the pinned graph completes end to end, and to `verified` only when the target device meets a pre-registered latency **and** validity threshold (fresh valid outputs, budget respected, preview/shutter unaffected, measured thermal cost) — with the independent labelled photo set, not synthetic sheets, supplying the quality verdict.
