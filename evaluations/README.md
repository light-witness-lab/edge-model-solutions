# Evaluations

Evaluation JSON files are evidence records, not capability claims. They must identify the solution, model revision, device, runtime/backend, measured latency, output validity, and failure behavior. Use `loaded-only` when a model runs but product quality or real-time behavior is not established.

## Index

- [`app-ranker-observations-2026-10-18.json`](app-ranker-observations-2026-10-18.json) — emulator observations after d1-omni-600M became the street-snap app's default ranker (`loaded-only`): rule layer ≈2.4 analyses/s at a 250 ms frame clock with ≈180 ms fixed per-tick overhead, serial single-pass scheduling with 3–15 s cooldown and scene-result reuse, d1 gate never stably accepted across 70+ positions (portrait often over the 30 s budget; landscape 16–22 s, mostly UNSTABLE/abstain), exactly one non-reproduced SmolVLM 256M acceptance (0.9947/0.9964, margins 0.1354/0.2191, ~10.4 s). Honor Vs3 and aesthetic quality remain not-verified.
- [`d1-prompt-v1-eval-2026-10-18.json`](d1-prompt-v1-eval-2026-10-18.json) — pre-registered isolated probe of a criteria-rich v1 prompt for the d1-omni-600M composition-choice task (20 deterministic contact sheets, desktop ORT 1.22, real pinned graphs). **Verdict: REJECT** — v1 moved the model's positional preference from panel C to panel A without improving content following (9/18 → 6/18 argmax correct; 6/18 → 3/18 gate-selected correct); the app stays on `comp-choice-v0`, evidence level unchanged at `loaded-only`. Harness: [`scripts/d1-prompt-eval`](../scripts/d1-prompt-eval/).
