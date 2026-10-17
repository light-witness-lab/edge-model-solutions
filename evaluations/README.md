# Evaluations

Evaluation JSON files are evidence records, not capability claims. They must identify the solution, model revision, device, runtime/backend, measured latency, output validity, and failure behavior. Use `loaded-only` when a model runs but product quality or real-time behavior is not established.

## Index

- [`d1-prompt-v1-eval-2026-10-18.json`](d1-prompt-v1-eval-2026-10-18.json) — pre-registered isolated probe of a criteria-rich v1 prompt for the d1-omni-600M composition-choice task (20 deterministic contact sheets, desktop ORT 1.22, real pinned graphs). **Verdict: REJECT** — v1 moved the model's positional preference from panel C to panel A without improving content following (9/18 → 6/18 argmax correct; 6/18 → 3/18 gate-selected correct); the app stays on `comp-choice-v0`, evidence level unchanged at `loaded-only`. Harness: [`scripts/d1-prompt-eval`](../scripts/d1-prompt-eval/).
