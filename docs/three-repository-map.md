# Three-repository map

One product effort across three repositories under [light-witness-lab](https://github.com/light-witness-lab). Updated 2026-10-25 (pose T108–T110; last org merge PR #6).

## What lives where

| Repository | Owns | Does not own |
| --- | --- | --- |
| [street-snap](https://github.com/light-witness-lab/street-snap) (途帧, Android app) | Camera UI and lifecycle, frame sampling/backpressure, rule layer, model adapters and download/checksum flow, Compose overlays, APK, device evidence | Model weights, APKs/signing material, private photos (all stay local/elsewhere) |
| [edge-model-solutions](https://github.com/light-witness-lab/edge-model-solutions) (this repo) | Pinned model artifacts + provenance (manifest, SHA-256, per-weight license, LFS for >100 MiB), solution SOPs, `evaluations/` evidence JSON, reproducible benchmark/eval harnesses, failure records | App code releases, private photo sets |
| [inference-engineering](https://github.com/light-witness-lab/inference-engineering) | App-decoupled inference knowledge: CPU inference playbook, case studies, frozen pipeline baselines, runtime/quantization/scheduling research notes | App code, model weights, benchmark raw results (those link back here) |

## Current anchors

- street-snap org main: `49123ad` (2026-10-24, PR #6) — shipping v0.3.0-dev; d1-omni-600M is the only composition ranker (SmolVLM retired 2026-10-20); MediaPipe pose assistant in experimental/demo-capable state.
- street-snap fork main: `5cf1738` (2026-10-25, org PR pending) — pose T108–T110: One-Euro landmark filtering, bone-length retargeting, single-ear fallback; geometric probes only, real-subject fit `not-verified`.
- edge-model-solutions main: `025b23d` — d1 pin `onnx-community/d1-omni-600M-ONNX@4ebc1b97` (417.7 MiB, LFM Open License v1.0); SmolVLM 256M retained as first-experiment artifact.
- inference-engineering main: `852226e` — composition pipeline v1 frozen at street-snap `f809fc2`; later changes are dated amendments, the freeze is never rewritten.

## How changes flow

1. App ships/validates code in street-snap (rules, adapters, scheduling, UI).
2. Every meaningful model/experiment result lands as a dated `evaluations/*.json` entry here, citing the exact street-snap commit; reusable harnesses live in `scripts/`.
3. Generalizable engineering lessons get distilled into inference-engineering documents, citing the frozen app commit and the evaluation record.
4. App docs link back: SOP → evaluation JSON → pinned model. No repository copies another's large artifacts.

## Evidence vocabulary (shared)

`verified` (real-device measured) · `loaded-only` (runs on desktop/emulator with real weights) · `experimental` (provable facts only) · `failed` (explicitly recorded) · `not-verified`. An emulator/QEMU number never predicts Honor Vs3 latency, heat, or battery; a model that loads is not a model that ships.

See also: [cross-project-contract.md](cross-project-contract.md) (ownership matrix and sync-record fields), the street-snap SOP `docs/composition-pipeline-sop.md`, and `inference-engineering/docs/on-device-composition-pipeline-v1.md`.
