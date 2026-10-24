# Cross-project evolution contract

`street-snap`, `edge-model-solutions` and `inference-engineering` are one engineering effort with three different responsibilities. Keeping the boundaries explicit lets the app iterate quickly, the organization retain reusable evidence, and runtime engineering knowledge stay decoupled from any single app. See [three-repository-map.md](three-repository-map.md) for the one-page map.

## Ownership

| Concern | `street-snap` | `edge-model-solutions` | `inference-engineering` |
| --- | --- | --- | --- |
| Camera preview, shutter, AF/MF, P/S/M, orientation, UI | source of truth | link to the app evidence | — |
| Frame sampling, latest-only backpressure, coordinate transforms | source of truth | reusable SOP and contract | portable playbook items distilled from shipped code |
| Model adapters and app integration | source of truth | adapter notes and compatibility matrix | runtime/quantization/scheduling research, no app code |
| Model weights and provenance | download by pinned revision; never commit weights | manifest, license notice, SHA-256, selected artifact | references pinned artifacts by repo link, never hosts them |
| Latency, memory, thermal and output validity | raw device run and APK evidence | normalized evaluation record | methodology and evidence-label discipline |
| Frozen pipeline baselines | the shipped commit being frozen | evaluation JSON for the same point | dated frozen-baseline document (amendments only) |
| APKs, signing material, private photos, local task queue | stays local to the app project | never copy in | never copy in |

## Sync record

For every meaningful experiment, add one entry to `evaluations/` containing:

1. the Street Snap source commit and, when relevant, APK path or digest;
2. model id, upstream repository, immutable revision, format and runtime;
3. input size, sample rate, scheduler policy and output schema;
4. p50/p95 or clearly labeled single-run timing, memory and device details;
5. valid-output rate, stale-result behavior and product acceptance status;
6. a link to the app-side implementation or test record.

The evidence level describes the result, not the ambition: `verified`, `loaded-only`, `experimental`, `failed`, or `not-verified`.

## Current handoff

The current Street Snap mainline is org commit `49123ad` (2026-10-24, PR #6; fork main `f83f2e5`). It ships the rules-first candidate generator (~2.4 analyses/s at a 250 ms frame clock on the AVD; 21 composition vocabularies, action sentences, alignment green-frame feedback and a move arrow) as the always-available first branch, with **d1-omni-600M as the only model ranker** (marker-choice contract, single forward, zero generated tokens) since the SmolVLM 256M/500M catalog removal on 2026-10-20. A MediaPipe pose landmarker is the second on-device inference path (demo mode without the model). The inference v1 pipeline is frozen as a baseline in `inference-engineering/docs/on-device-composition-pipeline-v1.md` (source baseline street-snap `f809fc2`); later changes go in as dated amendments, never by rewriting the freeze. The high-frequency YOLOv11-Nano subject branch remains `not-verified` with contracts reserved in the camera module but no wiring.

2026-10-25 (fork `5cf1738`, org PR pending): the pose path gains visibility-gated One-Euro landmark filtering, bone-length target retargeting and a single-ear head-pose fallback; evidence is `experimental` synthetic-geometry JVM probes only (`evaluations/pose-assistant-geometric-probe-2026-10-25.json`), and pose skeleton fit on real subjects stays `not-verified`.

Evidence remains `loaded-only` at best: the d1 gate never stably accepted across 70+ AVD positions (one margin-0.214 acceptance + one timeout in the pin record), and Honor Vs3 real-device timing, thermal behavior, pose skeleton fit and aesthetic quality are all `not-verified`.

## Linking convention

- App → solutions/evaluations: street-snap docs cite files in this repo at the commit recorded in the evaluation record.
- This repo → app: every `evaluations/` entry records the exact street-snap source commit (and APK digest when relevant).
- `inference-engineering` freezes baselines at named street-snap commits and links back; its case study and playbook must not invent numbers absent from this repo's records or the app SOP.

## What must not be inferred

- A model loading or an ONNX session opening does not prove useful composition quality.
- A legal rectangle does not prove that a model found the subject or chose the best crop.
- A simulator timing does not predict Honor Vs3 timing, thermal behavior, or battery cost.
- A model name mentioned in a discussion is not a pinned dependency until its source, revision, license and runtime contract are recorded.
