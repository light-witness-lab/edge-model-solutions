# Cross-project evolution contract

`street-snap` and `edge-model-solutions` are one engineering effort with two different responsibilities. Keeping the boundary explicit lets the app iterate quickly while the organization retains reusable evidence.

## Ownership

| Concern | `street-snap` | `edge-model-solutions` |
| --- | --- | --- |
| Camera preview, shutter, AF/MF, P/S/M, orientation, UI | source of truth | link to the app evidence |
| Frame sampling, latest-only backpressure, coordinate transforms | source of truth | reusable SOP and contract |
| Model adapters and app integration | source of truth | adapter notes and compatibility matrix |
| Model weights and provenance | external staging or explicit app bundle | manifest, license notice, SHA-256, selected artifact |
| Latency, memory, thermal and output validity | raw device run and APK evidence | normalized evaluation record |
| APKs, signing material, private photos, local task queue | stays local to the app project | never copy in |

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

The current Street Snap composition branch is source commit `670308e` (`feat/dual-branch-composition`). It uses a rules-first candidate generator and an optional SmolVLM 256M local comparison. The app has shown a valid content-related guide box, while the model's aesthetic selection remains unverified. The model matrix records the target YOLO/d1 split separately so a future adapter or benchmark does not silently rewrite this result.

## What must not be inferred

- A model loading or an ONNX session opening does not prove useful composition quality.
- A legal rectangle does not prove that a model found the subject or chose the best crop.
- A simulator timing does not predict Honor Vs3 timing, thermal behavior, or battery cost.
- A model name mentioned in a discussion is not a pinned dependency until its source, revision, license and runtime contract are recorded.
