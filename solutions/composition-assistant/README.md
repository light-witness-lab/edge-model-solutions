# Composition assistant

Status: `loaded-only` / `degraded`

This solution turns a live camera frame into a bounded composition suggestion. The output is a guide for the product UI; it never silently changes exposure, focus, zoom, or shutter behavior.

## Pipeline

```text
CameraX preview 24–30 fps
  → bounded latest-frame sampling
  → optional high-frequency subject tracking
  → low-frequency composition reasoning
  → score/problem/recommend_bbox schema validation
  → interpolation and stale-result gate
  → overlay box and Chinese guidance
```

The reference split is:

- **Every preview frame:** YOLOv11-Nano Int8 at 320×320 for a normalized subject box, sliding-average smoothing, and a subject-follow overlay.
- **One or two frames per second:** d1-omni-600M Q4 ONNX at 384×384 with an English composition instruction. Its structured result is `score`, `problem_en`, and `recommend_bbox`; a static dictionary supplies Chinese copy.
- **Before rendering:** validate the schema and box, interpolate only fresh recommendations, and discard stale or malformed results.

YOLOv11-Nano and d1-omni-600M are the target SOP components and are not bundled or verified here yet. The first measured low-frequency experiment is the bundled SmolVLM 256M q4 ONNX fallback described below. Both branches share one frame conversion, while the scheduler keeps only the latest pending frame so slow reasoning cannot block the preview.

## Contract

```json
{
  "score": 0.0,
  "problem_en": "subject_too_centered",
  "recommend_bbox": {"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}
}
```

`score` and every coordinate are range-checked. `problem_en` must map to a versioned dictionary. Unknown labels, malformed JSON, invalid boxes, and stale frames are discarded. An invalid model response never becomes a fake UI box.

## What we measured

On an Android 35 ARM64 AVD, the bundled SmolVLM 256M q4 artifact imported into the app-private store and created all three ONNX sessions. A single sampled inference took approximately 19–21 seconds. Repeated invalid token sequences were rejected, so the UI correctly stayed in the expired/no-reliable-suggestion state. This is evidence for the import, runtime, frame, and failure-gate path; it is not evidence of real-time quality on a phone. Honor Vs3 remains unverified.

## Reusable lessons

- Preview rate and inference rate are separate budgets.
- Keep only one running task and one latest pending frame per branch.
- Close camera frames before model generation continues.
- Treat a loaded model and a useful product result as different evidence levels.
- Measure stale-result age, memory, temperature, battery, and shutter response with the complete camera pipeline.
