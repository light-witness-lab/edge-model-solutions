# Pose assistant

Status: `experimental` — geometric coaching logic proven by JVM tests on synthetic
skeletons; fit, robustness and coaching effectiveness on photographed subjects are
`not-verified` (blocked on the Honor Vs3 real-device pass, T90).

This solution turns a live camera frame of one person into a pose-coaching overlay: a
detected stick figure that hugs the body, an amber target figure retargeted onto that
same body, and joint-level Chinese nudges. It is a **teaching surface only**: it never
drives crop, exposure, focus or shutter decisions, it never stores or uploads frames, and
the whole feature works offline in a demo browser without the model.

## Model pin

| Field | Value |
| --- | --- |
| Model | MediaPipe PoseLandmarker, **lite** tier, fp16 `.task` bundle |
| URL | `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task` |
| Bytes | 5,777,746 |
| SHA-256 | `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` |
| License | Apache-2.0, per the MediaPipe model card |
| Runtime | `com.google.mediapipe:tasks-vision:1.1.1` (build-time dependency, dl.google.com Maven; CPU delegate) |
| App pin | `PoseModelCatalog.kt` in street-snap; weights are never committed or packaged — explicit in-app download into `filesDir`, byte count + SHA-256 verified |

Full (9.4 MB) and heavy (30.7 MB) tiers stay documented options until real-device
measurements justify them.

## Pipeline

```text
CameraX ImageAnalysis (shared stream)
  → PoseFrameSource: 120 ms throttle, 384 px max edge, upright ARGB bitmap, proxy borrowed
  → PoseLandmarker LIVE_STREAM (numPoses 1, detection/presence/tracking 0.5, CPU)
  → 33 normalized landmarks + visibility
  → PoseSmoother (T108): one One-Euro filter per joint, visibility-gated
  → PoseCoach (pure Kotlin): person gate → per-pose conditions → hysteresis
  → PoseOverlay: white detected skeleton + amber retargeted target + hint card
```

- **Temporal filtering (T108).** Each joint has an independent
  [One-Euro filter](https://dl.acm.org/doi/10.1145/2207676.2208639) (min-cutoff 1.0 Hz,
  beta 0.8, derivative cutoff 1.0 Hz) driven by the real inter-result delta time — the
  LIVE_STREAM cadence is irregular, so no fixed frame rate is assumed. A visible landmark
  updates its filter; a briefly occluded one holds its last filtered position for up to
  350 ms (so crossed arms / one dropped result do not snap the skeleton); past 1000 ms
  away a reappearing point is adopted instantly instead of flying across the frame.
  Empty or near-empty detector results (fewer than 8 visible points) get a matching 350 ms
  presence grace before the overlay clears. Tuning is experimental.
- **Coaching.** Eight standing portrait poses; each condition emits a continuous 0..1
  score and a joint-level hint. Pose score is the unweighted mean; ready at 0.75 confirmed
  over 4 results, drops below 0.60 over 3 results. A missing required limb is surfaced
  before any angle hint. All thresholds are torso-length-normalised and experimental.
- **Target retargeting (T109).** The target figure is not a generic mannequin: a torso
  similarity transform gives the base projection, shoulders and hips are pinned to the
  observed joints exactly, and the head plus every visible limb chain are placed along
  the template's segment directions with **the observed person's own bone lengths**.
  Hidden segments fall back to the similarity projection; a hidden end joint keeps the
  chain connected with the template segment length.
- **Head-pose robustness (T110).** Look-back/chin-up poses hide the far ear, so one visible
  ear suffices: the face-width scale falls back to shoulder width × 0.40 (adult ratio,
  experimental). With neither ear visible the pose reports "face hidden".
- **Degradation contract.** No model, model error or unsupported ABI → demo mode (full
  pose browser, no camera use). Feature off/background → analysis unbound and the
  detector closed. R8 release builds keep flogger/protobuf (the T78 failure record is in
  the app plan doc).

## What is proven

JVM geometric probes on synthetic skeletons (street-snap `5cf1738`, merged to org main via
PR #7, merge `5903f99`, 2026-10-25): jitter attenuation on a static joint with bounded tracking lag under fast
motion, the occlusion-hold/reacquire contract, exact torso anchoring, observed bone-length
retargeting, hidden-segment fallback, single-ear scoring, plus the existing met/halfway/
missing-limb and hysteresis tests. See
[`../../evaluations/pose-assistant-geometric-probe-2026-10-25.json`](../../evaluations/pose-assistant-geometric-probe-2026-10-25.json).

These are structural facts about the code, not photographic evidence: a test body that
satisfies the angle/distance conditions is not proof a real person striking the pose is
scored correctly. Real-subject fit, skeleton-to-body pixel accuracy, coaching
effectiveness, cadence and thermals on the lite tier are all `not-verified` until the T90
device pass and the T91 labelled photo set.
