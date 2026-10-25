# Pose assistant — implementation notes & retrospective, 2026-10-25

*Companion to the solution SOP ([README.md](README.md)) and the geometric probe
(`../../evaluations/pose-assistant-geometric-probe-2026-10-25.json`). This is the "how it
was built and why" record after the T108–T115 iteration batch. Evidence level stays
**experimental**: nothing here has been measured on a photographed subject yet.*

## Timeline

| Batch | Content |
| --- | --- |
| T72 | v1: MediaPipe PoseLandmarker integration, 8-pose catalogue, coach hysteresis, demo mode without model |
| T108 | Per-joint One-Euro filtering with real irregular `dt`, visibility gating, occlusion hold, presence grace |
| T109 | Bone-length retargeting: the target figure is pinned to the observed body and rescaled to the user's own segment lengths |
| T110 | Single-ear robustness for look-back / chin-up proxies |
| T113 | Rolling `dev` pre-release CD: every main merge publishes stable-name debug APKs to the Releases page |
| T114 | Push → PR → CI-gated merge automation via scoped fine-grained PATs |
| T115 | Two-figure visual separation, joint-level pulse guidance, 12-pose catalogue, asset libraries |

## Stack pin

- MediaPipe Tasks Vision **1.1.1**, `PoseLandmarker` LIVE_STREAM on CPU, numPoses 1,
  all three confidence thresholds 0.5; lite fp16 `.task` 5,777,746 B
  (SHA-256 `59929e1d…d574a`), downloaded in-app, never packaged.
- Kotlin 2.2.10 / Compose; the geometry layer (`pose/Pose.kt`, `PoseCoach.kt`,
  `PoseCatalog.kt`, `PoseSmoother.kt`) is zero-Android so it runs under JVM unit tests.
- CameraX ImageAnalysis shared stream, 120 ms throttle, 384 px max edge, proxy borrowed.

## What the iteration proved matters

1. **Time is a first-class input.** Using real inter-result `dt` (never frame counts) and
   treating a backwards clock as a session reset made the filter stateless to reason about.
2. **Three temporal layers, not one smoothing constant.** Joint (One-Euro + 350 ms hold),
   presence (350 ms empty-result grace, 8 visible points), pose (0.75/0.60 hysteresis,
   4/3-result confirm) protect different failure modes.
3. **A co-located target needs a visual grammar.** Once the amber target is retargeted onto
   the observed body it overlaps the white detected skeleton by design. T115 resolves the
   ambiguity with depth cues, not separation in space:
   - observed body → faint **white dashed ghost** (1.5 dp, α .50; α .18 when ready);
   - target → bold **amber solid** (3.5 dp) with a soft glow and **ring joints**
     (dark-outlined hollow rings read on any background);
   - a target joint still > 0.06 normalised-frame units from its observed joint carries a
     slow **breathing pulse ring** — "this joint goes here" without drawing connector
     spaghetti;
   - the hint card carries a one-line legend; on ready the ghost fades and the target
     settles to green.
4. **Anatomy fallbacks for self-occluding poses.** Look-back/chin-up hide the far ear;
   demanding two ears made the coach fail exactly when the user does the pose right. One
   visible ear + an experimental 0.40 head-width/shoulder-width ratio is enough.
5. **Geometry is testable before devices exist.** Exact torso anchoring, own-segment-length
   retargeting, hold/reacquire windows, every catalogue pose's projectability and 12
   synthesised met/neutral coach cases are all JVM assertions. That safety net is why the
   catalogue could double (8 → 12) in one evening without a compiler on the machine.

## Release/automation architecture

- **Two tracks.** Dev track: `dev-release.yml` on every push to main builds preview + debug
  APKs (`-PciBuild=true`, applicationId `.ci`), signs with a runner-generated standard debug
  keystore persisted in `actions/cache` (in-place upgrades work; cache eviction costs one
  reinstall — no secret management), and `--clobber`-uploads fixed-name assets onto one
  rolling tag `dev`. Production track stays manual: `release.yml` on `v*` tags, distribution
  secrets, draft-only.
- **Automation credentials.** Two fine-grained PATs in the local keychain: a personal-owner
  token for fork pushes and an **organization-owner** token for PR/merge (an org with
  fine-grained PAT policy set to *Disable* rejects even owner tokens — resource owner on a
  token is fixed at creation). CI checks must be fully green before the automated merge.

## Failure log (cheap lessons)

- `printf '- …'` in bash parses `-` as an option: four release-note lines starting with
  "- " aborted the first CD run after a fully successful build. Replaced with one heredoc.
- `actions/checkout` defaults to fetch-depth 1; "last 10 commits" listed one line.
- Co-located figures shipped with no visual grammar (T115 fix above).

## What is still not claimed

Real-subject skeleton fit, scoring validity, the 0.40 ratio, One-Euro constants, the pulse
threshold, cadence and thermals on lite: all `not-verified` until the Honor Vs3 pass (T90)
and the labelled photo set (T91). The new T115 poses share the same experimental threshold
status as the first eight.
