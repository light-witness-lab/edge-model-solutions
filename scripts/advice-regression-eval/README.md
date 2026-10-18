# advice regression probe (rule layer)

Re-runnable regression baseline behind
[`evaluations/advice-regression-baseline-2026-10-19.json`](../../evaluations/advice-regression-baseline-2026-10-19.json).
It answers the T37 question — *is the advice varied enough, and does following it work?* —
against the **shipped rule layer** of the Street Snap app, with no model and no labelled
truth. Desktop JVM, zero Android dependencies, deterministic (no RNG).

## What runs

- **Verbatim snapshots** of the app's pure-Kotlin logic at street-snap `e891092`
  (2026-10-19): `FastCompositionAdvisor.kt`, `CompositionActions.kt`,
  `AlignmentFeedback.kt`, `CompositionStyle.kt` (whole files), and
  `NormalizedBoundingBox` extracted verbatim from `CompositionPipelineContracts.kt` (the
  rest of that file references `CompositionFrame`, which lives next to CameraX bindings).
  `internal` visibility resolves because everything is one Gradle module.
- **204 deterministic 96×96 scenes** (`SyntheticScenes.kt`): a flat baseline; a blob scan of
  3 sizes × 7×7 interior grid positions (list adjacency = a small camera move); small-mass
  minimal fields; edge/corner masses; crowded borders; a 15-step horizon walk; vertical-bar,
  diagonal, converging-line, symmetric, triangle, half-and-half, picture-frame and
  foreground-depth arrangements. Each family declares a geometric **intent**; the styles
  actually emitted are recorded, never tuned to match the intent.

## Metrics (fixed before the first run)

- **M1 coverage** — distinct first-choice styles seen. "Expressed" coverage excludes the
  CONSERVATIVE/GENERIC fallbacks (denominator 19), with per-family style sets.
- **M2 distribution** — first-choice counts over all 204 frames, entropy normalised by
  ln(21), top-style share, and the share of frames carrying an executable action sentence.
- **M3 change rate** — every frame is analysed twice and repeat pairs must never change;
  the blob-scan adjacent-edge change rate measures advice following the picture; mean style
  run length over the ordered suite quantifies sticky-vs-jumpy.
- **M4 alignment** — for every scan/minimal frame whose first advice carries an action, an
  "obedient re-framing" walk applies the planner's move directly to the blob parameters
  (60% of the geometric gap per step, 12-step budget, mass kept inside the frame),
  re-renders, and feeds the **real** `AlignmentTracker` (8 confirmation frames, 3-frame
  rearm). Success = an `ALIGNED` pulse within budget.
- **Tracker gate sanity** — 12 framed frames without a prior need must produce zero
  praise; after 12 unframed-needy frames, exactly one pulse must arrive on confirmation
  frame 8.

## Baseline result (2026-10-19)

| metric | value |
|---|---:|
| frames | 204 |
| repeat pairs changed | 0 / 204 |
| median rule-layer advice | 1.88 ms (desktop JVM) |
| M1 expressed coverage | **10 / 19** |
| M2 entropy (normalised) / top-style share | 0.436 / 0.564 |
| frames with an action sentence | 83.3% |
| M3 blob-scan change rate | 0.247 (36/146 edges) |
| M3 full-sequence change rate / mean run | 0.271 / 3.64 frames |
| M4 alignment success | **49 / 147 = 33.3%**, median pulse at step 11 |
| M4 failures | 98/98 stuck at inset 0.03–0.10 |
| gate sanity | 0 stray pulses; armed pulse at frame 20; exactly 1 |

First-choice distribution: THIRDS 115, MINIMAL 51, HORIZON 14, CONSERVATIVE 6, CENTERED 4,
SYMMETRY 4, VERTICAL 4, HORIZON_SPLIT 3, LEADING_LINES 1, CONTRAST 1, DIAGONAL 1.
Styles never winning first place: CONVERGENCE, EDGE_CUT, FOREGROUND_DEPTH, FRAME,
NEGATIVE_SPACE, REPETITION, TOO_TIGHT, TRIANGLE, WIDE_CONTEXT.

What the baseline says (and what it does not):

- Advice on flat geometry **concentrates on THIRDS + MINIMAL (81%)**; nine styles never win
  a first choice. Several families miss their intent by design of the statistics (crowded
  borders read SYMMETRY not TOO_TIGHT; converging lines read CENTERED; triangle masses read
  THIRDS; the small pale-ground mass reads CONSERVATIVE). These mismatches are recorded
  facts about geometric probe design vs rule triggers, **not** aesthetic verdicts.
- The M4 failure pile-up at inset 0.03–0.10 is structural: advisor crop scales step
  0.68/0.76/0.84/0.92/1.0, so the 0.92 crop sits at 4% inset while `FRAME_TOLERANCE` is 3% —
  an obedient walk that lands on a 0.92 first choice never lights the T34 green pulse. 126/147
  walks end with the big centred mass read as SYMMETRY. Any future change to crop stepping,
  the 3% tolerance, or style assignment moves these numbers and will be visible as a diff.
- This is a **regression tripwire**, not a quality claim. Synthetic shapes cannot measure
  photographic usefulness; the d1 gate, model ranking and scheduling do not participate;
  the walk assumes an obedient operator with exact parameter control; desktop latency is not
  device latency. Evidence level: `loaded-only`; aesthetic quality remains not-verified and
  needs the independent labelled photo set.

The scene suite is frozen for this baseline. A geometry revision that targets the missing
styles is a v2 probe with its own pre-registration, not a tune-up of this run.

## Reproduce

```sh
cd scripts/advice-regression-eval
JAVA_HOME=<jdk17+> ./gradlew run --offline \
  --args="../../evaluations/advice-regression-baseline-2026-10-19.json"
```

### Re-syncing the snapshot

Copy the four whole files verbatim from street-snap (`app/.../composition/`) and re-extract
`NormalizedBoundingBox` if `CompositionPipelineContracts.kt` changed; keep packages
identical; bump the `snapshot.source` field in `AdviceRegressionProbe.kt`. A snapshot whose
behaviour drifted fails the gate-sanity checks or changes repeat determinism first.
