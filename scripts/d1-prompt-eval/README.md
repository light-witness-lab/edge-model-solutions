# d1 prompt-revision probe (composition choice)

Frozen desktop harness behind [`evaluations/d1-prompt-v1-eval-2026-10-18.json`](../../evaluations/d1-prompt-v1-eval-2026-10-18.json).
It compares two prompt wordings for the Street Snap composition-choice task against the
real pinned graphs (`vision_encoder_fp16` + `embed_tokens_quantized` + `decision_q4`,
ORT 1.22), with **vision features computed once per sheet and shared** by both decisions.

## What it tests

20 deterministic, programmatic 512×512 contact sheets (no RNG, geometry mirrors the app's
`CandidateContactSheet`):

- **P1/P2 bias probes** — three identical panels; a position-neutral model should stay
  near uniform.
- **18 discrimination probes** — one GOOD panel (whole high-contrast subject on a thirds
  point, breathing room) vs CUT (subject half outside the photo edge) vs CORNER (sliver in
  a corner). The good panel rotates A/B/C, so following content across physical positions
  is simultaneously the permutation check.

These probes measure sensitivity to obvious framing signals, **not aesthetic truth**;
geometric shapes on gray are far from real photographs. Failure therefore cannot be
generalized to "the model is blind on photos", but it is sufficient to reject a claimed
prompt improvement.

## Pre-registered decision rule (set before any number was seen)

1. adopt v1 only if v1 gate-selected-correct ≥ v0 on the 18 discrimination sheets;
2. and v1 mean max-prob on P1/P2 is no more than 0.05 above v0;
3. a difference under 4/18 is reported as no observable difference.

## Result (2026-10-18)

| | v0 (shipped) | v1 (criteria-rich question, described options, layout state) |
|---|---:|---:|
| argmax correct | **9/18** | 6/18 |
| gate-selected correct (margin ≥ 0.10) | **6/18** | 3/18 |
| argmax when good panel is A / B / C | 3 / 1 / 5 | 6 / 0 / 0 |
| physical-panel argmax distribution | A5 B4 C9 | A13 B3 C2 |
| bias mean max-prob (P1/P2) | 0.4476 | 0.4393 |

v1 was **rejected**: both wordings behave as positional responders on this set (v0 leans
to panel C, v1 leans to panel A); the criteria-rich wording rotated the favoured slot and
halved gate passes without adding content following. The app keeps `comp-choice-v0`;
evidence level stays `loaded-only`. Real photographic judgement needs the independent
labelled photo set, not stronger wording on synthetic shapes.

## Reproduce

The files under `src/main/kotlin/dev/streetsnap/app/edge/` are snapshots copied from the
Street Snap app repository on 2026-10-18 (prompt encoder, NaFlex patch layout, decision
softmax/gate, BPE tokenizer); `verify/Main.kt` is the probe itself and inlines both prompt
wordings so results stay reproducible even if the app's `D1Prompt` changes later.

```sh
./gradlew run --args="/path/to/models/d1-omni-600m /tmp/prompt-eval.json"
```

Deterministic on CPU: a re-run the same day produced identical per-sheet probabilities.

## Resolution probe (2026-10-19, T36)

`verify/ResolutionProbe.kt` reuses the frozen 20-sheet design to test whether the exported
NaFlex vision graph genuinely accepts shorter visual sequences, and what the desktop
latency/choice-stability trade-off looks like across three input sides:

| side | grid | patches | vision prefix | repeat11 median | ratio vs 512 |
|---|---:|---:|---:|---:|---:|
| 512 (shipped) | 32×32 | 1024 | 256 | 1227 ms | 1.00 |
| 384 | 24×24 | 576 | 144 | 608 ms | 0.496 |
| 320 | 20×20 | 400 | 100 | 424 ms | 0.346 |

Each side independently rasterises the same 20 sheets proportionally (16-px patch size
unchanged), with 2 warm-ups and 11 fixed-sheet repetitions; decisions use the shipped v0
wording only.

### Pre-registered rules (set before numbers were read)

- **R4 shape** — a shape failure at a lower side means the exported graph is fixed at 512
  and would need re-export.
- **R1 latency** — consider 320 only if its repeat11 median ≤ 0.70 × 512.
- **R2 accuracy** — vs 512, neither 384 nor 320 may lose > 2/18 argmax-correct or
  gate-selected-correct.
- **R3 bias** — P1/P2 mean max-prob may rise by at most 0.05, and the most-voted physical
  panel must not gain votes.

### Result: no change (`loaded-only`)

All three sides ran with the expected prefix lengths — the exported graph is variable
length, R4 passes and no re-export is needed.

| | 512 | 384 | 320 |
|---|---:|---:|---:|
| argmax correct | 6/18 | 5/18 (−1) | 5/18 (−1) |
| gate-selected correct | 6/18 | 5/18 (−1) | 5/18 (−1) |
| bias mean max-prob (P1/P2) | 0.4477 | 0.4540 (+.006) | **0.5198 (+.072)** |
| physical-panel votes | A2 B1 C9 | A4 B1 C8 | A1 B1 **C10** |
| R1 / R2 / R3 | — | pass / pass / pass | pass / pass / **fail** |

384 clears every gate; 320 fails R3 (confidence inflation on the identical-panel probes and
one extra vote for physical panel C), so the most aggressive side is rejected. The overall
recommendation stays **no-change-loaded-only**: the 512 default stays. The 6/18 baseline is
the v0 positional-responder baseline already documented above, so R2 is only a
do-no-harm gate, not evidence of photographic judgement. Desktop CPU ratios do not
extrapolate to device absolute latency (threads, NNAPI, bandwidth all differ), and the app
side cannot run d1 inside budget at all today, so whether 384 would actually close the
on-device gap is unknown. Switching the app's visual input size is an app-contract change
(`D1NaFlex` snapshot plus model config/manifest) and needs a separate, device-measured
task; this probe deliberately changed no app contract.

```sh
./gradlew run --offline -PmainClass=verify.ResolutionProbeKt \
  --args="/path/to/models/d1-omni-600m /tmp/resolution-eval.json"
```

Evidence: [`evaluations/d1-resolution-eval-2026-10-19.json`](../../evaluations/d1-resolution-eval-2026-10-19.json).
