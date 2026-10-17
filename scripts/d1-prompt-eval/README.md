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
