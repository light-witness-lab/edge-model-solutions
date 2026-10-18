# d1 style-seeded candidate probe (composition choice)

Companion to the Street Snap T10 work (composition vocabulary 9 → 17 styles). It answers
one question that the T07 prompt probe could not: **once the three A/B/C crops of one
frame carry genuinely different composition grammars — because the app injects a
style-seed crop — does d1 follow the grammar-correct crop, or does it keep answering by
physical position?**

- T07 ([`d1-prompt-eval`](../d1-prompt-eval), [`evaluations/d1-prompt-v1-eval-2026-10-18.json`](../../evaluations/d1-prompt-v1-eval-2026-10-18.json))
  showed stronger prompt wording does not move the model; the app kept `comp-choice-v0`.
- T10 therefore fuses richer grammar on the **candidate side**, not in the prompt. The
  rule advisor (`FastCompositionAdvisor.enrichWithStyleSeeds`) replaces the third crop
  with a crop that fits a strongly evidenced style (mirror symmetry, spread vertical
  edges, a single dominant horizon, rhythmic repetition, Otsu tonal separation, or
  bilateral converging lines) when the energy-ranked top three have no such crop.
- This probe renders exactly that situation and runs the **same shipped v0 prompt**,
  so it measures candidate-side intervention only.

## What it tests

20 deterministic, programmatic 512×512 contact sheets (no RNG; geometry mirrors the app's
`CandidateContactSheet`: 512 canvas, rgb(244) ground, bold A/B/C labels, and the square
crop contained as a 208×208 image centred in each 240×208 photo area).

Each of six styles has one synthetic 416×416 source "frame" (mirrored facade, sky/ground
horizon, eight-post colonnade, four spread beams, road convergence to an upper-centre
vanishing point with a small figure, hard light/dark split with a subject on the
boundary). Three square crops of that **same frame** become the panels:

- **GOOD** keeps the style's structure (almost always the full frame centred);
- **BAD-1 / BAD-2** crop into a corner sliver or cut the structure away (a single window,
  sky only, ground only, two posts, one edge beam, one diagonal with the apex gone, an
  all-dark or all-light panel).

GOOD rotates through A/B/C across three sheets per style (18 discrimination sheets);
P1/P2 are identical-panel bias probes. Because GOOD is always a full-frame crop and BAD
panels are always tight crops, a low-level "prefer the wide one" bias gives no answer
about physical position — following content across slots is the only way to score.

## Pre-registered decision rule (set before any number was seen)

1. argmax correct **≥ 12/18 (2/3)** → the model reads style grammar; candidate seeding
   could improve ranking quality, pending a real labelled photo set.
2. a result near the T07 v0 level (9/18) together with P1/P2 mean max-prob **≥ 0.40** is
   a positional responder; style seeding is then a safe **diversity-only** change and the
   0.10 selection gate is not relaxed.
3. fewer than 4/18 between any two per-style groups = no observable difference.

These probes measure sensitivity to obvious geometric framing signals, **not aesthetic
truth**; shapes on grey are far from photographs. Outcome does not change the
`loaded-only` evidence level. Geometric scenes are never used to tune the rule layer.

## Result (2026-10-18)

Real pinned graphs, desktop ORT 1.22, 4 CPU threads (`evaluations/d1-style-seed-eval-2026-10-18.json`):

| Metric | Value |
| --- | --- |
| argmax correct | **5/18** |
| gate-selected correct (margin ≥ 0.10) | 5/18 |
| correct when GOOD panel is A / B / C | 4 / 0 / 1 |
| per-style correct (3 each) | SYMMETRY 0 · HORIZON 0 · VERTICAL 1 · REPETITION 1 · CONVERGENCE 1 · CONTRAST 2 |
| argmax physical-slot distribution (when it chose) | A9 · B0 · C1 |
| abstentions (no marker passed the margin gate) | 8/18 |
| bias mean max-prob (P1/P2) | 0.407 |

**Decision: positional responder — candidate seeding stays a diversity-only change.**
The model abstained on 8/18 sheets, and when it did choose it picked slot A in 9 of 10
decisions regardless of where the grammar-correct crop was (correct 4/6 when GOOD was A,
0/6 when GOOD was B, 1/6 when GOOD was C). No style group reached content following;
5/18 is below even the T07 v0 baseline of 9/18 on a different, easier probe. P1/P2
max-prob 0.407 meets the pre-registered positional-bias condition.

Interpretation and boundary: widening the *grammatical* difference between the three
crops does not give d1 content-following on synthetic geometry, so the app must not treat
the model's choice as evidence about composition grammar — the 0.10 gate and the
"frame coordinates always come from the rule layer" contract are unchanged. The
style-seed crop is still shipped because it is correct on the rule side (the scene
detector genuinely supports the style) and it makes the blue-frame advice set richer;
it is not shipped to improve model win rate. Shapes on grey cannot prove what the model
does on real photographs; that evidence still requires the independent labelled photo
set, and the level stays `loaded-only`.

## Reproduce

The files under `src/main/kotlin/dev/streetsnap/app/edge/` are snapshots copied from the
Street Snap app / the frozen T07 harness on 2026-10-18 (prompt encoder, NaFlex patch
layout, decision softmax/gate, BPE tokenizer); `verify/Main.kt` is the probe and inlines
the shipped v0 wording.

```sh
# render every sheet as PNG for visual inspection without loading the model
./gradlew run --args="dump /tmp/style-seed-probes"

# run the real pinned graphs (vision + embeddings + decision_q4, ORT 1.22, CPU)
./gradlew run --args="/path/to/models/d1-omni-600m /tmp/style-seed-eval.json"
```

Deterministic on CPU: the T07 harness produced identical per-sheet probabilities on a
same-day re-run.
