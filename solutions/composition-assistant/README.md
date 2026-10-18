# Composition assistant

Status: `loaded-only` / `degraded`

This solution turns a live camera frame into a bounded composition suggestion. The output is a guide for the product UI; it never silently changes exposure, focus, zoom, or shutter behavior.

## Pipeline

```text
CameraX preview 24–30 fps
  → bounded latest-frame sampling
  → optional high-frequency subject tracking
  → low-frequency composition reasoning (marker-choice ranking)
  → probability/margin gate and permutation-consistency validation
  → interpolation and stale-result gate
  → overlay box and Chinese guidance
```

The reference split is:

- **Every preview frame:** YOLOv11-Nano Int8 at 320×320 for a normalized subject box, sliding-average smoothing, and a subject-follow overlay.
- **One low-frequency serial pass:** d1-omni-600M ONNX (fp16 vision + Q4 block-gather embeddings + **Q4 decision**, 417.7 MiB) over one 512×512 image with a typed question. d1 is a decision model: it scores `<mask>` markers per named option in one forward pass with **zero generated tokens**. In the current wiring the image is the same A/B/C candidate contact sheet used by the SmolVLM fallback, so the model answers a `choice` question over rule-proposed crops; it never proposes geometry itself.
- **Before rendering:** validate the marker probabilities and gates, interpolate only fresh selections, and discard stale or inconsistent results.

YOLOv11-Nano remains a non-verified target and is not bundled. The d1 artifact is pinned in [`models/d1-omni-600m`](../../models/d1-omni-600m) (`loaded-only`: checksummed, desktop ORT 1.22 forward pass on three synthetic sheets, and one accepted live-preview decision plus one budget-timeout fallback on an Android arm64 emulator; real-device timing and aesthetic validity still open). The official 8-bit decision graph was rejected because ORT 1.22 only runs its unpacked MatMulNBits path for 4-bit weights (the export lacks `accuracy_level=4`). The first measured low-frequency experiment is the bundled SmolVLM 256M q4 ONNX fallback described below. Both branches share one frame conversion, while the scheduler keeps only the latest pending frame so slow reasoning cannot block the preview.

## Contract

### Current decision contract (d1-omni)

d1 answers typed questions; it does **not** emit JSON or coordinates. For composition the app sends a `choice` question whose options are the rule advisor's candidates:

```text
sequence: <bos> <reserved_7 state>… <reserved_8 question> instructions
          <reserved_9 option> <mask> " a: …" <reserved_10>
          <reserved_9 option> <mask> " b: …" <reserved_10>
          <reserved_9 option> <mask> " c: …" <reserved_10> <reserved_11 decide>
output:   softmax over the three marker logits (image answers use no temperature)
```

The selected index maps back to the rule candidate rectangle; abstention is decided by margin and the same permutation-consistency gate used for SmolVLM. Chinese copy always comes from the rule layer, never from the model.

### Superseded planning contract

An earlier SOP draft assumed free-form `score` / `problem_en` / `recommend_bbox` JSON from this model at 384×384. Inspecting the checkpoint (`LiquidAI/d1-omni-600M`) showed this was wrong: the model only distributes probability over option markers and uses 512×512 SigLIP2 inputs. The draft is kept here only as history; any model emitting free-form coordinates would instead go through the schema gate below — which is exactly how the SmolVLM coordinate path failed and was retired.

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

## d1-omni pin (2026-10-17; q4 revision 2026-10-18)

- Pinned the vision decision stack from `onnx-community/d1-omni-600M-ONNX@4ebc1b97` (upstream `LiquidAI/d1-omni-600M@02b55d70`): 11 files / 417.7 MiB, every SHA-256 verified; see [`models/d1-omni-600m/manifest.json`](../../models/d1-omni-600m/manifest.json) for graph inputs/outputs and the marker contract.
- The decision graph is `decision_q4.onnx` (4-bit MatMulNBits), not the official 8-bit `decision_quantized.onnx`: on ORT 1.22 the q8 graph fails on both macOS x86_64 and Android arm64 with `Only 4b quantization is supported for unpacked compute` because its nodes lack the `accuracy_level=4` attribute that selects the prepacked SQ8 kernel. Upstream reports image parity max |Δp| = 0.047 for the q8 stack; q4 image parity was not measured upstream.
- Verified end-to-end on 2026-10-18: desktop ORT 1.22 (three synthetic 512×512 sheets, all finite marker logits) and an Android arm64 SDK 35 emulator against the live camera preview — one accepted decision (`probabilities=[0.204,0.291,0.505]`, margin 0.214; vision 11.5 s / decision 4.5 s under software rendering) and one 30 s budget timeout that fell back to the rule overlay as designed. Details and raw scores in [`evaluations/d1-omni-600m-pin.json`](../../evaluations/d1-omni-600m-pin.json).
- License changed relative to SmolVLM: **LFM Open License v1.0** with a US$10M annual-revenue commercial threshold. The app's default APK must not bundle these weights silently and the install flow should name the model and license.
- Status is `loaded-only`, not product-verified: graph IO, kernels and the timeout/abstain path are evidenced, but real-device latency (Honor Vs3) and aesthetic validity require the independent labelled set before any product claim.

## Prompt revision v1 rejected (2026-10-18; T07)

A criteria-rich v1 wording — a panel-layout `state`, described options (`A: crop shown in panel A`, …), and a question naming thirds/centered symmetry/breathing-room/edge-cutting criteria — was tested against the shipped symmetric v0 (`"a: crop option A"` …, `"Which crop gives the best composition?"`, empty state) under a rule registered before seeing results. The probe used 20 deterministic programmatic contact sheets (2 bias probes + 18 good-vs-cut/corner probes with the good panel rotated across A/B/C), the real pinned graphs on desktop ORT 1.22, and vision features shared by both wordings.

v1 failed non-inferiority: 9/18 → 6/18 argmax correct, 6/18 → 3/18 gate-selected correct, and the physical-panel distribution rotated from C-leaning (A5/B4/C9) to A-leaning (A13/B3/C2) — i.e. the wording moved positional bias without adding content following; P1/P2 max-prob was unchanged (0.4476 vs 0.4393). The app keeps `comp-choice-v0`; the prompt revision constant and a `prompt=` log field were kept so device evidence stays attributable. Synthetic shapes are OOD from photographs, so this run rejects the claimed improvement but says nothing positive about aesthetic quality — that still needs the independent labelled photo set. Raw data and the frozen harness: [`evaluations/d1-prompt-v1-eval-2026-10-18.json`](../../evaluations/d1-prompt-v1-eval-2026-10-18.json), [`scripts/d1-prompt-eval`](../../scripts/d1-prompt-eval/).

## App integration: d1 as default ranker (2026-10-19; evidence through street-snap 10c050f)

After the pin, the street-snap app shipped the two-branch pipeline for real: the rule advisor is the always-available first branch (blue box, Chinese copy), and d1-omni-600M became the default on-device second-branch ranker in commit `f5d8cae`, with SmolVLM 256M still switchable in settings. Serial inference (`2f45d9b`, `74cbd41`) and on-demand advice with guarded local ranking (`670308e`) landed around it. Raw observations: [`evaluations/app-ranker-observations-2026-10-18.json`](../../evaluations/app-ranker-observations-2026-10-18.json).

- **Rule layer carries the visible product.** At a 250 ms frame clock the rule advisor runs about 2.4 analyses/s on the emulator (up from 0.87/s), with roughly 180 ms of that tick being fixed frame-grab/copy/dispatch overhead rather than advisor compute (p50 ≈ 3 ms). Its visible geometry is independently verified; the model branch never contributes coordinates.
- **Model cadence is deliberately lazy.** One serial model pass at a time, latest-frame gate, a 3–15 s cooldown after each pass, and unchanged scenes reuse the prior result with no repeated overlay emission. The preview never waits on the model.
- **The gate stays strict in the field sweep.** Across 70+ standing positions in both orientations d1 never produced a stable acceptance: portrait 1080×2340 (640×480 analysis) usually approached or exceeded the 30 s budget under software rendering, and landscape 540×1170 (384×288 analysis, different contact-sheet layout) finished in 16–22 s but came back UNSTABLE (the two permuted rounds disagreed) or abstained. SmolVLM 256M produced exactly one stable landscape acceptance (0.9947/0.9964, margins 0.1354/0.2191, ~10.4 s) that did not reproduce — consistent with the offline positional-bias findings, not a quality verdict either way.
- **Degradation is honest and structural.** Model absent, stale, malformed, undecided, inconsistent, or over budget: the fresh rule box stays, and the status line names the reason (timeout / unstable comparison). A 30 s wall-clock budget with this fallback is what makes a 16–40 s software-rendered emulator model usable inside a camera app at all.
- **One analysis stream, many consumers.** A single ImageAnalysis use case synchronously feeds both the rule frame sampler and a 200 ms-self-throttled luma histogram meter on one executor; opening a second analysis stream was unnecessary.
- Status remains `loaded-only`. The 417.7 MiB weights are never in the default APK (explicit in-app download naming the model and LFM license); real-device latency/thermal/shutter behavior on the Honor Vs3 and aesthetic validity against an independent labelled photo set are still open.

## Reusable lessons

- Preview rate and inference rate are separate budgets.
- Keep only one running task and one latest pending frame per branch.
- Close camera frames before model generation continues.
- Treat a loaded model and a useful product result as different evidence levels.
- Measure stale-result age, memory, temperature, battery, and shutter response with the complete camera pipeline.
- Ship a fast independently-useful rule branch first; design the model branch so every failure mode (absent, slow, unstable, malformed) degrades to it silently and names itself in the UI.
- Record model latency per orientation and analysis-stream resolution — frame size and candidate-sheet layout change when the device rotates, and a single latency number is wrong.
- One ImageAnalysis stream can fan out to multiple synchronous consumers on a shared executor; don't pay for a second camera stream until measurement demands it.
