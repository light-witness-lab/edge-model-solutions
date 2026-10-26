# edge-model-solutions

> Practical on-device model solutions for real-world visual, audio, and multimodal products.
>
> 面向真实产品的端侧模型解决方案：从输入采集、模型组合到结构化输出、可视化和性能评测。

`edge-model-solutions` is the solution and evaluation repository for the Light Witness Lab organization. It documents how small on-device models can be composed into useful product capabilities instead of treating a single checkpoint as the product.

这是一个端侧模型方案库，不是单纯的模型仓库，也不绑定某一个 App。每个方案都记录目标场景、模型角色、输入输出契约、推理流程、可视化方式、设备基准和已知限制。

## What belongs here

- **Composition assistance**: sample a camera frame, track a subject, reason about a better crop, validate a normalized bounding box, smooth it, and render a useful guide.
- **Subject and object understanding**: person, animal, bird-eye or fine-grained keypoint detection, with explicit confidence and failure behavior.
- **Image, audio, and multimodal workflows**: combine image, speech, text, and small specialist models when that produces a better device-side result.
- **Inference optimization**: quantization, input resolution, frame sampling, latest-only queues, CPU/GPU/NPU comparisons, memory pressure, thermal behavior, and battery cost.
- **Evaluation and visualization**: latency, freshness, throughput, stability, accuracy, resource use, and clear overlays for boxes, keypoints, scores, and recommendations.

## A solution is more than a model

Each solution should make the full path reviewable:

```text
input capture → preprocessing → model composition → decision/schema validation
             → smoothing/postprocessing → visualization or product action
```

The repository records which parts are verified, which are experimental, and which failed on a device. A model that loads successfully is not automatically a model that produces a useful product result.

## Reference SOP: composition assistance

The current composition-assistance design separates the camera preview budget from the slower reasoning budget:

```text
Camera preview: 24–30 fps
  ├─ high-frequency branch (target): YOLOv11-Nano Int8, 320×320, every frame
  │    → normalized subject box → sliding average → subject-follow box
  │    (not verified; the shipping app currently uses a pure rule layer,
  │     FastCompositionAdvisor, which proposes crop candidates every frame)
  └─ low-frequency branch: d1-omni-600M ONNX, 512×512 SigLIP2 NaFlex, serial
       → A/B/C rule-proposed crops rendered onto one contact sheet + typed question
       → single forward pass, zero generated tokens
       → softmax over the option <mask> marker logits (choice / noul / score)
       → margin + permutation-consistency gates map the chosen index back to a crop
       → interpolation, stale-result gate → composition guide overlay
       (Chinese guidance copy always comes from the rule layer, never from the model)
```

Status of the pieces:

- **d1-omni-600M is `loaded-only`** (2026-10-18): pinned at `onnx-community/d1-omni-600M-ONNX@4ebc1b9` with the 4-bit decision graph, the full vision → embeddings → decision chain runs on desktop ORT 1.22 and on an Android arm64 SDK 35 emulator against the live preview (one accepted margin-0.214 decision; one 30 s budget timeout that correctly fell back to rules). Aesthetic validity and real-device timing remain open; see [`evaluations/d1-omni-600m-pin.json`](evaluations/d1-omni-600m-pin.json).
- **SmolVLM 256M q4 was the first `loaded-only` experiment**: the Android 35 ARM64 AVD created all three sessions, but inference took about 19–21 seconds and did not return a reliable recommendation. It proved the import, runtime, frame conversion, schema gate, and degraded UI path, not real-time quality. It was retired from the street-snap catalog on 2026-10-20 (the checked-in bundle stays as the first experiment/failure record); d1-omni-600M is now the only shipping ranker.
- **YOLOv11-Nano is `not-verified`**: target SOP only, no pinned artifact or device run.

The decision contract is deliberately small — the model never emits JSON or coordinates:

```json
{
  "question_type": "choice",
  "marker_probabilities": [0.204, 0.291, 0.505],
  "selected_index": 2,
  "margin": 0.214
}
```

The selected index maps to a rectangle already proposed by the rule layer. Non-finite logits, thin top-1/top-2 margins, permutation inconsistency, stale frames, timeouts, and busy-device responses are discarded. A failed low-frequency branch leaves the preview and rule overlay usable and shows a clear expired/no-reliable-suggestion state. (An early draft imagined a free-form `score` / `problem_en` / `recommend_bbox` JSON contract at 384×384; inspecting the checkpoint showed this was wrong, and that contract is documented only as history in the solution README.)

## Research line: on-device inference performance

> 研究对象是手机上的整条推理链路：逐段测量延迟分布，逐项检验云端常用技术在单 SoC 统一内存上的成立边界；性能以持续负载下的热稳态为工作点，不采用冷启动数字。
>
> Subject: the full inference path on a phone. Latency is measured stage by stage; cloud-era techniques are examined for where they stop holding on a single SoC with unified memory; performance is quoted at sustained-load thermal steady state, never as cold-start numbers.

The composition stack runs gated decisions at 1–3 per second today; the target is content-adaptive understanding over a continuously analyzing viewfinder. The gap is an engineering research line rather than a model swap, so it is tracked here with the same evidence discipline as a solution record. Three questions structure it:

1. **Runtime landscape** — operator coverage, latency, binary size, threading and NNAPI/NPU support across ONNX Runtime, MNN, NCNN, ExecuTorch, LiteRT, MediaPipe LLM and llama.cpp, and which runtime fits which model role. Notes: [`docs/research/01-runtime-survey.md`](docs/research/01-runtime-survey.md).
2. **KV and prefix caching on phones** — what the vision prefix and fixed question let us cache, how A/B/C candidates should fan out from one shared prefix, which KV techniques are wasted at 256–600M scale, and why adjacent frames cannot share KV directly. Notes: [`docs/research/02-kv-and-prefix-caching.md`](docs/research/02-kv-and-prefix-caching.md).
3. **Prefill/decode disaggregation on-device** — whether cloud P/D separation survives one SoC with unified memory (short answer for our zero-token workload: effectively no), and what its real on-device avatars are — phase-aware NPU mapping, inter-frame reuse, a resident sentinel tier. Notes: [`docs/research/03-prefill-decode-disaggregation.md`](docs/research/03-prefill-decode-disaggregation.md).

The notes are external research with tiered sources (peer-reviewed → vendor docs → preprints → blogs); they are **not** device evidence and never use the `verified` / `loaded-only` labels. Claims turn into evidence only through the executable protocol in [`benchmarks/inference-perf`](benchmarks/inference-perf/) — segmented p50/p95 latency, W1 decision / W2 VLM-choice / W3 continuous-scan workloads, an explicit variable matrix, cold-vs-hot and sustained-thermal runs, gate outcomes recorded beside timing — with a runnable Kotlin/JVM skeleton in [`scripts/inf-perf-bench`](scripts/inf-perf-bench/) (self-check pipeline and d1 graph-load probe verified on desktop JVM 2026-10-18; all model/device numbers stay `not-verified` until measured on named hardware).

## Suggested solution record

Every solution should answer:

1. What user or product problem does it solve?
2. What inputs does it accept and at what rate or resolution?
3. Which models are used, and what role does each model play?
4. What is the output schema and how are invalid results rejected?
5. How does the pipeline behave when frames are stale, models fail, or the device is busy?
6. What are the measured latency, memory, power, temperature, and quality results?
7. Which device, runtime, revision, quantization format, and license were tested?

## Repository layout

```text
solutions/          Scenario-level pipelines and SOPs
models/              Model manifests, provenance, licenses, and adapters
assets/              CC0 coaching libraries: pose templates, composition patterns
benchmarks/          Reproducible runs, schemas, and device results
runtimes/            Runtime notes and integration constraints
visualizations/      Overlay examples, diagrams, and report figures
scripts/             Verify, benchmark, and report helpers
fixtures/            Small self-authored or clearly licensed test inputs
docs/                Shared conventions and engineering notes
```

Example solution areas:

```text
solutions/
  composition-assistant/
  pose-assistant/
  subject-detection/
  animal-recognition/
  bird-eye/
  voice-input/
  multimodal-scene-review/
```

## Evidence levels

Use explicit status labels in every report:

- `verified`: the stated behavior passed the defined device and quality checks.
- `loaded-only`: the model loaded and ran, but quality or product usefulness is not established.
- `experimental`: the pipeline is being explored and may change.
- `failed`: the approach was tested and did not meet the stated requirement.
- `not-verified`: the required device, runtime, or fixture is still missing.

## Model and weight policy

The repository keeps manifests, fixed revisions, SHA-256 values, licenses, adapters, and reproducible download or verification scripts. Large weights are included only when the upstream license, redistribution terms, storage, and provenance are clear. This repository includes the pinned SmolVLM 256M q4 bundle used for the first experiment and the d1-omni-600M decision stack; duplicate quantized variants remain excluded unless a pinned variant is proven unusable (that is exactly why d1 ships `decision_q4` rather than the official 8-bit `decision_quantized` — ORT 1.22 cannot run the latter).

Blobs above GitHub's 100 MiB file limit are stored with **Git LFS** (currently the two d1 external-data files; tracked in `.gitattributes`), while smaller graphs and configs follow the direct-git precedent. Cloning with LFS installed yields byte-identical files whose SHA-256 values match each model's manifest and `SHA256SUMS`.

Licenses stay per-model and must travel with the weights:

- **SmolVLM 256M**: Apache-2.0 (model card terms).
- **d1-omni-600M**: **LFM Open License v1.0** — commercial use is allowed below the US$10M annual-revenue threshold, and the full license text is mandatory alongside the weights. Default app APKs must not bundle these weights silently; the install flow names the model and its license.

Model records must include the source repository, revision, format, runtime, input contract, output contract, file sizes, checksums, and the devices on which the result was obtained.

## Pinned records

### SmolVLM 256M (first experiment)

- fixed revision: `7e3e67edbbed1bf9888184d9df282b700a323964`
- runtime format: q4 ONNX with ONNX Runtime on ARM64
- status: loaded on Android 35 AVD; product-quality crop recommendation not verified
- observed limitation: approximately 19–21 seconds per inference on the AVD, with invalid repeated token output rejected by the result gate
- device status: Honor Vs3 remains unverified

This record is intentionally useful even though it did not meet the real-time requirement.

### d1-omni-600M (marker-choice decision ranker)

- upstream checkpoint: `LiquidAI/d1-omni-600M@02b55d70`; ONNX derivative: `onnx-community/d1-omni-600M-ONNX@4ebc1b9`
- runtime format: fp16 SigLIP2 vision tower + Q4 block-gather embeddings + 4-bit MatMulNBits decision graph, ONNX Runtime 1.22; 11 files / 417.7 MiB
- input: one 512×512 RGB contact sheet of rule-proposed crops plus the upstream `prompt.py` choice sequence; output: softmax over three option marker logits in one forward pass — zero generated tokens, no coordinates
- status: `loaded-only` — desktop forward passes on three synthetic sheets and one accepted live-preview decision (plus one verified timeout fallback) on an Android arm64 emulator; q4 image parity vs fp32 is not measured upstream and aesthetic validity is unproven
- device status: Honor Vs3 remains unverified; emulator timings (vision ≈ 10–12 s, decision ≈ 4.5 s under software rendering) are not representative of phone silicon

See [`models/d1-omni-600m/`](models/d1-omni-600m/) and [`evaluations/d1-omni-600m-pin.json`](evaluations/d1-omni-600m-pin.json).

## Relationship to Street Snap

Street Snap is the first consumer of these solutions. The repositories evolve together through a clear split:

- `light-witness-lab/street-snap` owns the camera UI, capture lifecycle, frame routing, model adapter wiring, APKs, and device evidence.
- `light-witness-lab/edge-model-solutions` owns reusable SOPs, model provenance, checked-in or externally referenced weights, comparison matrices, benchmarks, and failure records.

When a Street Snap experiment changes a model, runtime, prompt, sampling budget, or acceptance result, add a dated evaluation here and link the Street Snap source commit or APK evidence. The app repository does not receive this repository's model weights by default, and this repository does not absorb APKs, signing files, private photos, or local task queues. See [the cross-project contract](docs/cross-project-contract.md) and [the model comparison](docs/model-comparison.md).

## Principles

- Start from a product capability, then choose model roles.
- Keep the pipeline local when the device can meet the requirement.
- Separate preview rate from inference rate.
- Reject stale, malformed, or low-confidence outputs.
- Measure the complete pipeline, including capture and rendering.
- Record failures as carefully as successful runs.
- Keep model licenses and project licenses separate.

## License

Code, schemas, scripts, and documentation are intended to use MIT unless a directory states otherwise. Model weights, checkpoints, conversion files, and third-party assets retain their own licenses and notices.
