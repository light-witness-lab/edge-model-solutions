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
  ├─ high-frequency branch: YOLOv11-Nano Int8, 320×320, every frame
  │    → normalized subject box → sliding average → subject-follow box
  └─ low-frequency branch: d1-omni-600M Q4 ONNX, 384×384, 1–2 fps
       → image + English composition instruction
       → score, problem_en, recommend_bbox
       → static problem_en → Chinese copy
       → bbox validation, interpolation, stale-result gate
       → composition guide overlay
```

This is the target SOP and its YOLOv11-Nano and d1-omni-600M components are still marked `not-verified` in this repository. The first runnable experiment uses the bundled SmolVLM 256M q4 ONNX artifact as a low-frequency fallback. It is `loaded-only`: the Android 35 ARM64 AVD created all three sessions, but inference took about 19–21 seconds and did not return a reliable recommendation box. The fallback proves the import, runtime, frame conversion, schema gate, and degraded UI path; it does not prove real-time quality.

The product contract stays deliberately small and testable:

```json
{
  "score": 0.0,
  "problem_en": "subject_too_centered",
  "recommend_bbox": {"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}
}
```

Unknown labels, malformed JSON, out-of-range coordinates, stale frames, and low-confidence results are discarded. A failed low-frequency branch leaves the preview and subject-follow branch usable and shows a clear expired/no-reliable-suggestion state.

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

The repository keeps manifests, fixed revisions, SHA-256 values, licenses, adapters, and reproducible download or verification scripts. Large weights are included only when the upstream license, redistribution terms, storage, and provenance are clear. This repository includes the pinned SmolVLM 256M q4 bundle used for the first experiment; duplicate quantized variants remain excluded.

Model records must include the source repository, revision, format, runtime, input contract, output contract, file sizes, checksums, and the devices on which the result was obtained.

## First planned record

The first record is the SmolVLM 256M ONNX composition-assistance experiment used by Street Snap:

- fixed revision: `7e3e67edbbed1bf9888184d9df282b700a323964`
- runtime format: q4 ONNX with ONNX Runtime on ARM64
- status: loaded on Android 35 AVD; product-quality crop recommendation not verified
- observed limitation: approximately 19–21 seconds per inference on the AVD, with invalid repeated token output rejected by the result gate
- device status: Honor Vs3 remains unverified

This record is intentionally useful even though it did not meet the real-time requirement.

## Relationship to Street Snap

Street Snap is one consumer of these solutions. Its camera-specific implementation remains in the Street Snap repository; this repository stores the reusable solution knowledge, model combinations, measurement records, and visual explanations that can support Street Snap and future Light Witness Lab projects.

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
