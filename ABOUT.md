# About

## Repository overview

edge-model-solutions documents how small on-device models can be composed into useful product capabilities instead of treating a single checkpoint as the product. It is a solution and evaluation library from Light Witness Lab — not a model zoo, and not tied to one app.

Each record makes the full path reviewable: input capture, preprocessing, model composition, schema validation, smoothing, and visualization — with fixed revisions, input/output contracts, measured latency, explicit failure behavior, and an evidence level (`verified`, `loaded-only`, `experimental`, `failed`, `not-verified`). A model that loads is never reported as a model that ships.

The first consumer is the Street Snap camera app. Its composition pipeline keeps a pure rule layer on the preview path and runs a low-frequency model only to rank the rule layer's proposed crops: the pinned **d1-omni-600M** ONNX stack answers a typed choice question as a softmax over option marker logits in a single forward pass — zero generated tokens, no coordinates from the model. The artifact is `loaded-only`: the 4-bit decision graph runs end-to-end on desktop ORT 1.22 and an Android arm64 emulator, while aesthetic validity and real-device timing remain open. The earlier **SmolVLM 256M** q4 experiment is kept as the first `loaded-only` fallback record.

Large weights are included only with clear provenance and redistribution terms: smaller files live in Git, blobs over 100 MiB live in Git LFS, and every file is pinned by SHA-256 with its license attached (Apache-2.0 for SmolVLM; the LFM Open License v1.0, with its US$10M commercial-revenue threshold, for d1-omni-600M).

## GitHub About

### Recommended description

On-device model solutions for visual, audio, and multimodal products: composition, recognition, model combinations, optimization, and reproducible benchmarks.

### Shorter alternative

Practical on-device model solutions, evaluation notes, and reusable multimodal pipelines.

### Chinese description

端侧模型解决方案：构图、识别、多模态组合、推理优化与可复现实测。

### Suggested topics

`edge-ai` `on-device-ml` `multimodal` `computer-vision` `speech` `model-composition` `inference-optimization` `benchmarks` `onnx` `onnx-runtime` `android` `composition-assistance` `quantization`
