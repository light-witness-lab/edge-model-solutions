# Models

Each model directory is a pinned, inspectable artifact or a provenance record. Keep the upstream repository, immutable revision, format, runtime contract, license notice, file sizes, SHA-256 values, and validation status together.

- [`d1-omni-600m/`](d1-omni-600m/) — the **currently shipping** ranker in street-snap: `onnx-community/d1-omni-600M-ONNX@4ebc1b97`, 11 files / 417.7 MiB (large external weights in Git LFS), FP16 vision tower + Q4 embeddings + Q4 decision graph under ORT 1.22, LFM Open License v1.0. See its README for the marker-choice contract and validation status.
- [`smolvlm-256m/`](smolvlm-256m/) — the first experiment. The checked-in bundle is intentionally the selected q4 ONNX runtime subset rather than every export produced upstream. Retired from the street-snap catalog on 2026-10-20 and retained as the first `loaded-only` experiment/failure artifact.
