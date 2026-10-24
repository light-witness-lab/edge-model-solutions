# SmolVLM 256M ONNX q4

This is the first pinned model artifact in `edge-model-solutions`. It is included to make the initial experiment reproducible; it is not a claim that the model meets real-time product requirements.

- Upstream: `HuggingFaceTB/SmolVLM-256M-Instruct`
- Revision: `7e3e67edbbed1bf9888184d9df282b700a323964`
- Runtime format: ONNX q4
- Runtime target: ONNX Runtime ARM64
- Included runtime files: 7
- Total: `182,318,056` bytes (`173.9 MiB`)
- Local validation: Android 35 ARM64 AVD can import the bundle and create vision, embedding, and decoder sessions.
- Product status: `loaded-only`; AVD inference took approximately 19–21 seconds and did not produce a reliable composition box. **Retired from the street-snap app catalog on 2026-10-20** (kept here as the first experiment/failure artifact; d1-omni-600M is the only shipping ranker).
- Honor Vs3: not verified.

The repository intentionally includes only the selected q4 files used by the experiment. q4f16, generic quantized, cache, and duplicate export variants are not included.

See [manifest.json](manifest.json), [SHA256SUMS](SHA256SUMS), and [LICENSE-NOTICE.md](LICENSE-NOTICE.md).
