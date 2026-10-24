# Model and pipeline comparison

This matrix consolidates model candidates and the non-model baseline found in the Street Snap source history. `not-verified` means there is no reproducible device result yet; it is not a quality ranking.

| Candidate | Role in composition workflow | Input / cadence | Output contract | Evidence | Main trade-off or open question |
| --- | --- | --- | --- | --- | --- |
| **Rules-first `FastCompositionAdvisor`** | Generate content-related legal crop candidates and a safe visible guide | Current frame; fast local image statistics | Candidate rectangles plus rule reason | `verified` for visible UI and geometry checks in Street Snap `670308e`; not an AI quality claim | Fast and dependable fallback, but it does not learn aesthetics or object identity |
| **SmolVLM 256M q4 ONNX** *(retired from the app catalog 2026-10-20)* | Historical low-frequency candidate comparison experiment | 512×512 A/B/C contact sheet; one serial comparison or guarded second ordering | A/B/C choice with support and consistency gates | `loaded-only`; Android 35 ARM64 AVD sessions created, roughly 19–21 s for the measured path; invalid or inconsistent choices rejected | Checked-in bundle retained as evidence; was too slow on the AVD for live use and not trained for this composition contract |
| **SmolVLM 256M full Int8** *(retired 2026-10-20)* | Same experiment with a faster quantization profile | Same task and candidate contract | Same guarded A/B/C choice | `loaded-only` / historical timing; old coordinate task on AVD about 4.3–6.4 s, while guarded two-order comparisons were about 9.5–11.4 s | Larger package and still no proof of aesthetic improvement; timings are task- and scheduler-specific |
| **SmolVLM 500M** *(never shipped; catalog entry removed 2026-10-20)* | Larger VLM comparison candidate | Was to be benchmarked with the same candidate contact sheet | No app acceptance | `not-verified`; revision `a588b6298f96d180aff4c31e259409d106b5571e`, roughly 347 MiB cited in old app notes | More capacity and a much larger artifact; no local device quality or latency result |
| **YOLOv11-Nano Int8** | High-frequency subject detection / tracking branch | 320×320 preview frame, intended every preview frame | Normalized subject bounding box | `not-verified`; target SOP only, no pinned revision or device run in local history | Likely suitable for a fast stable subject box, but subject detection is not the same as a recommended crop |
| **d1-omni-600M ONNX (q4 decision image stack)** | Low-frequency decision branch: pick between rule-proposed candidates / rate the frame, zero generated tokens | One 512×512 RGB image (SigLIP2 NaFlex; single fixed crop in our use) plus a typed question; intended as a serial low-frequency pass | Distribution over named option markers (`choice` / `noul` / `score`); the model emits **no** JSON and **no** bbox — geometry stays with the rules | `loaded-only`; artifact pinned at `models/d1-omni-600m` from `onnx-community/d1-omni-600M-ONNX@4ebc1b97` (upstream `LiquidAI/d1-omni-600M@02b55d70`), 417.7 MiB; desktop ORT 1.22 forward pass and Android arm64 emulator live-preview run recorded in `evaluations/d1-omni-600m-pin.json` (one accepted margin-0.214 decision, one budget-timeout fallback); real-device latency and aesthetic validity unproven | Purpose-built for option decisions; the official 8-bit decision graph is unusable on ORT 1.22 (missing accuracy_level=4, 4-bit unpacked-only), so the bundle ships decision_q4 (upstream q4 image parity unmeasured); LFM Open License v1.0 carries a US$10M commercial-revenue threshold |
| **`jEV` / `GV` / `GEV` (names as recalled)** | Unresolved VLM suggestion from an earlier discussion | Unknown | Unknown | `not-verified` and **unidentified** | Do not map these names to a checkpoint until the exact repository or model card is supplied |

## Reading the comparison

The target architecture is a composition of roles, not a contest for one universal model:

```text
preview frame → fast subject box (optional) → candidate/geometry layer
             → slower composition choice → schema gate → smoothing → UI overlay
```

The rules-first layer keeps the app useful when the model is absent, slow, stale, malformed or undecided: the shipped d1 branch preserves that failure behavior, and a future YOLO branch must preserve it as well and report its own measurements beside the existing evidence instead of replacing it.

## Promotion criteria

A candidate can move from `not-verified` to `loaded-only` only after its fixed artifact opens and runs on the named device. It can move to `verified` only after the complete camera path meets the stated acceptance checks: fresh valid outputs, controlled stale-result behavior, preview and shutter responsiveness, measured resource cost, and a quality test set independent from prompt tuning.
