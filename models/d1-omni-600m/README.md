# d1-omni-600M ONNX vision decision stack (q4 decision)

This directory holds the second pinned model artifact in `edge-model-solutions`, intended as the low-frequency **decision** branch of the composition pipeline: a rule-based advisor proposes geometry, and this model chooses between named candidates / rates the frame with zero generated tokens.

It is pinned now to unblock the end-to-end wiring into the street-snap app and to make that wiring testable. It is **not** a claim that the model meets aesthetic-quality or real-time requirements.

- Upstream checkpoint: `LiquidAI/d1-omni-600M` @ `02b55d7076f15129e59ab3f94783f32c4b088674`
- ONNX derivative: `onnx-community/d1-omni-600M-ONNX` @ `4ebc1b97bf1477485371c79d1cf8d5e4e8eebfa0`
- License: **LFM Open License v1.0** (commercial use allowed below the US$10M revenue threshold; the full text is [LICENSE](LICENSE)) - unlike SmolVLM's Apache-2.0, this must travel with the weights and must not be silently rebadged
- Runtime format: fp16 SigLIP2 tower + 4-bit token gather + **4-bit** bidirectional trunk/decision head (`MatMulNBits`)
- Runtime target: ONNX Runtime 1.22 ARM64 (vision opset 21, decision opset 18; both import `com.microsoft`)
- Included files: 11 (3 graphs + 3 external-data files + 5 configs, including the license)
- Total: `437,985,561` bytes (`417.7 MiB`)
- Image size: **512×512** (SigLIP2 NaFlex; the earlier SOP note "384×384" was wrong and has been corrected)
- Product status: see `status` in [manifest.json](manifest.json) and [../../evaluations/d1-omni-600m-pin.json](../../evaluations/d1-omni-600m-pin.json)
- Honor Vs3: not verified

## Why this model

d1-omni is explicitly trained for single-forward-pass decisions (`choice` / `noul` / `score`) over a state plus image, with answers read from the model distribution over option markers — no generation, no coordinate parsing. That matches the post-mortem conclusion from SmolVLM: free-form coordinate generation failed (`[0,0,0,0]`, fragments), and ranking a fixed contact-sheet choice is the safer contract. The earlier planning note that imagined free-form `recommend_bbox` output from this model was incorrect; the bbox stays with the rule branch.

## Bundle selection

Per the ONNX repo's parity table against PyTorch fp32:

| Stack | Image max |Δp| |
|---|---:|
| q8 decision + fp16 vision | 0.047 (upstream-reported) |
| **q4 decision + fp16 vision (this bundle)** | not measured for images upstream |

This bundle ships `vision_encoder_fp16` + `embed_tokens_quantized` + **`decision_q4`** (not `decision_quantized`). The official 8-bit decision graph cannot run on ONNX Runtime 1.22 — reproduced on both macOS x86_64 and Android arm64:

```
MatMulNBits '/trunk/conv/in_proj/MatMul_Q8': nbits_ == 4 was false.
Only 4b quantization is supported for unpacked compute.
```

The exported q8 nodes lack the `accuracy_level=4` attribute that makes ORT select the prepacked SQ8 (`CompInt8`) kernel; without it ORT falls back to the unpacked fp32 path, which implements 4-bit only. The 4-bit graph is the supported all-platform path and was verified end-to-end (vision → embeddings → decision → finite marker logits) on ORT 1.22. Audio graphs and the merged text-only model are excluded because the app never sends audio and builds its own prefix layout. Graph IO is recorded in [manifest.json](manifest.json) under `contract`.

## How image decisions work

1. Prepare one 512×512 RGB crop (NaFlex): `/127.5 − 1`, reshape to 1024 patches of 768 floats, `spatial_shapes=[[32,32]]`, all-valid attention mask. Tiling (up to ten tiles + thumbnail) is skipped while the app feeds a fixed square contact sheet.
2. `vision_encoder_fp16` → 256 prefix embeddings.
3. Build the question sequence exactly as upstream `prompt.py`: `<bos> <state>… <q> instructions <opt> <mask> option <opt_end>… <decide>` with reserved delimiter ids 17–21, mask id 16; image budget 896 positions including the 256 prefix.
4. `embed_tokens_quantized(ids)`, prepend the vision prefix, run `decision_q4(…, prefix_len=256, marker_pos, marker_mask, qtype=0 for choice)`.
5. Softmax logits over the option markers. Image answers deliberately skip the per-type temperature; noul marker order is `[false, true]` and must be reversed when reporting P(yes).

Graph IO names/shapes are recorded in the manifest's `contract` block. The tokenizer is a byte-level BPE with a Llama-style split regex (digit runs split 1–3 chars) — no normalizer, no prefix space in the pre-tokenizer.

## Redistribution

The ONNX repo describes itself as a Derivative Work under LFM Open License v1.0. This directory ships the license text unmodified (`LICENSE`) and records both repositories with pinned revisions. The MIT license of edge-model-solutions covers documentation/manifests only and does not override the model license.

See [manifest.json](manifest.json), [SHA256SUMS](SHA256SUMS) and [LICENSE-NOTICE.md](LICENSE-NOTICE.md).
