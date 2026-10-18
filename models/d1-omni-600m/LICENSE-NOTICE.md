# License notice

The model files in this directory originate from:

- **`LiquidAI/d1-omni-600M`** @ `02b55d7076f15129e59ab3f94783f32c4b088674` (checkpoint (c) Liquid AI, Inc.)
- **`onnx-community/d1-omni-600M-ONNX`** @ `4ebc1b97bf1477485371c79d1cf8d5e4e8eebfa0` (Derivative Work: export to ONNX, quantization/cast, graph merges; no retraining)

Both are published under the **LFM Open License v1.0**, whose complete text is included here as [LICENSE](LICENSE). Key constraints, summarized in plain language and without legal force:

- **Commercial-use threshold** (Section 5): commercial use is licensed only while the user or their Legal Entity stays below the Threshold — annual revenue of **US$10,000,000**. Commercial use above the threshold is not licensed and requires a commercial agreement with Liquid AI.
- **Redistribution** (Section 4): provide a copy of this LICENSE, carry prominent notices on modified files, and retain attribution notices.
- **No trademark grant** (Section 7): the names "Liquid AI", "d1", and "LFM" may not be used to endorse or promote a product, apart from customary origin descriptions.
- The Work is provided "AS IS", with the warranty disclaimer and liability limits of Sections 8-9.

This is materially more restrictive than SmolVLM's Apache-2.0. An Android APK bundling these weights must bundle or link this license and must not present the weights as Apache- or MIT-licensed. The default street-snap APK ships without weights and fetches them only after an explicit user install action; that flow should surface the model name, source revision, and this license before download.

No legal review has been performed. Before any distribution beyond personal research, read the full license text and Liquid AI's current model card in full.

The MIT license of this repository applies to repository-authored code, manifests, scripts, and documentation only. It does not replace the model's upstream license or the licenses of its tokenizer and runtime assets.
