# On-device inference research

> 收录工程调研、benchmark 计划、可复现的数字与被证伪的设想。每篇笔记区分结论在「云端」还是「端侧」成立，区分一手证据、厂商口径与预印本；未通过本仓库 benchmark 协议检验的内容不称证据。
>
> Engineering surveys, benchmark plans, reproducible numbers, and disproved ideas. Every note separates what holds on-device from what holds in the cloud, distinguishes primary evidence from vendor claims and preprints, and claims no evidence until it passes the benchmark protocol in [`benchmarks/inference-perf`](../../benchmarks/inference-perf/).

## Notes

| Note | Question | Status |
| --- | --- | --- |
| [01 – On-device inference runtimes](01-runtime-survey.md) | Which runtime (ORT / MNN / NCNN / ExecuTorch / LiteRT / MediaPipe / llama.cpp) fits which model role? | external survey, 2026-10 |
| [02 – KV and prefix caching on phones](02-kv-and-prefix-caching.md) | Which cloud-era KV/prefix-cache techniques survive on a phone, and which multimodal prefixes can actually be reused? | external survey, 2026-10 |
| [03 – Prefill/decode disaggregation on-device](03-prefill-decode-disaggregation.md) | Does cloud-style P/D separation make sense on one SoC, at 1–3 decisions/s now and continuous analysis later? | external survey, 2026-10; conclusion: effectively no — see the note |

## Evidence discipline

These notes are **external research**, not repository evidence. They do not use, and must not be quoted as, `verified` / `loaded-only` device results. Sources are tiered inside each note:

1. peer-reviewed systems work and first-party industrial data;
2. vendor documentation and specification-derived numbers (TOPS and bandwidth are often marketing-derived — flagged when so);
3. non-peer-reviewed preprints (treated as order-of-magnitude evidence);
4. engineering blogs and forum reports (practice signals only).

A claim promoted from these notes into a solution record needs a run under [`benchmarks/inference-perf`](../../benchmarks/inference-perf/) on a named device.
