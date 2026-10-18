# Prefill/decode disaggregation on a phone

> 设想很朴素：云端把 prefill 和 decode 拆到不同机器上吃到了红利，手机上能不能让「重的一眼」和「轻的持续判断」错峰跑在不同核心、不同 session 上？本篇的结论偏扫兴——对我们这个零生成 token 的决策 workload，云端意义上的 PD 分离在单芯片手机上基本不成立；真正值得做的是它在端侧的两个「转世」：相位感知的异构后端映射与跨帧缓存。
>
> Survey, 2026-10. External research only — see [evidence tiers](README.md#evidence-discipline). Nothing here is a device result.

## TL;DR

1. Cloud P/D disaggregation is paid for by three things a phone lacks: physically separate devices, separate memory pools, and separate power/thermal envelopes. On one SoC with unified memory, "two instances" is just time-sharing the same die with twice the resident weight — that is not disaggregation.
2. Our workload (vision encode + one zero-token decision forward) **has no decode phase at all**. The multimodal version of the idea is Encode/Prefill/Decode splitting; we only have E+P.
3. The on-device mechanisms with measured evidence are phase-aware backend mapping (vision tower on NPU) and inter-frame KV/feature reuse — not twin model instances. HeteroInfer measured ~40–45 GB/s usable with a single engine vs ~60 GB/s combining GPU+NPU on a Snapdragon 8 Gen 3 (tier-1 source, below).
4. Full VLM understanding at 120 Hz (8.3 ms budget) is physically out of reach today: the fastest measured mobile vision encoder alone takes ~14.3 ms (NPU INT8, 336×336). Realistic streaming VLM operating points in the literature are 4–14 FPS. The deliverable shape is a low-frequency heavy model plus a resident micro-sentinel.
5. Priority order: input shedding (resolution / frame sampling / visual-token budget) → vision-tower architecture + NPU compilation → per-phase quantization → inter-frame caching → persistent hot session / zero-copy → engine-level heterogeneous splitting. Twin-instance P/D comes last, i.e. do not build it.

## 1. What cloud P/D disaggregation is and why it pays

Autoregressive inference has two opposing phases:

| Phase | Compute character | Optimized for | Hardware preference |
| --- | --- | --- | --- |
| Prefill | all input tokens in parallel; high arithmetic intensity; compute-bound | TTFT | dense compute, tensor parallelism |
| Decode | one token per step; full weight re-read per step; memory-bound | TPOT; large batches amortize weight reads | memory bandwidth, data parallelism, older/cheaper cards |

Colocated runs force one parallelism/batching policy on both. Disaggregation puts the phases on separate GPU pools and ships the KV cache between them over RDMA.

Measured cloud results (tier 1):

- **DistServe (OSDI'24)**: Lluna-2-13B on A100, colocated ~1.6 rps goodput vs prefill-only 5.6 / decode-only 10; up to 4.48× more requests served within SLO, or a 10.2× tighter SLO at the same compliance. <https://arxiv.org/abs/2401.09670>
- **Splitwise (ISCA'24, Microsoft)**: Llama-2-70B/BLOOM-176K, KV transfer over InfiniBand with "no perceptible delay"; up to 1.4× throughput / 20% lower cost vs H100 baseline; 2.35× vs A100 at equal power. <https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/>
- **Mooncake (FAST'25 best paper)**: centralized KV-cache disaggregation; +59%–498% SLO-compliant throughput in traces, backed by up to 8×400 Gbps RDMA and a PB-scale external KV pool. <https://www.usenix.org/conference/fast25/presentation/qin>
- **EPD (ICML'25)** splits multimodal inference into Encode/Prefill/Decode pools: up to 71% lower TTFT, 15× lower peak memory, 22× larger batch — still multi-machine cloud. <https://arxiv.org/abs/2501.15460>

The three sources of the gain: (a) interference physically disappears with separate pools; (b) the decode pool amortizes weight-bandwidth over a **large batch of independent requests**; (c) heterogeneous hardware arbitrage and independent scaling. All three evaporate for a single request stream on one chip.

The honest same-die relative is **vLLM's chunked prefill** — time-slicing prefill chunks around prioritized decode steps on one GPU. That, not DistServe, is the analog available on a phone. <https://docs.vllm.ai/en/stable/configuration/optimization/>

## 2. What transfers to a phone and what does not

### Fundamentally does not transfer

| Cloud mechanism | Phone reality |
| --- | --- |
| Separate memory pools; KV over network | Unified memory: "KV transfer" degenerates to passing a pointer (zero benefit), while two sessions still contend for the same ~77–85 GB/s LPDDR bus (no isolation) |
| Independent power/thermal envelopes | One SoC, one heatsink: big-core + NPU bursts push both instances into the same thermal wall sooner (sustained GPU load drops 30–60% on flagship phones, §3.4) |
| Decode-pool batching amortizes weights | One request stream, batch 1; decode stays underutilized; no pooling gain |
| Per-instance optimal parallelism | Duplicated weights (~0.3–0.6 GB per 600M W4 copy) plus duplicated KV plus sync overhead; in our zero-decode workload the D instance sits idle |
| P on big cores, D resident on little cores | HeteroInfer concludes CPUs are control-plane only for generation ("too energy-inefficient for compute"); the realistic low-power resident path is a sensing-hub-class micro-NPU with a sentinel model, not a second copy of the same model |
| 120 Hz phase staggering | Cannot beat the compute floor (§4); staggering only helps as pipelined overlap when the sampling rate is far below display rate |

### Does transfer, with measured support

1. **Phase-aware backend mapping (the real on-device avatar of E/P separation).** Snapdragon 8 Elite measurements of FastVLM-0.5B: NPU vs CPU gives only 1.64× on prefill and 1.18× on decode (bandwidth-bound), but the **vision encoder is 20–45× faster on NPU** and 1.7–6.9× faster than GPU (tier-3 single-case preprint — treat the ratios as indicative). <https://arxiv.org/pdf/2606.27906>
2. **Tensor-level GPU+NPU co-execution aggregates bandwidth.** HeteroInfer (SOSP'25, Snapdragon 8 Gen 3): W4A16 billion-class models, 1.34–6.02× end-to-end vs the best single backend; prefill up to 8.68×, decode 1.50–2.53×; prefill as low as 2.23 W, 55% less energy than GPU-only. Cost: ~400 µs per GPU-NPU sync, same magnitude as a single kernel — splitting too fine loses. <https://arxiv.org/html/2501.14794v2>
3. **Frame-paced pipelines, not instances.** NPU doing vision/heavy forward while a CPU/low-power path runs a cheap gate is an async double-buffered pipeline with explicit bandwidth yielding, not DistServe-style pool separation.
4. **Persistent hot sessions.** NPU executor cold start ~536 ms vs ~84 ms CPU; one NPU-only end-to-end cold latency of 26.8 s (compiler work) vs 2.08 s hot (tier 3). A viewfinder app needs a persistent worker with pinned residency. <https://arxiv.org/pdf/2606.27906>
5. **Speculative decoding (draft/target) is the only "two models" pattern actually used on phones** — Qwen-0.5B draft → 7B target at 2–3× / 40+ tok/s on Android (tier-4 blog). It matters only for long generation; it is meaningless for zero-token decisions. <https://mvpfactory.io/blog/speculative-decoding-on-android-wiring-a-draft-model-to-llama-cpp-for-2-3x>

## 3. The real phone constraints (2025–2026 flagship class)

### 3.1 Memory bandwidth — decode's ceiling, not even fully reachable

| SoC | Memory | Theoretical bandwidth |
| --- | --- | --- |
| Snapdragon 8 Gen 3 | LPDDR5X-4800 | 76.8 GB/s (HeteroInfer's platform text says 68; measured bulk-copy ceiling 61.9) |
| Snapdragon 8 Elite | LPDDR5X-5300 | ~84.8 GB/s |
| Dimensity 9500 | LPDDR5X-10667 | ~85.3 GB/s |
| A18 / A19 Pro | LPDDR5X-9600 | ~75.8–76.8 GB/s |

Spec-derived (tier 2): NotebookCheck / NanoReview / Low End Mac SoC pages; Qualcomm does not publish exact GB/s. Key tier-1 measurement: on 8 Gen 3 any single engine sustains only 40–45 GB/s; GPU+NPU together reach ~60. A100/H100 deliver 1.55–3.35 TB/s HBM — phones are 20–40× lower, and decode scales linearly with bandwidth.

Roofline sanity check for a 600M W4A16 stack (~300–330 MB weights): one full weight read per decode token is ~4 ms at full bandwidth, ~8 ms at a realistic 40 GB/s, typically 20–40 ms with operator overhead. **We generate ~zero tokens, so this bill almost never arrives; the bill we pay is vision encode + prefill.**

### 3.2 VLM phase throughput on a real phone — the most directly useful table

First-party Qualcomm AI Hub data, Qwen2.5-VL-7B q4, Galaxy S25 (8 Elite), captured from the model page: <https://aihub.qualcomm.com/models/qwen2_5_vl_7b_instruct>

| Backend | prefill tok/s (ctx 2048) | decode tok/s | TTFT range |
| --- | --- | --- | --- |
| CPU (llama.cpp) | 36.9 | 9.4 | 3.5–111 s |
| GPU | 87.7 | 5.3 | 1.5–47 s |
| NPU (llama.cpp) | 474.9 | 8.2 | 0.27–8.6 s |
| **NPU (QAIRT/GenieX)** | **873** | **10.6** | **0.15–2.3 s** |

Read it this way: prefill varies ~24× across backends; decode is welded to 5–14 tok/s everywhere (<2× spread). Optimize E/P; do not fuss over D. (A laptop-class X2 Elite CRD reaches 1756 prefill / 23 decode on the same page — a genuine generation gap between phone and larger silicon.)

Historical reference: MiniCPM-Llama3-V 2.5 (8B) on Xiaomi 14 Pro went from 64.2 s vision encode / 1.3 tok/s on llama.cpp CPU to 17 s / 3.2 tok/s after optimization + NPU (tier-1 paper, 2024 — the 2025–26 stack has since moved substantially). <https://arxiv.org/abs/2408.01800>

### 3.3 TOPS claims are unusable for planning

Qualcomm publishes relative NPU gains, not absolute TOPS; the "45 TOPS" figure belongs to the PC-class X Elite platform total. HeteroInfer measured ~10 usable TFLOPS NPU / ~1 TFLOPS GPU (theoretical 2.8) on 8 Gen 3 — several times off marketing numbers. MediaTek Dimensity 9500 ~100 TOPS and Apple Neural Engine ~35 TOPS are third-party estimates (Apple publishes nothing), and MLX — used by most iPhone LLM demos — does not target ANE at all. Qualcomm itself published a 2025 note on why dense vs sparse TOPS are not comparable. <https://www.qualcomm.com/news/onq/2025/07/dense-tops-vs-sparse-tops-whats-the-difference>

### 3.4 Sustained performance and heat

- 3DMark 20-loop stress tests (PhoneArena/NotebookCheck compilation, tier 2): 8 Elite phones throttle 30–50% — S25 Ultra 6208→2981 (~48%), plateau by loop 7; some 8 Elite Gen 5 phones drop 55–60%; only fan-cooled gaming phones hold ~80–100%. <https://www.phonearena.com/news/the-snapdragon-8-elite-gen-5-is-fast-but-not-all-phones-can-handle-the-heat_id175544>
- Sustained-LLM test, Qwen2.5-1.5B q4, 20 rounds (tier 3): iPhone 16 Pro drops ~40% within 3 rounds to a 23.7 tok/s thermal plateau (peak 40.5); S24 Ultra decays ~15% over 20 rounds; a 1 s gap does not permit thermal recovery. <https://arxiv.org/pdf/2603.23640v2>
- FastVLM-0.5B ×100 on 8 Elite (tier 3): NPU path steady at 40.8 °C skin vs 51.3 °C CPU; after 15 minutes the CPU path hits an 82 °C throttle point while NPU stays at 66 °C. For an always-on viewfinder, NPU is not just faster, it is the only sustainable path. <https://arxiv.org/pdf/2606.27906>

### 3.5 Camera/ISP contention — the real opponent of "staggering"

- Apple developer forum (tier 4): ANE is a single serial resource system-wide; AVCapture runs its own models on it, and a 7%-utilization client can still block behind the camera stack. <https://developer.apple.com/forums/thread/817111>
- Edge-SoC case study (85 TOPS NPU, vendor blog tier 4): ISP/NPU DMA contention on the LPDDR controller pushed end-to-end to 120 ms vs a 30 ms target at only 34% NPU utilization. <https://promwad.com/news/edge-ai-accelerators-custom-hardware-for-smarter-accelerators>
- **SERENO (OSDI'26, tier 1)**: on production phones the SoC prioritizes NPU memory traffic for critical media tasks; a background LLM inheriting that priority causes foreground jank. Using speculative-draft steps as sub-millisecond yield points cut foreground jank by up to 92.6% while *increasing* LLM throughput up to +67.9%. <https://dong.mk/static/papers/26osdi26-sereno-xin.pdf> — meaning: with the camera open, NPU bursts need explicit bandwidth courtesy or preview drops frames.

### 3.6 App memory budget

ART Java heap is ~192–256 MB (largeHeap ~512 MB on 16 GB devices) and does not cover native. Android 17 adds cgroup v2 total-memory limits (anon RSS + zRAM); Play Vitals P90 thresholds from Feb 2027 are 4.25 GB foreground / 2.0 GB background on 16 GB devices. Two 600M instances fit arithmetically but leave little room with camera buffers and KV, and background residency is effectively killed. <https://developer.android.com/about/versions/17/behavior-changes-all>

## 4. Prior art scan

Found:

1. **HeteroInfer (SOSP'25)** — the closest serious work to "on-device P/D thinking"; fundamentally heterogeneous co-execution, with the ~400 µs sync tax. <https://arxiv.org/html/2501.14794v2>
2. **Phase Matters (2026-06 preprint)** — first VLM E/P/D CPU/GPU/NPU phase map; a "Hybrid-PD" config (encoder CPU, prefill NPU, decode CPU) gave the best cold/hot trade-off; visual-token budget 64→224 stretches TTFS 2.4× without recompilation. <https://arxiv.org/pdf/2606.27906>
3. **P/D-Device (Huawei, 2025-08)** — despite the name, cloud-device disaggregation over the network: cloud does prefill, device decodes, TTFT −60%; requires connectivity and conflicts with a local/privacy positioning. <https://arxiv.org/html/2508.09035v1>
4. **SERENO (OSDI'26)** — the strongest systems evidence for "staggered scheduling" under bandwidth contention.
5. **Inter-frame KV reuse — far more relevant than P/D for a viewfinder**: InfiniPot-V (Qualcomm AI Research, training-free, KV peak −94%, real-time target 14 FPS, 0.5% compression overhead, <https://arxiv.org/html/2506.15745v1>); DyCoke temporal token merging (~1.5×/1.4×, <https://arxiv.org/html/2411.15024v1/>); StreamingVLM (compact KV sliding window, 8 FPS even on H100 — evidence of how hard streaming VLM is, <https://arxiv.org/html/2510.09608v2>); V-Rex on edge accelerators (3.9–8.3 FPS, <https://arxiv.org/abs/2512.12284v2>).
6. **Frameworks**: llama.cpp / ExecuTorch (Vulkan, QNN, 8-bit KV, 4 shards) / MLC (prefill chunking, q4 KV) all run P+D in a single process with a single context; no production support for two on-device P/D instances. See the [runtime survey](01-runtime-survey.md).
7. Silicon direction: the 8 Elite Gen 5 **Sensing Hub dual micro-NPUs** (low-power resident) plus the big NPU for bursts is the vendor-sanctioned hardware shape of "little sentinel / big reasoner". <https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-gen-5>

Not found: no phone vendor, chip vendor, or mainstream framework ships a productized split of one model into a big-core prefill instance and little-core decode instance; no paper reports positive results from a DistServe-style twin-instance design on single-chip UMA. Every on-chip system instead maps phases/tensors to heterogeneous engines within one engine stack. That absence is itself the finding.

## 5. Consequences for our pipeline

### Correct the problem definition first

Our current 10–25 s per decision on 256–600M models is not explained by "missing P/D separation". The AI Hub table suggests the likely causes: a CPU/general stack (CPU prefill ~37 tok/s vs 873 on the vendor NPU path for a 7B model), too many visual tokens, no phase-specific NPU compilation, cold sessions. Fix the stack before redesigning the architecture.

With ~zero generated tokens there is nothing to disaggregate. The cost equation is:

```text
decision interval = vision tower (E) + visual_token_count × prefill-per-token (P) + one decision head
```

All three multipliers live on the E/P side. If the aesthetic gate rejects INT8 (our SmolVLM history), note that FastVLM INT8-on-NPU matched FP16-on-CPU semantically while HeteroInfer insists on W4A16 — **per-phase quantization** (vision INT8, LLM W4A16) is more likely to pass the gate than model-wide INT4.

### Priority order toward higher analysis rates

1. **Input shedding — the largest lever (1–2 orders of magnitude available):** ROI/resolution reduction, event-triggered (scene-change) frame sampling instead of a fixed cadence, visual-token compression. FastVLM emits 256 tokens at 1536² with sub-120 ms TTFT for 0.5B on an iPhone 16 Pro (<https://machinelearning.apple.com/research/fast-vision-language-models>); MiniCPM-V 2.6 compresses 1.8 MP to 640 tokens; token budget 64 vs 224 halves TTFS in the Phase Matters data.
2. **Vision-tower architecture + full NPU compilation**: encoder choice alone differs 7× (14.3–103.5 ms), more than any backend switch; NPU vision is 20–45× CPU.
3. **Per-phase quantization**: encoder INT8 / LLM W4A16, keep the decision head high precision.
4. **Inter-frame KV/vision reuse**: adjacent viewfinder frames are highly redundant; this is the algorithmic version of the "P/D payoff" — it eliminates repeated prefill, our actual cost. InfiniPot-V-style training-free compression gives a hard memory ceiling.
5. **Engineering residency**: persistent NPU worker (no recompilation), camera→NPU zero-copy, SERENO-style yielding to protect preview, optionally HeteroInfer-style bandwidth aggregation.
6. **Twin-model/twin-instance P/D: do not build.** If free-form generation (natural-language suggestions) is ever added, try speculative decoding first, then E/P backend mapping; pool-style P/D remains meaningless.

### The honest 120 Hz position

At an 8.3 ms frame budget the fastest vision encoder alone exceeds the budget at 14.3 ms, so per-frame full VLM understanding is physically impossible regardless of model size. The shippable shape is tiered:

```text
120 Hz preview (rendering only)
  └─ sub-ms CV sentinel on Sensing Hub / micro-NPU (saliency, scene change, rule geometry)
       └─ event-triggered small VLM at 5–15 Hz
            └─ heavy composition decision async at 1–3 Hz (current cadence)
```

External wording should say "120 Hz preview with content-adaptive understanding frequency" — matching the 4–14 FPS reality of StreamingVLM/InfiniPot-V — rather than "120 Hz AI analysis".

### Falsification checklist for our benchmark

These map directly into [`benchmarks/inference-perf`](../../benchmarks/inference-perf/) runs (the W3 continuous-scan workload covers items 3–6):

1. Segment timing: E ms, visual-token count, P ms/token, decision-head ms; cold (new process + compiler work) vs hot, p50/p95.
2. **A direct A/B that falsifies the P/D hypothesis**: single-instance time-sharing vs twin instances, same fixture; record decision interval, RSS, temperature. Expect zero gain and ~2× RSS at zero generation — re-open the question only if a stable >15% win appears.
3. Sustainability curve: 10/30/60-minute runs at 1 s cadence, per-round throughput + skin/junction temperature + joules per decision, NPU path and CPU path separately.
4. Bandwidth: effective DRAM GB/s per engine solo vs combined, GPU-NPU sync µs, NPU utilization; roofline attainment vs theoretical bandwidth.
5. **Camera co-residency**: measure inference latency and preview jank/dropped frames with the real camera preview (and ISP AI pipeline) active — SERENO shows omitting this inflates results badly.
6. Inter-frame caching experiment: at 1/2/5/15/30 FPS sampling, KV/vision cache hit rate, recompute saved, end-to-end latency.
7. Precision-latency Pareto over resolution, visual-token budget (64/128/224), and per-phase bit width — judged against our own aesthetic gate, never a generic benchmark score.
