# KV and prefix caching on phones

> PagedAttention、radix prefix cache、KV 量化、token 淘汰等云端技术迁移到手机之前，先要回答三个问题：KV cache 占多少内存、本工作负载的序列中什么真正恒定、相邻帧像素变化后还有什么可以复用。结论：在 256–600M、短上下文、零到单 token 输出的场景下，KV 内存不是瓶颈；可落地的主要收益是「同帧只编码一次、固定题面只 prefill 一次」，主流端侧 runtime 现已支持。
>
> Survey, 2026-10. External research only — see [evidence tiers](README.md#evidence-discipline). KV-size arithmetic below uses public model configs and is independently recomputable.

## TL;DR

1. At our model sizes KV cache memory is a non-issue: SmolVLM-256M's text stack (SmolLM2-135M, 30 layers, 3 KV heads) costs **22.5 KiB per token in FP16 — 90 MiB at 4k context**. INT8/INT4 KV buys almost nothing here and adds gate risk; do not do it.
2. The prize is **prefill/vision reuse**: a single-token decision costs ≈ ViT encode + one prefill + one decode step. Re-encoding the same frame and re-prefilling the fixed question three times for A/B/C is pure waste. Reusing vision + the constant prefix across the three candidates cuts vision/prefill compute to roughly one third, at a residency cost of a few MiB of KV. Every mainstream mobile runtime can do this today, and our own `scripts/d1-prompt-eval` already computes vision features once per sheet and shares them.
3. Only **MLC LLM** ships vLLM-style paged KV + a radix prefix pool + `ForkSequence` (a natural fit for one-prefill / A-B-C fan-out). llama.cpp offers per-slot contiguous KV with common-prefix reuse; ORT GenAI / ExecuTorch / MediaPipe / Core ML use static pre-allocated KV with no paging and no cross-request prefix pool.
4. **Adjacent viewfinder frames cannot directly share KV** — pixels change and vision tokens sit mid-prompt, which invalidates positional semantics downstream. Research remedies exist (STC-Cacher, RETAKE, V-Rex, MPIC, VLCache) with good numbers, but none has landed in a mobile runtime. The shippable tool is keyframe gating.
5. INT8 KV is a production recipe on the Qualcomm NPU path (ExecuTorch+QNN tutorials pin 8-bit KV), but ORT's int8/int4 KV builder explicitly supports only CPU/CUDA EPs — QNN/NNAPI are refused. INT4 KV is usable only via llama.cpp (CPU/Metal/Vulkan).
6. Cross-layer KV sharing (YOCO/CLA, productized as Gemma 3n's KV sharing + PLE) requires training for it; it is a next-architecture selection criterion, not a switch on existing weights.

## 1. How big is the KV cache, really

```text
KV bytes = 2 (K and V) × layers × tokens × n_kv_heads × head_dim × bytes_per_element
           FP16=2, INT8=1, INT4=0.5
```

Two counter-intuitive facts: KV size is nearly independent of parameter count — it depends only on architecture hyperparameters — and grouped-query head counts dominate it. Grouped INT4 adds ~6–12% scale metadata.

Recomputed from public configs (SmolLM2-135M: L30, 576 hidden, 9 Q / **3 KV** heads, d=64; 360M: L32, **5 KV**; 1.7B: L24, full **MHA 32 KV**; Qwen/Llama from their HF configs):

| Text stack | FP16 per token | 1k FP16 / INT8 / INT4 | 4k FP16 / INT8 / INT4 |
| --- | --- | --- | --- |
| **SmolVLM-256M** (135M, L30, 3 KV) | 22.5 KiB | 22.5 / 11.3 / 5.6 MiB | **90 / 45 / 22.5 MiB** |
| SmolVLM-500M (360M, L32, 5 KV) | 40 KiB | 40 / 20 / 10 MiB | 160 / 80 / 40 MiB |
| SmolVLM2-2.2B text (1.7B, **MHA 32 KV**) | 192 KiB | 192 / 96 / 48 MiB | 768 / 384 / 192 MiB |
| Qwen2.5-1.5B (L28, **2 KV**, d128) | 28 KiB | 28 / 14 / 7 MiB | 112 / 56 / 28 MiB |
| Llama-3.2-3B (L28, 8 KV, d128) | 112 KiB | 112 / 56 / 28 MiB | 448 / 224 / 112 MiB |

Qwen2.5-1.5B's KV is **6.9× smaller** than SmolLM2-1.7B's despite similar parameter count. SmolLM2-135M/360M happen to be 3:1 GQA; the 1.7B is the MHA exception.

Against a phone budget where the app has hundreds of MB to a couple of GB and weights dominate (3B q4 ≈ 1.7 GB, 256M q4 ≈ 0.15 GB), **10–40 MiB of FP16 KV makes KV quantization a negative optimization** at our sizes. KV quantization earns its keep at 3B+, 4k+ contexts, or many coexisting sessions.

## 2. Multimodal vision-prefix reuse

### 2.1 How few vision tokens SmolVLM emits

SmolVLM-256M/500M use SigLIP-B/16 (~93M params): a 512×512 input gives (512/16)² = 1024 patches, and pixel-shuffle (factor 4, 16× spatial compression) leaves only **64 vision tokens per 512 tile** entering the decoder. SmolVLM2-2.2B (SigLIP-SO400M, 384/p14) yields 81. LLaVA-style models instead feed 576–2880 image tokens; SmolVLM's 64–81 is itself a visual-token-budget design. <https://arxiv.org/html/2504.05299>

### 2.2 Reorder the sequence so the constant part is a real prefix

Per-layer K/V of a token depends only on preceding tokens, so an exact shared prefix (same token ids, same RoPE positions) is reusable and only the divergent suffix needs prefill. The A/B/C decision layout should be:

```text
[system prompt][fixed option-budget question][64 shared vision tokens][candidate-specific tail]
 └────────────── constant across all three requests: cached ──────────┘  └ few tokens change
```

The multimodal correctness trap: image placeholders are identical special-token ids, so a naive content-hash over token ids makes **different images collide** on the same placeholder blocks. vLLM mixes an image-content hash (`multi_modal_uuids`) into blocks spanning placeholders for exactly this reason. An app-level cache key must be image-byte hash plus the token prefix. <https://docs.vllm.ai/en/latest/design/prefix_caching.html>

Order-of-magnitude savings (decoder prefill ≈ 2N FLOPs/token): SmolLM2-135M ≈ 0.27 GFLOPs per cached token; a constant part of 64 vision + ~100 question tokens ≈ 44 GFLOPs; SigLIP-B/16 at 512 ≈ 90 GFLOPs per encode. Three unfanned candidates ≈ 3×(90+44) ≈ 400 GFLOPs; fanned ≈ 90+44 plus three one-token decodes — vision and prefill each computed once. Holding the shared KV costs 164 × 22.5 KiB ≈ **3.7 MiB FP16**.

Cloud references for the engineering shape (directly portable in architecture): vLLM 0.11.1's encoder disaggregation caches ViT outputs across requests at zero encoder cost on hit (<https://vllm.ai/blog/2025-12-15-vllm-epd>); a shared-memory prefix cache measured −40.5% TTFT / +69.9% prefill throughput on hit (<https://vllm-project.github.io/2025/11/13/shm-ipc-cache.html>); LMCache extended to multimodal in 2025-07.

### 2.3 Fan-out mechanics on mobile runtimes

- **Sequential fan-out**: prefill the shared prefix once in one session, append tail A, then rewind KV to the fork point for B and C. llama.cpp exposes this via `llama_memory_seq_rm`; ORT GenAI via `generator.rewind_to()`.
- **Parallel fan-out (fork)**: branch three sequences from one KV snapshot. **MLC LLM's `ForkSequence` is the only native mobile API**; elsewhere you would copy KV buffers yourself. For single-token outputs sequential fan-out is sufficient — do not pay for concurrency.

### 2.4 Adjacent frames (1–3 Hz now, 120 Hz north star)

Direct reuse is invalid: pixels differ, vision KV differs, and mid-prompt placement poisons downstream text positions. 2025–2026 research:

- **STC-Cacher**: full ViT on every Nth reference frame; for adjacent frames recompute only ~25% "dynamic" tokens selected by key-projection cosine similarity, reuse K/V for the other 75% (deep-ViT adjacent-frame similarity ≈ 0.85); −45.3% LLM prefill latency at 99% retained accuracy. <https://arxiv.org/html/2512.00891>
- **RETAKE / PivotKV**: keyframe peak selection plus non-keyframe token pruning; handles 8× frame counts, −20% decode latency. <https://arxiv.org/pdf/2412.20504>
- **V-Rex**: dynamic KV retrieval for streaming video VLMs, 3.9–8.3 FPS on edge accelerators. <https://arxiv.org/abs/2512.12284>
- Position-independent multimodal reuse: **MPIC** (−54.1% TTFT, image KV reusable at any position, <https://arxiv.org/html/2502.01960v2>); **VLCache** (computes 2% of vision tokens, reuses 98%, 1.2–16× TTFT, <https://arxiv.org/html/2512.12977v1>).

All are research code; **none is integrated in llama.cpp / MLC / ORT / QNN / MediaPipe**. The production-ready tool today is keyframe gating on frame-diff / SSIM / lightweight embedding distance — at 1–3 decisions/s the system already analyzes a latest frame, and a 120 Hz viewfinder is not a 120 Hz inference rate (see the [P/D note's tiered architecture](03-prefill-decode-disaggregation.md#the-honest-120-hz-position)).

## 3. Runtime reality table (2026-10)

| Runtime | KV layout | Cross-request / prefix reuse | KV quantization | Multimodal prefix |
| --- | --- | --- | --- | --- |
| **llama.cpp** | Per-slot **contiguous** pre-allocated tensor; explicitly not PagedAttention ([discussion #8860](https://github.com/ggerganov/llama.cpp/discussions/8860)) | Server `cache_prompt` compares common prefixes against slot history and skips matched prefill; alternating A/B needs multiple slots + pinning; CLI `--prompt-cache` persists | `f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1`; quantized V cache requires flash-attn | mtmd images inserted at prefill; reuse safe only for identical images — never share placeholder prefixes across different images |
| **MLC LLM** | **PagedKVCache**, same C++ engine on mobile | **Radix-tree prefix pool**, cross-request sharing, `RecycleSequence`, **`ForkSequence`** | Mature weight quants; KV still basically fp16, no official int4/int8 KV | No image-hash specialization; manage keys yourself |
| **Qualcomm QNN / Genie (HTP NPU)** | KV resident on HTP in native layout (MHA2SHA permutation); AR/CL dual graphs; AOT context binaries at 512–4096 buckets | No public prefix-pool API; reuse within a session | **Production recipe: W4 weights, fp16 activations, INT8 KV** (ExecuTorch QNN tutorial) | None |
| **ONNX Runtime GenAI** | Static pre-allocation (`past_present_share_buffer`, full max-length); ONNX opset 24 standardized in-place KV mode | Keep the `Generator` alive for multi-turn reuse; `rewind_to()` for forks; no paging/radix | Builder offers INT8/UINT8/INT4 (optional Hadamard rotation) but **source explicitly restricts to CPU/CUDA EPs; QNN/NNAPI unsupported** | None |
| **ExecuTorch** | Static KV tensors + custom update op (XNNPACK/QNN) | None | INT8 KV on the QNN path; int4 weights on XNNPACK | None |
| **Apple Core ML / MLX** | Core ML persists KV via MLState; MLX takes cache objects | No fork/prefix pool; experimental disk prompt caches in MLX | No system-level KV quant; one paper-grade int4-KV Metal kernel | None |
| **MediaPipe LLM Inference** | Internal static KV | No public prefix API | None | — |

llama.cpp is additionally converging on a unified KV pool (`--kv-unified` direction) and idle-slot KV offload between RAM and VRAM; these remain contiguous-buffer improvements, not block-level global radix reuse. Per the survey agent's own reliability note, newly added 2026 flags should be checked against the release notes of the pinned version before use.

## 4. Quantization, GQA, eviction, pruning, cross-layer sharing

- **INT8 KV is close to free** (llama.cpp q8_0 is the community default; QNN ships it). **INT4 KV is task-dependent**: fine for short chat, degrades long-context/math/multi-step reasoning; the common compromise is q8 K + q4 V (K is more quantization-sensitive). An Android engineering test on Llama-3.2-3B cut 2048-token KV from 224→84 MiB with <0.5-point MMLU/MT-Bench loss (tier 4, <https://mvpfactory.io/blog/kv-cache-quantization-for-on-device-llms-int4-key-caches-sliding-window>). 2-bit schemes (KIVI etc.) have no phone kernels.
- **KV quant helps decode bandwidth, not prefill**: prefill is compute-bound, decode batch-1 is memory-bound (~2 FLOPs/weight-byte, utilization often ~1% on GPUs); the win comes from reading less history+KV per token.
- **GQA/MQA is already the default** in our stacks (3 and 5 KV heads). MLA-style latent compression stays server-side for now.
- **Token eviction (H2O, StreamingLLM, SnapKV, Quest) solves 32k+ context problems, not ours.** 2025–26 independent benchmarks are negative on reasoning: at budgets 128–512 SnapKV stayed near lossless while H2O collapsed (GSM8K 0.33–0.64 vs 0.81) and StreamingLLM to 0.07–0.59; H2O lost 23 HellaSwag points at 10% KV retention on Llama-3.1-8B. Quest (page-level min/max upper bounds, up to 7× self-attention) is GPU research code. At a few hundred tokens, none earns its place.
- **Visual-token pruning is the one relevant technique, with a sharp warning for us.** FastV (drop vision tokens after layer K=2 by attention rank) retains ~92.8% average quality, VisionZip ~93.7% — but aggressive pruning is catastrophic on fine-grained document/OCR tasks (FastV drops to 42.7% relative on document parsing). Aesthetic crop choice is exactly a detail-dependent visual task: prune conservatively (≤30–50%) and validate against our own gate, never against a generic benchmark. No mobile runtime ships it built-in; FastV is simple enough to insert ourselves (delete vision-token positions from KV after a chosen layer).
- **Cross-layer sharing needs architecture/training**: YOCO (one global KV for a few layers, ~80× KV reduction at 65B scale), CLA (shared K/V projections across adjacent layers, ~2× beyond MQA), and the productized **Gemma 3n** pattern (cross-layer KV sharing + PLE embedding offload + MatFormer, ~2 GB for a multimodal E2B on phones). Apple's on-device AFM instead trains with KV sharing + 2-bit quantization-aware training. These are next-generation selection criteria, not switches for SmolVLM weights.

## 5. Prefill vs decode economics, with phone numbers

QNN official figures (W4A16, Hexagon NPU, tier-1 first-party): Llama-3.2-3B on 8 Elite Gen 5 — **decode 16.3 tok/s, prefill 1,332 tok/s**; 1B — decode **73.2 tok/s**. <https://aihub.qualcomm.com/mobile/models/llama_v3_2_3b_instruct> Prefill runs ~80× faster than decode on the same die: for short-output decision tasks, saving prefill matters vastly more than optimizing decode — the opposite emphasis from cloud LLM serving.

For a single-token composition decision:

```text
total ≈ ViT forward (compute) + one prefill (compute) + one decode step (memory)
```

Prefix caching collapses the three candidates' compute portion into one; P/D disaggregation has nothing to gain on a single SoC (no second memory bus). The on-device analog is heterogeneous scheduling — ViT/prefill on NPU, the small decode batch on GPU or big-core CPU — covered in the [P/D note](03-prefill-decode-disaggregation.md).

For **d1-style encoder→decision-head models there is no autoregression and no KV cache at all**. Every frame's full forward is the whole cost; "reuse" reduces to two mundane things our eval harness already does: (1) share one vision feature vector across the three candidates on the same sheet (keep it resident, or batch the three crops), and (2) keyframe-gate across frames. The W1 protocol in [`benchmarks/inference-perf`](../../benchmarks/inference-perf/) must record `prefix_or_vision_shared` so this reuse is measured rather than assumed.

## 6. Three-tier action list

**A. Available now (production, 2026-10)**

1. Decode-phase KV cache — default-on in every runtime.
2. In-session common-prefix reuse — fixed system prompt + option-budget question prefills once (llama.cpp `cache_prompt`, MLC radix, persistent ORT generator + `rewind_to`, QNN/Genie session).
3. One ViT encode per frame + vision-prefix reuse, keyed by image-content hash; candidates as a shared-prefix + divergent-tail sequential fan-out. MLC `ForkSequence` is the only ready-made fork.
4. INT8 KV on llama.cpp (all backends) or the Snapdragon QNN/ExecuTorch path; INT4 KV only via llama.cpp with flash-attn, gated by composition-task evaluation.
5. GQA in existing stacks; keyframe gating at 1–3 Hz (zero-risk).

**B. Research-grade, needs self-integration during 2026**

1. FastV/VisionZip pruning at small ratios with our own A/B and gate thresholds.
2. Adjacent-frame KV/activation reuse (STC-Cacher −45% prefill, RETAKE, V-Rex, MPIC, VLCache) — paper code, no phone kernels.
3. Query-aware sparsity / 2-bit KV (SnapKV, Quest, KIVI) — no benefit at short context.
4. INT4/FP8 KV on NPU — QNN stops at INT8; no production path elsewhere.
5. Cross-layer KV sharing — only by choosing/training a YOCO/CLA/Gemma-3n-class architecture.

**C. Blocked by today's mobile runtimes**

1. PagedAttention / global radix pool / cross-session block sharing: MLC only; llama.cpp is contiguous+slot-scoped, ORT/ExecuTorch static, MediaPipe/Core ML absent.
2. Multimodal automatic prefix caching with image-hash awareness and pooled ViT embeddings: cloud vLLM/SGLang only.
3. ORT int8/int4 KV on QNN/NNAPI: explicitly refused upstream.
4. NPU INT4/FP8 KV and sparse-attention kernels: no production path.

### Mapped to our workloads

- **W2 (SmolVLM choice now)**: do not invest in KV compression — full FP16 KV at 4k is 90 MiB. Engineer: one ViT encode per sheet, question+image resident as a fixed prefix, candidate tails forked sequentially, single-token output. Expected vision/prefill compute ≈ 1/3 for the three-candidate round, at a few MiB residency cost. Measure this as the `prefix_reuse_ms` block of the benchmark schema, and report gate outcomes on the same row — reuse that perturbs logits and trips the permutation gate is not a win.
- **W3 (continuous scan)**: caching does not solve changing pixels; keyframe gating + a lighter vision tower + tight visual-token budgets (SmolVLM's 64 is the model to imitate); run FastV A/Bs at conservative ratios.
- **Next architecture selection**: track Gemma-3n-style KV sharing + PLE and the QNN INT8-KV/W4A16 toolchain. The on-device counterpart of cloud P/D is a ViT(NPU)/prefill(NPU)/decode(GPU-or-CPU) heterogeneous pipeline, not two machines.
