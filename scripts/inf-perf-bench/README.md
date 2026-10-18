# inf-perf-bench

`benchmarks/inference-perf` 协议的最小可运行骨架（Kotlin/JVM + 桌面 ONNX Runtime 1.22，与 `scripts/d1-prompt-eval` 同一套 pinned 版本）。它只负责把「预热 → 分段纳秒计时 → p50/p95 统计 → 按 `inference-perf/v1` schema 落 JSON」这条管道做对、做成可复跑；**本身不产生任何模型性能结论**。

The runnable skeleton for the `benchmarks/inference-perf` protocol. It validates the timing/percentile/JSON plumbing; it is not a source of model performance claims.

## 两种模式 / Modes

```sh
# 1) 管道自测：确定性合成 workload（排序），无模型参与
./gradlew run --args="--selfcheck /tmp/inf-perf-sample.json"

# 2) 模型探针：枚举模型目录下全部 *.onnx，逐个计时一次「冷 OrtSession 创建」
./gradlew run --args="--model ../../models/d1-omni-600m --workload W1 --threads 4 --out /tmp/w1-probe.json"

# 可选：--warmup 3 --iters 20（默认值，仅 selfcheck 生效）
```

- `--selfcheck` 的输出固定标 `"workload": "synthetic-selfcheck"`，证明计时、预热剔除、百分位与 JSON 序列化可用；**任何情况下都不得把它当模型结果引用**。
- `--model` 目前只测冷 session 创建（含外部权重 `.onnx_data` 的加载与图优化），这是真实但单点的数字；所有前向分段保持 `null`，直到把 `scripts/d1-prompt-eval` 里冻结的 NaFlex vision 布局、prompt encoder 输入与 decision softmax 绑定进来。未绑定的段不编造数字。

## 当前验证状态 / Verification status

2026-10-18，桌面 JVM（macOS，ORT 1.22 CPU，4 线程）：

- selfcheck 跑通：warmup 3 + measured 20，五个分段（vision_encode / prompt_embed / decision_or_prefill / decode 占位 / end_to_end_round）p50/p95/min/max/mean 齐全，结果可复跑（合成 workload 确定性）。
- d1-omni-600m 探针跑通：三个图冷 session 创建均成功、无错误（单次桌面值留在本地报告，不入库、不作证据，升级判据见 benchmark README）。

设备端（Android AVD / Honor Vs3）数字只能由 App 侧仪表化采集后归一化进同一 schema；本 harness 不模拟设备。

## 输出 / Output

输出对齐 [`benchmarks/inference-perf/result-template.json`](../../benchmarks/inference-perf/result-template.json)（`inference-perf/v1`）。证据等级恒为 `not-verified`；原始报告写 `reports/local/`（gitignored）或临时目录，只有归一化、可复现的记录才提交。
