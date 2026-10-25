# 构图素材库

经典摄影构图法的参考库，机器可读版本见 [`patterns.json`](patterns.json)。共 29 条：
**22 条已端上覆盖**（与 street-snap `CompositionStyle` 枚举一一对应，附中文执行动作），
**7 条标记 `app_covered=false`** 是端上尚未实现但已建档的经典法：

| 未端上法 | 卡在哪 |
| --- | --- |
| 黄金螺旋 | 与三分高度重叠，街拍常被三分替代；可做叠加引导线 |
| 仰拍 / 俯拍机位 | 需要重力方向（设备姿态）输入 |
| S 曲线 | 需要曲线检测，超出当前直线/色块特征 |
| 奇数法则 | 需要人物计数（依赖未来 YOLO 主体分支 T97） |
| 剪影 | 本质是曝光选择，待测光分区与曝光建议联动 |
| 视觉平衡 | 需要质量分布估计，可在现有规则特征上扩展 |

## 约定

- 每条含：中英语标题、分类（placement / lines / depth / space / light / balance /
  shapes / camera_position / anti_pattern / fallback）、端上枚举映射、说明、执行提示。
- 三条反例（水平腰斩、取景太满、主体贴边）与正常法同库建档——端上对它们输出纠正动作，
  而不是单纯识别。
- 证据边界：规则层在合成几何测试上 `verified-for-visible-geometry`；真实照片审美结论
  一律 `not-verified`，以 T91 人工标注集为准。d1 模型在评测中基本是「位置应答器」，
  构图主张当前主要由规则层承担。

## 同步契约

street-snap 每次增删 `CompositionStyle` 枚举后同步本 JSON 的 `app_style` 映射；未实现
的经典法只在本库维护，不回写代码。
