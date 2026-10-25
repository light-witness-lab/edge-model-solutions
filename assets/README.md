# 素材库（Assets library）

可移植、可审查、版权干净的拍摄引导素材：姿势模板库与经典构图法库。数据格式为 JSON +
脚本生成的 SVG，不含任何第三方摄影图片。

| 库 | 数据 | 示意图 | 端上对应 |
| --- | --- | --- | --- |
| [姿势库 pose-library](pose-library/README.md) | [`poses.json`](pose-library/poses.json)（12 个站姿模板） | [`diagrams/pose-library.svg`](pose-library/diagrams/pose-library.svg) | street-snap `PoseCatalog.kt` |
| [构图库 composition-library](composition-library/README.md) | [`patterns.json`](composition-library/patterns.json)（29 条，22 条已端上覆盖） | —（规划中） | street-snap `CompositionStyle.kt` |

## 约定

- **版权**：坐标、文案、示意图均为本项目原创，统一 **CC0-1.0**；任何带版权的网络图片
  不得放入本库。需要真人示例图时，只能用自行拍摄并授权的照片。
- **真相源**：端上可执行逻辑以 street-snap 代码为准；本库是它的可移植副本与人审视图，
  每次姿势/构图批次同步，并在 JSON 内记录对应 app commit。
- **证据纪律**：阈值全部标注 experimental，真人有效性 `not-verified`（T90/T91 关卡）。
- **未来**：素材规模扩大后独立为新仓（暂定名 lwl-asset-library），本目录保留为初始版本。
