# 实验索引

当前推荐与统一命令见 [CURRENT.md](../CURRENT.md) 和 [README.md](../README.md)。以下资料供复查，不作为日常版本选择菜单。

| 阶段 | 状态与用途 | 资料 |
| --- | --- | --- |
| v2 群体关系关联 | 实验候选；冻结后 30 文件本地 79.4033→81.2450，尚未官网验收 | [首轮结果与运行命令](../experiments/v2_association/RESULTS_20261116.md)、[对照入口](../experiments/v2_association/README.md) |
| 原始简单基线 | 稳定参考及协议排查 | [历史开发说明](LEGACY_GUIDE.md) |
| Track-MT3 模拟复现与 B02 适配 | 历史研究，当前入口不依赖上游模型 | [模拟阶段](../experiments/track_mt3_stage1/README.md)、[B02 适配](../experiments/track_mt3_stage2/README.md) |
| 点级分类器与扩大训练 | 当前轻量推理权重的来源 | [扩大训练](../experiments/point_scorer_expanded/README.md) |
| 稀疏关联与输出门限 | 4000／8000 步和阈值对照 | [关联研究](../experiments/association/README.md) |
| ID 诊断及状态管理改动 | 解释瓶颈与策略取舍 | [诊断](../experiments/association/DIAGNOSIS.md)、[单项](../experiments/association/INTERVENTION_RESULTS.md)、[组合](../experiments/association/COMBINATION_RESULTS.md) |

原始统计、源码摘要和检查点继续保存在本地 `runs/`、`analysis/` 及云端。当前查看器统一从 `runs/current/index.html` 进入，历史逐圈数据保留，旧 HTML 入口清理后可按历史命令重新生成。
