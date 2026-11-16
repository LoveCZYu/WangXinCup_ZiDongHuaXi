# 实验索引

当前推荐与统一命令见 [CURRENT.md](../CURRENT.md) 和 [README.md](../README.md)。以下资料供复查，不作为日常版本选择菜单。

| 阶段 | 状态与用途 | 资料 |
| --- | --- | --- |
| v2 联合匹配实验 | 零碎度 62.8056→63.5741／63.6658，两组学习版均保留为对照 | [八文件诊断和实验数据](../experiments/joint_matching/RESULTS_20261116.json) |
| v3 可靠航迹记忆 | 30 文件回归估分 81.2450→81.7080，16 文件确认估分 81.9094→82.3299 | [汇总数据](../experiments/v3_tracking/RESULTS_20261116.json)、[数据划分](../splits/v3_tracking.json) |
| v2 群体关系关联 | 实验候选；冻结后 30 文件本地 79.4033→81.2450，尚未官网验收 | [首轮结果与运行命令](../experiments/v2_association/RESULTS_20261116.md)、[对照入口](../experiments/v2_association/README.md) |
| 原始简单基线 | 稳定参考及协议排查 | [历史开发说明](LEGACY_GUIDE.md) |
| Track-MT3 模拟复现与 B02 适配 | 历史研究，当前入口不依赖上游模型 | [模拟阶段](../experiments/track_mt3_stage1/README.md)、[B02 适配](../experiments/track_mt3_stage2/README.md) |
| 点级分类器与扩大训练 | 当前轻量推理权重的来源 | [扩大训练](../experiments/point_scorer_expanded/README.md) |
| 稀疏关联与输出门限 | 4000／8000 步和阈值对照 | [关联研究](../experiments/association/README.md) |
| ID 诊断及状态管理改动 | 解释瓶颈与策略取舍 | [诊断](../experiments/association/DIAGNOSIS.md)、[单项](../experiments/association/INTERVENTION_RESULTS.md)、[组合](../experiments/association/COMBINATION_RESULTS.md) |

原始统计、源码摘要和检查点继续保存在本地 `runs/`、`analysis/` 及云端。当前查看器统一从 `runs/current/index.html` 进入，历史逐圈数据保留，旧 HTML 入口清理后可按历史命令重新生成。

## v2 联合匹配实验

以长期记忆版为对照，八个校准场景得到 1211 个额外 ID 片段，其中 497 个只含一到三个命中点。445 个片段首次出现时，已有输出最多的历史 ID 仍有该点作为候选，其中 387 个原 ID 当圈未获得匹配。另有 337 个片段开始时原 ID 已离开活跃集合。其余情形包括门外观测和初次输出分配给后来的短 ID。

用最终主 ID 回溯修正候选内片段并保留原输出点，零碎度估分从 62.8056 升到 74.9573。再纳入门外近处的片段，估分为 77.2766。这个追溯对照依赖真值和完整观测，并跳过同圈 ID 冲突。

联合匹配在同一已确认航迹分配阶段中比较最多三条航迹的整组方案。每组保留已选强点，可以把点转交给当圈未匹配的轨迹。规则对照使用历史输出和运动证据，学习版另用 24 个拟合场景训练，8 个拟合池内场景选检查点。三种方案均完整回放八个校准场景。种子 20261116 的总分为 83.0859，零碎度为 63.5741，点命中率为 94.4328%，连续跟踪为 82.1467。种子 20261117 对应为 83.0938、63.6658、94.4182% 和 82.1372。两个种子均低于零碎度增加 3 分的版本目标。

实验入口为 `experiments/joint_matching/run.py`，`--method audit` 复现归因和回溯诊断，`--method continuity_rule` 回放规则，`--method learned` 回放训练权重。`fit.py` 提供样本生成和训练，`merge.py` 合并各分片并执行同口径计分。回放使用缓存的冻结点分数。已保存的实验数据仅包含汇总指标和分组统计。

## v3 航迹连续性实验

在八个校准文件上复现 v2 的逐航迹统计后，发现 2865 次变号中有 1160 次属于旧轨迹和其他轨迹争同一点，999 次发生在旧轨迹匹配到另一条真实航迹时。训练扩大到 64 个场景，并采集实际跟踪过程中形成的历史。长期记忆版保存最近 16 次可靠关联及首个可靠关联，可靠性由配对分差、点置信度和运动残差共同判断。

长期记忆版两个种子的八文件估分为 82.8768 和 82.9504，v2 对照为 82.3127。独立重分配已选点的三种 ID 策略降低了零碎度子分。额外的跨航迹归属模型相对同一个长期记忆底座分别增加 0.3410 和 0.0731 分。采用门槛设为两个种子均增加至少 0.25 分，据此固定长期记忆版的种子 20261116、1000 步权重。

固定后，30 个开发文件的零碎度子分从 60.6471 升至 61.6283，连续跟踪从 79.9876 升至 81.0909。16 个新增确认文件按预先固定的哈希顺序从未参与关联模型训练和选型的训练池中选取，考察同一固定候选的泛化表现。正式保留集仍供之后验证。

16 文件确认在本机完成。零碎度子分从 62.9593 升至 63.5409，连续跟踪从 80.4151 升至 81.5530，换号从 5606 降至 5518。两项子分均有改善，幅度低于原定的各提高 3 分。官网现有提交为 v1 的 75.24 分。

在 `b02-starter` 目录可用以下命令回放导出的 v3 权重。参数 `YOUR_DATA.zip` 改为本地训练包路径。

```bash
python experiments/v3_tracking/candidate.py --profile profiles/v3-tracking-candidate.json --data YOUR_DATA.zip --scene train_015 --out runs/v3-preview
python -m b02 visualize YOUR_DATA.zip --run runs/v3-preview --scene train_015 --out runs/v3-preview/train_015.html
```
