# 跟踪架构借鉴调研（2026-11-17）

建议下一轮优先研究**边评分与拒配共同读取行、列竞争上下文**。本轮完整历史输入与候选感知拒配均未通过预设门槛；这些失败应保留，下面是待验证的结构假设，不代表已经涨分或复现论文。图片领域权重不能直接视为雷达点迹的有效预训练。

## 已核实的方法与适配判断

| 方法 | 来源事实 | B02 借鉴方案与限制（我们的推断） |
| --- | --- | --- |
| Deep Sets | 研究排列不变的集合函数及相应网络结构。[论文](https://arxiv.org/abs/1703.06114) | 共享边编码后做 mean/max 池化并加入候选数量，适合无序、变长候选集；这是受启发实现，并非定理的字面复现。单行池化不能自动解决多个 ID 抢同一点。 |
| MPNTrack | 在检测关联图上进行消息传递，联合利用节点与边信息求关联；原方法跨帧图推理，不是可直接部署的逐圈因果版。[论文](https://arxiv.org/abs/1912.07515)、[官方代码](https://github.com/dvl-tum/mot_neural_solver) | 用“历史轨迹—当前点”二部图，边同时接收本行其他候选和本列其他轨迹的摘要，再共同预测匹配与拒配。只访问历史和当前圈，不回改已输出结果；属于重新设计。 |
| SuperGlue | 将两组图像局部特征的对应与不可匹配点拒绝联合建模，使用注意力聚合和可微最优传输。[论文](https://arxiv.org/abs/1911.11763)、[官方代码](https://github.com/magicleap/SuperGluePretrainedNetwork) | 借鉴双侧上下文及显式未匹配槽。官方实现的未匹配参数是共享可学习标量，不能据此宣称它已验证候选感知拒配头。雷达需自建特征与训练目标。 |
| OC-SORT | 保留在线跟踪，以观测构造虚拟轨迹，在失配后重新关联时修正滤波误差。[论文](https://openaccess.thecvf.com/content/CVPR2023/html/Cao_Observation-Centric_SORT_Rethinking_SORT_for_Robust_Multi-Object_Tracking_CVPR_2023_paper.html)、[官方代码](https://github.com/noahcao/OC_SORT) | 可尝试可信观测方向与恢复后的内部状态重更新；恢复只用截至当前的观测，不补造比赛输出点。需重设三维空间、角度和噪声模型，错误恢复可能反而污染状态。 |
| MOTR | track query 逐帧传递更新，结合身份持续监督和时序聚合。[论文](https://arxiv.org/abs/2105.03247)、[官方代码](https://github.com/megvii-research/MOTR) | 以后可将每个 ID 的历史压成可学习记忆，再查询当前原始点。因果推理可行，但需序列训练及记忆更新机制，工作量明显大于小型关联头。 |

## 实施顺序与验收

1. **先做一层行列消息传递的联合关联头**：同一缓存、划分和双种子，与独立边评分对照；同时保留空匹配监督。它解决的是关系信息不足，不是简单增加训练步数。计算量主要随候选边数增加，可沿用稀疏物理门控。
2. **再单独试 OC-SORT 式观测更新**：不与新关联头同时改，区分“身份辨别失败”和“运动预测漂移”；无需先训练大型网络。
3. **最后考虑持久 query**：只有轻量方案的证据表明时序记忆仍是瓶颈，再承担完整序列训练成本。

先看同覆盖率下的错误身份、缺席候选误接与拒配，再看自由运行中的零碎度、连续时长、命中和总分；静态配对改善不等于整场改善。继续保留失败对照，封存数据不用于本轮选择。

## 代码许可核验

MPNTrack 与 OC-SORT 官方仓库为 MIT；MOTR 自有部分为 MIT，同时保留 DETR/Deformable DETR 的 Apache-2.0 声明。SuperGlue 官方代码有非商业内部研究及再分发限制，本建议只借论文思路，不复制其代码或权重进提交包。[MPNTrack 许可](https://github.com/dvl-tum/mot_neural_solver/blob/master/LICENSE)、[OC-SORT 许可](https://github.com/noahcao/OC_SORT/blob/master/LICENSE)、[MOTR 许可](https://github.com/megvii-research/MOTR/blob/main/LICENSE)、[SuperGlue 许可](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/LICENSE)。
