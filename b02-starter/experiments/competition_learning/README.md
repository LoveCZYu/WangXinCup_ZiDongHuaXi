# 跨 ID 竞争与拒绝学习

本轮在固定原关联候选上采集实际运行数据，比较二元身份目标与联合目标。它是实验目录，不修改 `profiles/current.json`、正式提交入口或冻结权重。

## 数据与标签边界

`protocol.json` 固定 24 个训练场景和 8 个内部验证场景，均来自已有 fit64；二者按场景不重叠。底层主关联器训练过 fit64，旧身份头也曾用内部验证池选检查点，所以这里是新头的内部开发验证，不能称整套系统的独立盲测。原八文件校准、新 16 工作验证、确认集和正式 29 文件本轮均不读取。

采集采用原候选的真实运行历史。`Tape` 只复制当前因果特征并原样返回旧评分；标签由外部 `label_call` 根据本圈以前已发布的已知真目标历史生成。至少三次、纯度至少 80% 才建立身份锚；已知杂点计入纯度分母，−1 不当负例。纯杂点拥有者和身份不确定的历史未被强贴真目标身份标签，这也使部署到全部成熟 ID 时存在分布差异，需要整场检查。

标签保留单候选和“当前没有本身份正确点”的行。若没有已知正确候选、又存在未知候选，该行不用于教 null。列偏好在正确候选中优先历史主 ID，否则保留全部同身份候选；二元标签仍表示是否同身份。数据使用阶段依赖的实际候选，不能学习恢复未生成的边。

## 公平对照

两种模型均从既有 93 维身份模型初始化，新增三项发布历史元数据以零输入权重起步，初始函数保持不变。每种目标运行两个固定种子，各 1600 步，固定检查点和选取规则。

- `binary`：在相同采样批次上，只训练同身份二元损失；null 层不更新。
- `joint`：相同二元损失，再加 0.5 行内“点或 null”的集合交叉熵，以及 0.5 列内“ID 或暂不分配”的集合交叉熵。

只用训练数据拟合新增归一化。每个模型按预定的内部验证目标选检查点，不按整场分数或种子选赢家。内部验证同时检查固定接受数量下的误配、正确行召回和错误接受。原协议任一主条件不通过就不能使用 `replay.py` 扩大试验。

## 入口

在 `b02-starter` 内使用已有 PyTorch／NumPy／SciPy 环境。替换大写路径；输出目录要求不存在，避免混入历史文件。

```text
python experiments/competition_learning/data.py --data FIT64_CAL8.zip --cache CACHE64 --prior MEMORY.pt --affinity AFFINITY.pt --head CONTINUATION.pt --fusion FUSION.json --freeze experiments/robust_validation/freeze.json --protocol experiments/competition_learning/protocol.json --out runs/competition-learning/part-0
python experiments/competition_learning/train.py --samples SAMPLES --protocol experiments/competition_learning/protocol.json --prior MEMORY.pt --affinity AFFINITY.pt --out runs/competition-learning/models
```

采集可用 `--shards 4 --index I` 分片；合并时只链接／复制固定 32 个样本及其元数据。训练要求 CUDA，预算在协议中固定；推理使用 CPU。新模型只存 tensor state 和来源摘要，模型与原始样本不进 Git。

`features.py` 是不读取真值的共享输入，`data.py` 负责外部监督，`model.py`／`samples.py`／`metrics.py`／`train.py` 定义训练和验证。`runtime.py` 保留两种接入对照：仅列内身份代价，以及将成熟 ID 的实边／null 代价替换为校准 logits。原几何候选、点模型和输出门保持固定。

## 原门槛失败后的额外诊断

本轮联合目标没有通过原离线门槛；`replay.py` 会据此拒绝执行。该失败不能因后续诊断而改写。

`reject_calibration.py` 另存运行前诊断协议，只在训练 24 文件拟合一个接受偏移，保持二元对照的正确行数量，然后一次性检查原内部验证 8 文件。它不允许覆盖原门槛，也不授权版本升级。为避免未知候选被屏蔽后产生乐观判断，此诊断排除包含任何未知候选的行。

`pilot.py` 是单独注册的、只研究自由运行状态变化的整场诊断：原候选、二元绝对评分、联合原工作点、联合训练拟合工作点。固定默认种子，不选较好种子，不在验证场景调阈值。该 pilot 不代表原联合学习胜出，不具备升级版本的权限；结果必须完整保留，包括退化。

训练数据快照、四个训练结果、接受偏移和 pilot 各有来源摘要。所有标签只进入外部监督／评价，模型输入不包含真值 ID、原始数值 ID 或未来观测。

随后 `column_transfer.py` 在同批八文件上保留原行内／null 规则，只分别使用新旧身份头的列内证据，协议为 `column-transfer.json`。新联合头这个预定主比较未达到幅度门槛；新二元头是对照中出现的探索性信号，没有被事后改写为主比较赢家。

`replicate.py` 根据另存的 `replication.json` 复核第二个已训练二元种子，既不重训也不换参数。改善方向一致，但第二种子的零碎度增幅未达预设复核门槛，不能宣称所有条件通过或选择较好的种子覆盖默认结果。

完整汇总入口为 `compare_pilot.py`、`compare_transfer.py` 和 `compare_replication.py`，它们分别读取 `pilot-*`、`transfer-*`、`replica-*` 目录。`report.py` 生成结果 HTML，原二维查看器从预选 train_152 的实际 trace 导出。所有结果与失败边界见 `RESULTS_20261117.md`。
