# B02 第二阶段：Track-MT3 适配原型

本阶段将第一阶段模拟数据权重接入 B02 的训练 CSV、逐圈回放、结果表和查看器。它是实验模型，尚未取代默认基线，也尚未完成官方收发环境的部署验收。

## 数据与适配范围

- 固定使用 `splits/v1.json` 前 4 个训练场景：001、002、004、005；前 2 个开发场景：015、016。按已有划分顺序选择，不读取保留集。
- 模型输入顺序为归一化方位角、距离、俯仰角。这里不假定它们是真实 XYZ 坐标；第一版尚未利用 Amp/SNR。
- 同时保留最多 96 条航迹、96 个检测查询。模型使用最近 6 圈，保留预训练的 20 项时间嵌入并右对齐到第 19 项。
- 每圈最多 512 点，超过时按原始点号排序后等距抽取，绝不根据标签筛选；完整当前圈仍用于最终原始点匹配。初次 200 步流程试验用 192 点，因训练输入保留率偏低已提高到 512。
- 标签 0 是已知杂波；-1 点仍参与输入，但不组成对比学习正负配对。置信度损失忽略最近原始点为 -1 且处于匹配门内的未分配查询，已分配真值的查询始终监督。这是对集合预测损失的局部忽略近似，不代表官方标签定义有变化。
- 从模拟权重复用 Transformer、分类与大部分跟踪参数。输入层、输出头和查询传播输入扩展第三维；沿用二维 Fourier 布局并加入幅度 0.1 的俯仰角编码。该设计保留可复用形状，迁移收益仍需后续对照确认。
- 一对一指派把模型预测匹配到本圈原始点，门限对应方位角 .025、距离 .020、俯仰角 .065。无合法原始点的预测不输出；保留所有原始字段，模型 ID 加 1 转为正批号。
- 所有开发圈都评分，包括首圈和空圈；每个文件重新初始化状态，不使用未来圈，不将 GT 提供给推理。

## 环境与源码隔离

复用第一阶段的 PyTorch、NumPy、SciPy、PyYAML 环境。脚本只支持已固定的 CNSF 提交 `22c3a19c15df6c560eaadc74462b9b632ede2602`，在新目录中创建三维副本。原上游、模拟训练和第一阶段权重不修改。副本保留原 LICENSE，并写出精确修改补丁 `b02-stage2.patch`。

```bash
python experiments/track_mt3_stage2/prepare_upstream.py /path/to/CNSF /path/to/CNSF-b02-stage2
python experiments/track_mt3_stage2/check.py --upstream /path/to/CNSF-b02-stage2 --project . --pretrained /path/to/best-validation.pt
```

`check.py` 检查三维前后向、俯仰角梯度、未知标签损失屏蔽、标签隔离、空圈、原始点与一对一匹配。默认基线仍无需安装 PyTorch。

## 有限预算训练

```bash
python experiments/track_mt3_stage2/run.py \
  --upstream /path/to/CNSF-b02-stage2 --project . \
  --data /path/to/b02-training.zip --pretrained /path/to/best-validation.pt \
  --out runs/b02-stage2-pilot --steps 1000 --seconds 1200
```

每步随机从四个训练场景选一个连续 4 圈片段，batch=1，Adam 学习率 5e-5，梯度裁剪 10；保留上游多层辅助损失、跨帧指派和对比损失。定位损失与位置匹配代价权重改为 20，以适应从 20 单位场景跨度到归一化跨度 1 的变化。25% 片段起点从文件前 30 圈抽取。当前单 GPU 按概率 .5 使用真值辅助；验证始终完全自主传播。最初 pilot01/02 使用了上游前两次传播强制辅助的逻辑，后续诊断已修正为只在初始化分布式训练时强制，详见 DIAGNOSIS.md。这是适配试验，不是原论文配方复现。

达到步数或训练时间上限后保存权重，验证保存与加载预测一致，然后完整运行两个开发场景的传统基线和模型。时间预算仅限制训练循环，后续回放、计分及导出另计。脚本退出不关闭容器。

结果目录包含：`manifest.json`、训练输入保留率 `data-profile.json`、迁移明细 `transfer.json`、`training.jsonl`、`final.pt`、`report.json`，以及 `baseline/`、`candidate/` 的逐圈结果和计分表。第二轮起同时保存本次适配源码及上游修改补丁。不要将权重或赛事原始数据提交 GitHub。

## 统一回放入口

用第二阶段训练产生的权重，不能直接传入第一阶段二维权重。适配代码版本、点数上限和权重 manifest 必须一致。

```bash
python -m b02 replay /path/to/b02-training.zip --scene train_015 \
  --tracker track-mt3 --checkpoint /path/to/stage2/final.pt \
  --upstream /path/to/CNSF-b02-stage2 --device cuda --out runs/mt3-dev
python -m b02 scorecard runs/mt3-dev --baseline runs/dev-baseline
python -m b02 visualize /path/to/b02-training.zip --run runs/mt3-dev \
  --scene train_015 --baseline runs/dev-baseline --out runs/mt3-dev/train_015.html
```

默认 `--tracker baseline` 的原命令不变。模型环境兼容的队友可使用 CPU 回放，但会明显较慢；无需在个人电脑训练模型。官方 TCP 收发模型部署、完整 30 场景开发集、极端密集圈的资源检查仍属于后续验收。

## 如何解读结果

本地总分包含未获官方完全确认的假设，且低召回时某些子分可能偏高，必须同时看点命中率、虚警、起批、ID 稳定性，不能只看总分。只有独立开发数据的结果有竞争力，才考虑扩大训练。不能把“小样本拟合成功”“损失降低”或“输出格式合规”当作已经超过基线。

后续已完成单 GPU 真值辅助诊断（`DIAGNOSIS.md`）和两随机种子的等预算初始化对照（`INITIALIZATION_ABLATION.md`）。当前迁移方法在短预算下未体现模拟预训练优势；两种初始化都不可替代传统基线。对照耗时约 11 分钟，已停止，无后台续训。

随后完成 1/2/4/8 片段的训练覆盖对照（`COVERAGE_ABLATION.md`），总计 2400 次更新、约 12 分 14 秒。补足每片段训练次数能改善已见片段拟合，但 8 片段最终已见命中率 26.86%、固定跨文件片段 2.60%，仍不可用。该实验已退出，结果与源码快照已保存；没有继续长训练。

原始点打分方案的后续实验位于 `../point_scorer/README.md`：小网络直接分类原始点，再结合既有基线关联，在两个完整开发文件上显著减少杂波。时序输出过滤的两种子本地估算总分平均 68.05（基线 65.87），但仍有召回和连续跟踪损失。该结果是方案比较，不是对 Track-MT3 回归头的单因素替换验证。
