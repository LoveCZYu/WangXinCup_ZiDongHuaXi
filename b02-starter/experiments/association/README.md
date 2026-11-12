# 航迹关联对照

在扩大训练的两个 4000 步权重上复用已冻结的输出规则。每圈采用原基线的运动预测、距离门限、起批和航迹更新。将已确认航迹和待确认航迹各自的逐边贪心匹配改为稀疏一对一分配，仍优先处理已确认航迹。

两种代价分别为门内归一化距离平方，以及该距离加上 `0.50 × (1 − 点分数)`。每条航迹最多保留代价最低的 12 个候选。允许航迹暂时无匹配，其代价为 1.15。点分数来自既有时序分类器，标签只进入回放计分。

在原训练划分中的 8 个内部校准文件上测试两种代价与两个种子。优先选择同时提高命中率和连续子分、已知杂点不超过现有方案 110% 的方案，其中本地估算总分较高者入选。如果无方案满足全部条件，则在减杂约束内选总分较高者。若两组都超出杂点约束，选择杂点较少的一组继续验证。选择冻结后，只将入选方案的两个种子在全部 30 个开发文件上回放。原始基线与现有过滤方案复用相同数据的已保存统计，数据包 SHA-256 和模型特征源码已在运行入口核对。

预选的查看器场景是 015 和 123。015 方便与之前的回放对照，123 是上一轮相对原基线命中率损失最大的开发场景。30 个封存测试文件保留在官方数据包中。

在有 CUDA、PyTorch 和 SciPy 的环境中，从项目根目录运行。参考目录需要包含扩大训练的权重、冻结规则和校准及开发汇总。

```bash
timeout --signal=TERM --kill-after=10s 1200s python experiments/association/run.py \
  --project . --data /path/to/point-expanded-data.zip \
  --reference runs/b02-point-expanded01 --out runs/association-new
```

外层时限为 1200 秒，脚本软截止为 1140 秒。每个完整场景的摘要会在结束后写出。开发集未完整结束时，报告会标记为未完成。

## 完整结果

内部校准中，单用归一化距离的方案达到 94.17% 命中率、70.91 连续子分，平均杂点 11874。加入点置信度后分别为 94.38%、72.76 和 11816，满足预设条件并获选。其余开发集统计来自冻结后的完整回放。

| 30 文件开发集 | 点命中率 | 已知杂点 | 连续跟踪子分 | 本地估算总分 |
| --- | ---: | ---: | ---: | ---: |
| 原基线 | 95.13% | 938650 | 60.70 | 66.67 |
| 原关联与现有输出过滤 | 93.79% | 47608.5 | 59.88 | 75.18 |
| 新关联与同一输出规则 | 94.19% | 48004 | 72.09 | 78.75 |

两种子平均命中率提高 0.40 个百分点、连续子分提高 12.21 分、本地总分提高 3.57 分。平均杂点比上一版增加 0.83%，比原基线减少 94.89%。以两种子均值统计，30 个开发文件的点命中率、连续子分和本地总分均较上一版提高。命中率与原基线还相差 0.94 个百分点。

两个种子分别达到 94.31%／94.08% 命中率、72.04／72.15 连续子分及 78.66／78.83 本地估算总分。云端完整计算耗时 701.47 秒（11 分 41 秒），进程正常退出。复用了既有权重，训练步数为每个种子 4000 步。

本地报告在 `runs/b02-association01/report.json`，校准及开发子目录各有逐航迹摘要和计分表。该轮最初简单基线、点级过滤版、改进航迹关联版在同一批 30 个开发场景、第一个预设种子 20261112 的本地估算总分分别为 66.67、75.13、78.66。历史回放统计保留，当前 015 和 123 查看器已更新为下文的 8000 步完整开发集对照。

从 `b02-starter` 目录重新生成（将 `train_015` 换成 `train_123` 可生成另一入口）：

```text
python -m b02 visualize ../analysis/point-expanded-data.zip --scene train_015 --run runs/b02-association01/dev/global_score_20261112 --baseline runs/b02-point-expanded01/dev/baseline --previous runs/b02-point-expanded01/dev/expanded_20261112__history_05_015 --out runs/b02-association01/train_015-association.html --replace
```

三版对照另存于当前回放目录的 `viewer-comparison.json` 和 `viewer-comparison.csv`，原实验结果表保留。旧 HTML 入口已清理，历史回放数据保留供复查。

## 4000 与 8000 步内部校准

沿用原 manifest 记录的 32 个训练文件、256 个片段和 2048 个监督帧，复用两个种子、每批 4096 点、Adam 和原学习率。旧权重缺少 Adam 和随机数状态，所以按原训练顺序从头复现，核对第 4000 步的参数后在同一训练轨迹保存第 8000 步。新检查点同时保存模型、优化器和随机数状态。

两种步数均用同一分配代价、点分数权重 0.50、同一输出门限和三圈历史规则，从头回放同一批 8 个内部校准文件。回放第 4000 步时逐场景核对既有统计。开发集与 30 个封存测试文件继续留作后续验证。

```bash
timeout --signal=TERM --kill-after=10s 480s python experiments/association/compare_steps.py \
  --project . --data /path/to/point-expanded-data.zip \
  --training-run runs/b02-point-expanded01 --association-run runs/b02-association01 \
  --out runs/association-steps-new
```

两个种子的 4000 步参数均与旧权重逐项相等，校准逐场景统计也与前次结果一致。以下为两个种子的均值。

| 步数 | 目标点命中率 | 已知杂点 | 连续跟踪子分 | 本地估算总分 |
| --- | ---: | ---: | ---: | ---: |
| 4000 | 94.384% | 11816 | 72.762 | 79.762 |
| 8000 | 94.079% | 8655.5 | 72.774 | 79.866 |

在固定输出规则下，8000 步将杂点减少 26.75%，相对原基线减少 96.22%。两个种子的命中率分别下降 0.588 和 0.022 个百分点。连续子分一个种子下降 0.249，另一个提高 0.272，均值增加 0.012。两个种子的训练损失均降低，校准结果呈现减杂与命中率之间的取舍。

整轮用时约 309 秒。四份检查点、训练日志、逐航迹摘要和报告保存在 `runs/b02-association-steps01/`。全部 32 次场景回放限于内部校准，封存测试文件继续保留。

## 8000 步输出阈值校准

在相同的 8 个内部校准场景，使用冻结的 8000 步权重、点分数关联和三圈历史。先按比例降低起始及维持阈值，再分别调整两项。每个种子的杂点均须不超过同种子的 4000 步对照。每个场景共享一次逐圈关联，候选仅改变对已关联输出点的取舍。

| 方案 | 起始／维持阈值 | 目标点命中率 | 已知杂点 | 连续跟踪子分 | 本地估算总分 |
| --- | --- | ---: | ---: | ---: | ---: |
| 4000 步对照 | 0.50／0.15 | 94.384% | 11816 | 72.762 | 79.762 |
| 8000 步原阈值 | 0.50／0.15 | 94.079% | 8655.5 | 72.774 | 79.866 |
| 8000 步选定阈值 | 0.50／0.05 | 94.429% | 9685.5 | 73.086 | 79.981 |

选定阈值相对 4000 步，平均命中率提高 0.045 个百分点、连续子分提高 0.324，杂点减少约 18.0%。两个种子的杂点分别由 13710 降到 9462、由 9922 降到 9909。命中率分别变化 −0.203 和 +0.293 个百分点，连续子分分别提高 0.107 和 0.540。0.35／0.105 在均值上也通过，但第二个种子的杂点达到 11373。阈值是在这 8 个文件上选出的，改善幅度适用内部校准结果。

两轮云端合计用时 312.07 秒。原始报告及计分表保存在 `runs/b02-association-threshold01/` 和 `runs/b02-association-threshold02/`。后者的 `train_085-threshold.html` 回放第一个预设种子的调阈值结果。三列依次为原始简单跟踪基线、4000 步对照、8000 步调阈值版，按同一批 8 个内部校准文件计算的总分依次为 67.94、79.76、79.93。8000 步原阈值结果见上表及校准报告。查看器使用真实 CPU 全链路回放，逐航迹统计与 GPU 校准结果一致。

```bash
timeout --signal=TERM --kill-after=10s 360s python experiments/association/sweep_thresholds.py \
  --project . --data /path/to/point-expanded-data.zip \
  --steps-run runs/b02-association-steps01 --out runs/threshold-proportional
timeout --signal=TERM --kill-after=10s 200s python experiments/association/sweep_thresholds.py \
  --project . --data /path/to/point-expanded-data.zip \
  --steps-run runs/b02-association-steps01 --out runs/threshold-separate --grid separate --seconds 190
```

本地复查某个场景可使用 `replay_checkpoint.py`。8000 步调阈值版本传入 `--enter 0.50 --maintain 0.05`，4000 步对照使用默认阈值。

```bash
python -m b02 visualize /path/to/point-expanded-data.zip --scene train_085 \
  --run runs/b02-association-threshold02/20261112-step8000-maintain005 \
  --baseline runs/b02-point-expanded01/calibration/baseline \
  --previous runs/b02-association-steps01/20261112-step4000 \
  --out runs/b02-association-threshold02/train_085-threshold.html --replace
```

## 冻结阈值后的完整开发集验证

8000 步固定起始阈值 0.50、维持阈值 0.05、历史三圈和关联点分数权重 0.50，在原划分的全部 30 个开发文件上完成两个种子的 60 次回放。没有追加训练或开发集阈值搜索。原始基线和 4000 步对照复用同批文件的完整统计，先核对数据摘要、4000 步权重逐项相等，并分别重放 015 核对全部非耗时统计。

| 30 文件开发集，两种子均值 | 目标点命中率 | 已知杂点 | 连续跟踪子分 | 本地估算总分 |
| --- | ---: | ---: | ---: | ---: |
| 原始简单基线 | 95.135% | 938650 | 60.696 | 66.665 |
| 4000 步，0.50／0.15 | 94.195% | 48004 | 72.093 | 78.746 |
| 8000 步，0.50／0.05 | 94.199% | 38784.5 | 72.324 | 78.946 |

平均命中率基本持平，连续子分增加 0.231，平均杂点减少 19.21%。第一个种子的命中率下降 0.258 个百分点、杂点减少 18700；第二个种子的命中率提高 0.266 个百分点、杂点增加 261（约 0.65%）。因此在开发集上未满足两个种子各自杂点均不增加的校准条件。两个种子的连续子分和总分均提高。完整口径、逐种子和逐场景结论见 [DEV_RESULTS.md](DEV_RESULTS.md)。

```bash
timeout --signal=TERM --kill-after=10s 1200s python experiments/association/evaluate_dev.py \
  --project . --data /path/to/point-expanded-data.zip \
  --training-run runs/b02-point-expanded01 --steps-run runs/b02-association-steps01 \
  --association-run runs/b02-association01 --out runs/association-dev-new
```

脚本软截止为 1100 秒，本轮完整运行耗时 480.74 秒。实际输入仍是之前的数据包，封存测试文件不在包内，30 个封存测试文件保持未使用。每个场景结束保存摘要，全部完成才将报告标记为完成。

当前查看器为 `runs/b02-association-dev01/train_015-dev.html` 和 `train_123-dev.html`，回放种子 20261112 的真实 GPU 输出。三列依次为原始基线、4000 步、8000 步，完整 30 文件总分为 66.67、78.66、78.85。此处为单个固定种子，与上表两种子均值区分。原 015/123 HTML 入口已清理，历史统计和三版回放数据保留。

```bash
python -m b02 visualize /path/to/point-expanded-data.zip --scene train_015 \
  --run runs/b02-association-dev01/20261112-step8000-maintain005 \
  --baseline runs/b02-point-expanded01/dev/baseline \
  --previous runs/b02-association01/dev/global_score_20261112 \
  --out runs/b02-association-dev01/train_015-dev.html --replace
```

替换场景及输出文件名可生成 123 查看器。`report.json` 和 `comparison.csv` 保存整批两种子均值、逐种子结果及逐场景对照；各种子目录保留 `summary.json`、`scorecard.json/csv`、`results.csv`。查看器三版表另存于 `viewer-comparison.json/csv`，不覆盖实验对照表。Git 仓库只提交代码和结果说明，数据与检查点需沿用实验环境中的本地文件。
