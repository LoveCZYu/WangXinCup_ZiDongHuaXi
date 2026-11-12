# B02 目标检测跟踪开发起点

团队共用一个逐圈算法入口。本地回放读取官方训练 CSV，官方联调经原版收发程序读取 DAT，两条路径都调用 `b02/algorithm.py` 的 `process_circle(points, state)`。默认跟踪器在 CPU 上运行。

## 点打分与航迹关联

从团队获取 4000 步或 8000 步检查点，在已安装 PyTorch、NumPy、SciPy 的 Python 环境中，使用下面的命令回放指定场景。默认使用 CPU，可用 `--device cuda` 切换到 CUDA。默认输出阈值为 0.50／0.15，可用 `--enter` 和 `--maintain` 指定对照值。

```text
python experiments/association/replay_checkpoint.py --data "你的训练数据.zip" --checkpoint "团队检查点.pt" --scene train_085 --out runs/point-replay
python -m b02 visualize "你的训练数据.zip" --run runs/point-replay --scene train_085 --out runs/point-replay/train_085.html
```

[关联与阈值实验](experiments/association/README.md)记录了规则、校准结果和复现命令。[扩大训练](experiments/point_scorer_expanded/README.md)说明训练数据子集和权重的生成方式。

## 队友首次运行

获取团队仓库或解压团队代码包后，进入 `b02-starter` 目录。使用 Python 3.10 或以上，建议 3.12。Windows 可使用 `python` 或 `py -3`，macOS 和 Linux 如果使用 `python3`，替换下方命令开头的 `python`。基线回放使用 Python 标准库。

```text
python --version
python -m unittest discover -s tests -v
python -m b02 replay examples/smoke.csv --out runs/smoke
```

合成样例应报告 **5 圈、4 个输出点、8 个真值点中命中 4 个、已知杂波输出 0 个**。两个目标在第三次命中时起批，随后各自跨过一个空圈并保持批号。`runs/smoke/summary.json` 保存诊断，`runs/smoke/smoke.jsonl` 保存逐圈输出。重复运行时使用新的 `--out` 目录。

## 回放真实训练数据

从赛事渠道获取训练 ZIP。把命令中的路径换成自己的文件路径，支持中文和空格。也可传入一个解压的 CSV 或包含 CSV 的目录。

```text
python -m b02 replay "你的训练数据.zip" --scene train_015 --out runs/train-015
python -m b02 replay "你的训练数据.zip" --split dev --out runs/dev-baseline
```

多次使用 `--scene` 可挑选多个文件。`splits/v1.json` 固定了 140 个训练场景、30 个开发场景和 30 个锁定场景。清单按文件随机划分，未对同源场景做分组。用 `train` 开发和拟合，用 `dev` 做日常比较，`holdout` 留给里程碑验证。CSV 按原有圈号回放，缺号以空圈推进状态，保留原始时间与坐标，浮点字段按协议转换为 float32。CSV 回放以有记录的最后一圈结束。

`summary.json` 包含参数、每场景的目标点命中、已知杂波输出、未知标签输出、航迹换号和逐圈耗时。命中按原始点对应真值标签统计。每个真值航迹记录首次输出延迟、各输出 ID 的匹配次数和单一 ID 最多匹配多少点。多个真值也可能关联到同一个输出 ID，可以在逐航迹结果中检查。以上是本地诊断，不是官方分数。逐圈文件的耗时会随设备变化，航迹输出可逐条比较。

每次回放还会生成 `scorecard.csv` 和 `scorecard.json`，包含整批及各场景的指标。CSV 可用 Excel 打开。已有回放可直接从 `summary.json` 补生成结果表，也可指定另一个回放目录作为基线。

```text
python -m b02 scorecard runs/dev-baseline
python -m b02 scorecard runs/dev-candidate --baseline runs/dev-baseline
```

对照时按候选版实际跑过的场景汇总两边，检查圈数、点数和真值航迹覆盖是否相同。比例的变化用百分点，其他变化用原单位。首次输出延迟从首个真值点算起，始终未命中的航迹单列。主输出 ID 覆盖率取每条真值航迹命中最多的那个输出 ID，允许中间漏圈。表中列出赛题权重，供对照各项时参考。

根据组委会答复，正式评测按原始点序号匹配，评测数据没有 `-1` 点，同一个输出 ID 按一条航迹处理。连续跟踪公式中的 `max` 应为 `min`。本地结果表保留原有诊断，另列两项估计子分：

- 虚警估计为 `100 × max(0, 1 - 已知杂波输出数 / 真值点数)`。训练集 `-1` 输出单列，分数用于同一批训练场景上的比较。
- 连续跟踪估计对每条真值取命中最多的输出 ID，按 `100 × min(1, 该 ID 的匹配点数 / (真值点数 - 2))` 求分，再按航迹平均。按现有文档的字面口径，前两点从分母扣除。同 ID 的命中跨漏圈累计，真值点不足三点的航迹暂不计入该估计。

现在还提供五项子分的**本地估算总分**，使用官方权重 `0.3 / 0.1 / 0.2 / 0.2 / 0.2`。其中掌握率采用逐航迹平均命中率；零碎度暂以至少命中一点为掌握成功，对成功航迹的 `100 / 输出 ID 数` 求平均，无成功航迹为 0；及时性暂按每圈 5 秒、首圈后第三圈为起点、首次匹配点为输出起点，延迟截断至 0–50 秒，未命中为 0，少于三点的航迹不计入该子分。这些假设不代表官方已确认。

总分和子分同时保存到 `scorecard.json`、`scorecard.csv`；`results.csv` 每行一个场景，便于比较，`scorecard-policy.txt` 保存完整计算口径。整批分数合并航迹和点数后计算，不是各场景总分的简单平均；任一子分无法定义则总分留空。组委会不提供本地评测脚本，正式分数以官网为准。

## 可视化排查航迹

用下面的命令从合成样例生成查看器。若还没做过合成样例回放，请先运行「队友首次运行」里的回放命令。

```text
python -m b02 visualize examples/smoke.csv --run runs/smoke --scene smoke --out runs/visualizations/smoke.html
```

真实场景使用对应的训练数据和回放目录。下面假设已经回放 `train_015` 到 `runs/train-015`。

```text
python -m b02 visualize "你的训练数据.zip" --run runs/train-015 --scene train_015 --out runs/visualizations/train_015.html
```

双击生成的 HTML 即可离线打开，页面包含所选场景的点迹和回放结果。重新导出时用新的 `--out` 路径，或加 `--replace` 更新原入口。查看器会核对圈数、圈号和输出点是否对应原始数据。

展开页面上方的「结果评分表」，切换「当前场景」和「整批回放」。导出时可在 `visualize` 命令后加 `--baseline runs/dev-baseline`，在网页中并排显示对照版本、当前版本和变化量。再加 `--previous runs/dev-previous` 则只列「基线、上一版、这一版」三版，逐场景列表也并排显示三版总分。三版对照保存为当前回放目录中的 `viewer-comparison.csv` 和 `viewer-comparison.json`，保留原实验结果表；具体示例见 `experiments/association/README.md`。

- 从右侧「失败定位」选一个杂波输出多或真值漏点多的圈，也可以挑选输出 ID 分散的航迹。
- 用滑条、圈号输入框或左右键翻圈。切换方位角、俯仰角和距离来观察不同投影。
- 悬停查看点号、时间、真值与输出 ID。点击真值点或使用下方下拉框选择航迹。
- 时间线上白线是当前圈，彩色标记对应输出 ID，橙色标记表示有真值点但未输出。点击时间线或起批、变号事件即可跳圈。
- 评分表顶部显示当前统计范围的本地估算总分；展开逐场景结果列表可对照全部场景的总分与子分。计算假设也可展开查看。每次导出查看器会同步更新运行目录的结果表。

查看器显示的是点级回放结果。关联代价、预测位置、门限及删除原因需要算法侧提供额外日志。

## 联调官方收发

Windows 上用随包 `serve.exe` 和原版 Python 收发代码。Linux 上原版启动器需要 Wine、wineserver、wineboot 和 Xvfb。macOS 可以运行本地 CSV 回放，官方收发请使用符合要求的 Windows 或 Linux 环境。

将代码样例 ZIP 放在任意本地路径，使用下列命令。`prepare-official` 从 `participant_demo_v2` 包提取原版脚本、收发程序和样例数据到本地 `.official`，只需运行一次。

```text
python -m b02 prepare-official "你的官方代码样例.zip"
python -m b02 official --original --out runs/official-original
python -m b02 official --out runs/official-baseline
```

`--original` 运行官方首点示例，用于排查环境。默认模式调用团队跟踪器。联调成功时，`run_summary.json` 的 `status` 为 `success`，输出目录包含原版收发软件写出的航迹 DAT 和逐圈耗时 DAT。用 `--port 19902` 可以更换本地端口，用 `--input-dir` 可以指定另一个合规的 DAT 输入目录。运行失败时先查看命令提示和 `runs/对应运行目录/logs/server.log`。需要检查所有选项时运行 `python -m b02 official --help`。

默认联调会处理全部 5 个样例。首次想缩短验证时间，可以把 `.official/data/inputs/train_005.dat` 复制到一个单独的目录，再用 `--input-dir` 指向该目录。此包应完整处理 719 圈。

队内共享时发代码和这份说明。`.official`、原始数据和运行产物通过 `.gitignore` 排除，每位队友使用自己从赛事渠道获取的材料。

## 改算法和比较结果

`b02/algorithm.py` 默认采用归一化空间内的匀速预测、门控和贪心一对一匹配。优先匹配已确认航迹，连续三次命中起批，短时失配保留原 ID。每圈以收到的帧计时，方位角按普通坐标处理。候选航迹数和每条航迹的候选边数有限制。这些都是首周基线的工程选择，待实际失败分析后再调整。

算法输入是本圈点迹列表，字段为 `Time, Azimuth, Elevation, Distance, Amp, SNR, CircleID, PntID`。`state` 由回放和官方客户端在每个新数据包中创建，包内跨圈保留。每个返回点使用本圈原始时间、坐标和点号，附上正整数 `BatchID`。无结果时返回空列表。预测坐标用于内部关联。

可以用 JSON 覆盖默认参数，回放和官方联调都接受 `--config`。例如将下面内容保存到 `config.json`，用 `--config config.json` 比较多保留一圈的影响。

```json
{"max_missed": 4}
```

A 可以从 `b02/replay.py` 的诊断和场景清单继续做失败分析。B 可以用同一入口替换关联或跟踪逻辑。C 可以用原版联调命令复查依赖与收发，再记录候选版本的运行结果。每个实验保留代码版本、参数、场景清单和输出目录。

## 开源模型阶段 1 实验

`experiments/track_mt3_stage1/` 提供 CNSF 仓库中 Track-MT3 的云端短训练验收脚本和复现说明。已在 RTX 3090 上通过相关 61 项原仓库测试，以及训练、检查点恢复、继续训练和模拟数据推理验证。400 步试运行在原阈值下仍未输出航迹，尚未证明模型效果；该实验没有接入 B02 数据，也没有替换本项目的可提交基线。

后续已完成模拟数据 10000 步训练，并开始 `experiments/track_mt3_stage2/` 的 B02 适配。该目录提供隔离的三维模型扩展、训练与独立开发集回放、原始点关联及复现说明。可通过 `replay --tracker track-mt3 --checkpoint ... --upstream ...` 使用第二阶段权重；默认基线不变。请以该实验目录的结果说明判断当前效果，不要直接将实验模型当作可提交的优化版。
