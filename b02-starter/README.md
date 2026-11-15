# B02 团队运行入口

> 当前代码包用于开发与联调。11.15 新版正式提交要求及当前缺项见 [SUBMISSION.md](SUBMISSION.md)，不要将开发 ZIP 直接作为已验收的最终提交包。

从这里运行当前点模型。权重、参数配置及本地回放／官方联调使用同一套加载逻辑。当前推荐、完整开发集结果与限制统一记录在 [CURRENT.md](CURRENT.md)。

## 队友首次运行

进入 `b02-starter` 目录，使用 Python 3.10 或以上的独立环境。已有配套 PyTorch、NumPy、SciPy 的环境可直接运行。

```text
python -m pip install -r requirements-point.txt
python -m b02 replay examples/smoke.csv --tracker point-scorer --device cpu --out runs/smoke-current
```

代码附带约 40 KB 的本队推理权重，无需上游模型或完整训练检查点。主配置 `profiles/current.json` 自动校验权重、步数与种子。重复回放时换一个尚未使用的输出目录。

## 真实数据与查看器

使用自己从赛事渠道取得的训练 ZIP。模型输入不含真值标签，标签只用于回放计分。

```text
python -m b02 replay "你的训练数据.zip" --tracker point-scorer --scene train_015 --out runs/train-015-current
python -m b02 visualize "你的训练数据.zip" --run runs/train-015-current --scene train_015 --out runs/train-015-current/viewer.html
```

双击生成的 HTML 查看真实点迹、输出 ID、漏点、杂点和评分表。回放同时写出 `summary.json`、逐圈 JSONL、`scorecard.json/csv`、`results.csv` 与完整计分口径。

设备默认 `auto`，有 CUDA 时使用 GPU，也可指定 `--device cpu` 或 `--device cuda`。冻结对照使用云端环境；CPU 可用于联调和查看，浮点结果可能有细微差异。

完整开发集使用 `--split dev`。封存测试集清单仍保留，当前阶段不使用 `--split holdout`。

## 官方收发联调

Windows 使用官方代码样例包准备本地收发程序。

```text
python -m b02 prepare-official "官方代码样例及数据收发程序.zip"
python -m b02 official --tracker point-scorer --out runs/official-current
```

该命令在本机通过官方 TCP 收发程序运行，不上传官网。可用 `--input-dir` 指定少量样例目录，用 `--port` 更换端口。Linux 官方 EXE 联调需要 Wine／Xvfb，说明见 [历史开发文档](docs/LEGACY_GUIDE.md)。

## v1 Docker 提交包

团队编号为 `T000906`。运行 `python tools/build_submission.py --out runs/submission-build`，会生成 ZIP 根目录直接包含 `Dockerfile` 的 `T000906_v1_code.zip` 和配套构建目录。先使用上面的 `prepare-official` 命令准备 `.official`，官方 EXE 与收发源码由本地官方包取得，不上传到 Git。

构建脚本会拒绝覆盖已有目录，并检查冻结算法、配置和权重，避免将后羿实验或后改源码混入 v1。容器命令与离线验收流程见 [提交模板说明](submission-template/README.md)。`B02当前版本.zip` 继续用于开发共享，正式程序包单独命名；不要把样例联调结果当作正式验证结果上传。

已通过的官方样例双跑、内存和耗时记录见 [交付验收报告](docs/SUBMISSION_ACCEPTANCE.md)。正式验证集和官网接收仍待实际验证。

## 团队约定

- 常规运行统一使用 `python -m b02`，当前模型显式指定 `--tracker point-scorer`。
- `profiles/current.json` 是主配置，`assets/point-scorer-8000.pt` 是配套权重。旧 8000 步参考配置为 `profiles/reference.json`。
- 本地当前结果只从 `runs/current/index.html` 进入；其他 `runs/` 目录用于实验追溯。原始数据和运行结果不进入 Git，队友按上述命令生成自己的查看器。
- 总分均为本地估算，正式成绩以官网为准。30 文件开发集与 8 文件内部校准的分数分开比较。
- 原始简单基线仍可用 `--tracker baseline`。不指定 `--tracker` 时保持原基线，避免旧命令行为改变。
- 新实验使用独立分支和输出目录，不覆盖主配置或冻结结果。当前共享分支为 `b02-association-calibration`，通过 PR 汇总后续改动。

## 复查与研究资料

```text
python -m unittest discover -s tests -v
python experiments/association/check_interventions.py
```

[历史开发文档](docs/LEGACY_GUIDE.md)保留早期基线、收发和评分细节；[实验索引](docs/EXPERIMENTS.md)汇总研究记录。历史脚本用于复现，日常运行以本页及主配置为准。
