# 冻结验证与诊断入口

该目录用于一次固定对照，不提供自动调参。`freeze.json` 是实际运行前协议，包含权重／源码 SHA、16 文件名单、主要比较和三态结论规则。运行前先读 `RESULTS_20261117.md`；新 16 文件已完成评估，后续不再视作未见盲测。

需要既有点模型、长期记忆权重、affinity 权重、接续头和 fusion 配置，均不随 Git 分发。数据 ZIP 只含固定的 16 个 CSV，摘要必须与 freeze 一致。已有 CPU 环境需 PyTorch、NumPy、SciPy，沿用前序实验依赖。

以下在 `b02-starter` 目录执行，替换大写路径。默认单 worker；并行时固定 `--shards N --index I`，各回放写不同目录，缓存完成后再合并。

```text
python experiments/robust_validation/evaluate.py cache --data NEW16.zip --freeze experiments/robust_validation/freeze.json --out CACHE
python experiments/robust_validation/evaluate.py merge-cache --cache CACHE --freeze experiments/robust_validation/freeze.json
python experiments/robust_validation/evaluate.py evaluate --data NEW16.zip --cache CACHE --prior MEMORY.pt --affinity AFFINITY.pt --head HEAD.pt --fusion FUSION.json --freeze experiments/robust_validation/freeze.json --out runs/v2-validation/verify-0
python experiments/robust_validation/diagnose.py --data FIT64_CAL8.zip --cache CACHE64 --prior MEMORY.pt --affinity AFFINITY.pt --head HEAD.pt --fusion FUSION.json --freeze experiments/robust_validation/freeze.json --reference PREVIOUS_CANDIDATE/summary.json --out runs/v2-validation/diagnosis-0
python experiments/robust_validation/summarize.py runs/v2-validation
python experiments/robust_validation/report.py runs/v2-validation
```

cache 自动读取冻结点模型 `assets/point-scorer-8000.pt`，当前点与历史五圈用于因果评分。固定每场景状态重置，三个关联方案共用相同点分数。真值只进入回放评分和外部诊断。`diagnose.py` 仅允许原八个校准文件，实际输出先对照前序逐场景参考；真值辅助修复与实际模型分开保存。

二维回放仍使用 `python -m b02 visualize`，例如：

```text
python -m b02 visualize NEW16.zip --run runs/v2-validation/validation/candidate --scene train_011 --baseline runs/v2-validation/validation/wide --out runs/v2-validation/train_011-candidate.html
```

回放只导出了事先固定的 train_011。查看器中“基线”在此指成熟宽方案，完整结果包含三方案及所有 16 文件，而非最初简单基线。输出目录需要先由 summarize 产生。

`uncertainty.py` 根据与现有 scorecard 相同的充分统计量做场景配对重采样；它不是逐点重采样。`oracles.py` 的影子一对一归属和真值辅助修复都不是官方计分脚本。`frozen.py` 的 Probe 只记录候选，测试及实际回放确认它不改变输出。

重新运行旧结果汇总可覆盖 JSON、CSV、HTML；推理输出和缓存目录要求不存在，避免混入历史文件。原始数据、权重、缓存和查看器应通过现有团队文件渠道共享，不在 Git 存储。

`provenance/` 保存本次审核、无标签选择及冻结脚本的原样副本，供核查实际选择过程；它们保留当时工作区相对路径，不是重新划分数据的日常入口。不要在已完成的实验上重新运行 `freeze.py` 覆盖协议。精确数据审核 JSON 的摘要与冻结协议一致。
