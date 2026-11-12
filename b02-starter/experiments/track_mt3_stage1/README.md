# Track-MT3 阶段 1：云端环境与训练链路验证

目的：在原项目的二维模拟数据上验证训练、权重保存、精确恢复、继续训练和逐帧推理。这里没有接入 B02 数据，也没有复现论文成绩或产生可提交的比赛模型。

上游：[daihuangyu/CNSF](https://github.com/daihuangyu/CNSF)，固定提交 `22c3a19c15df6c560eaadc74462b9b632ede2602`，使用其中的 **Track-MT3**，不是完整 CNSF 递归模型。上游许可证保留在克隆目录。

来源澄清：这是 CNSF 作者提供的 Track-MT3 对照实现，不是已经核实的 Track-MT3 原作者代码。Track-MT3 原始中文论文是陈辉等《Track-MT3：一种基于 Transformer 的新型多目标跟踪算法》，雷达学报，2024，[DOI 10.12000/JR24164](https://doi.org/10.12000/JR24164)。当前仓库关联论文为 [CNSF，2026](https://arxiv.org/html/2609.16054v1)。本文“原模型/原阈值”均指所固定的上游代码配置。

## 环境与安装

已验证的并行云公共镜像：界面名称 `PyTorch-24.01-py3（PyTorch2.2.0-Ubuntu22.04）`，实际 Python 3.10.12、PyTorch `2.2.0a0+81ea7a4`、CUDA runtime 12.3、驱动 550.54.14、RTX 3090 24GB。

`bootstrap.sh` 使用独立虚拟环境并继承镜像中的 PyTorch 和依赖，不全局升级 PyTorch。镜像缺少 `ensurepip`，所以使用 `--without-pip` 并继承已有 pip。实际依赖快照见每次结果中的 `pip-freeze.txt`；它用于记录，不建议直接在任意机器全量安装平台依赖。

将本目录上传到云端，例如 `/root/shared-nvme/b02/scripts/stage1`，然后：

```bash
bash /root/shared-nvme/b02/scripts/stage1/bootstrap.sh /root/shared-nvme/b02
cd /root/shared-nvme/b02/code/CNSF
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/root/shared-nvme/b02/envs/trackmt3/bin/python -m pytest -q \
  tests/test_capacity_matched_control.py tests/test_data_and_metrics.py \
  tests/test_model_cta_loss.py tests/test_operating_points.py \
  tests/test_tgospa.py tests/test_training_and_tracking.py
```

上述 6 个测试文件覆盖本次所用 Track-MT3 功能，不代表整个 CNSF 仓库全部测试。

## 运行短训练验证

保持工作目录为上游 CNSF 仓库根目录，执行：

```bash
/root/shared-nvme/b02/envs/trackmt3/bin/python \
  /root/shared-nvme/b02/scripts/stage1/verify.py \
  --out /root/shared-nvme/b02/runs/my-stage1-001 --steps 400 --frames 100
```

`--out` 必须是不存在的新目录。使用 tmux 或 nohup 启动可避免连接断开影响任务；建议外层加 `timeout 900`，完成后检查退出状态和 `report.json`。

脚本加载原始 `paper.yaml` 与 `training/track_mt3.yaml`，保持约 1938 万参数的原模型结构、20 帧窗口、损失设置和推理阈值。仅把 batch size 改为 2、clip_windows 改为 2、课程预热关闭，限制更新次数，并关闭外部 W&B 日志和周期全量评测。这是低成本运行验证，不是论文训练配方复现。

验收包括：每步损失和梯度有限、参数发生更新、保存后参数逐项精确恢复、恢复后再训练一步并再次更新、以新建模型进行逐帧推理、检查输出数值有限和单帧 ID 唯一。包含不满 20 帧的冷启动窗口。

## 产物与解释

- `report.json`：配置概要、验收状态、训练耗时、PyTorch 分配显存和推理耗时。
- `train/metrics.jsonl`：逐步损失、梯度和显存；`train/config.json`：完整生效配置。
- `train/checkpoints/stage1.pt`：短训练权重；`restore/checkpoints/resumed.pt`：恢复后再训练一步的权重。
- `inference.jsonl`：每帧模拟量测、真实目标位置、预测位置/ID/置信度和耗时。
- `pip-freeze.txt`：实际环境依赖快照。

随机训练批次的首末损失不能当成验证集效果；原推理阈值下也可能输出零条航迹。仅数百步训练不保证形成有效跟踪能力。输出是原模型的二维状态预测，尚未关联回 B02 的原始点号。

推理显存只统计新建模型推理时的 PyTorch 分配量，不等于 nvidia-smi 的进程总显存。耗时包含窗口传到 GPU 和模型推理，排除模拟数据生成、窗口构造和文件写入；统计均值/P95 时排除前 5 帧。小规模模拟场景不能代表 B02 密集点云下的资源需求。

代码、配置和说明可提交团队仓库；实验结果放在被忽略的 `runs/`，权重和凭据不要提交。每位队员使用自己的云账号和登录凭据。
