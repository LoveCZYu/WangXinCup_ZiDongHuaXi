# MasterTracker V19.5 + Houyi-V18-Joint + TrackSeqCrossAttnNet 复现指南

本目录包含 **MasterTracker V19.5** 完整自包含源码、三阶段训练流水线、端到端验证基准与官方打包脚本（100% 逐圈因果在线跟踪，零未来帧泄漏）。

## 成绩总览（严格 170 场景训练 / 30 场景无泄漏验证）

| 版本 | `dev_30` 本地总分 | 掌握率 (30%) | 虚警分 (10%) | 零碎度 (20%) | 及时性 (20%) | 连续跟踪 (20%) | ID 切换 | `STRESS` 压力测试 | 官网 A 榜 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Starter 基线 (`point-scorer-8000`) | 79.40 | 91.11 | 93.90 | 57.92 | 82.08 | 73.41 | 15,363 | — | 75.24 |
| V14 (`Houyi-Dense-v3`, `T=1.15`) | 94.7600 | 95.30 | 99.28 | 88.86 | 96.76 | 95.59 | 1,808 | — | **90.62** |
| V15 (`v3` + `HouyiPairHead`) | 95.0639 | 95.30 | 99.29 | 90.28 | 96.76 | 95.69 | 1,392 | 88.7954 | — |
| V16 (`v3+v4` + `Pair` + `IMM`) | 95.3084 | 95.45 | 99.31 | 90.85 | 96.85 | 96.01 | 1,406 | 89.5798 | — |
| V17 (`M1` + `ContextualConflictNet`) | 96.0247 | 95.95 | 99.29 | 92.04 | 97.98 | 96.53 | 1,007 | 90.5922 | — |
| V18-ABC + `v3_TTA` | 96.5769 | 96.31 | 99.33 | 93.30 | 98.27 | 97.19 | 864 | 91.8483 | — |
| **V19.5 生产版（当前）** | **96.7147** | **96.27** | **99.34** | **94.13** | **98.24** | **97.13** | **937** | **94.4637** | — |

- 实测全链路 GPU 推理基准（RTX 5070 Laptop，bf16 SDPA）：
  - `train_015`：线上 `95.9078` vs 离线缓存 `95.9078`（差值 `-0.0000`），均值 `182.1 ms/圈`，max `393.7 ms`
  - `train_123`：线上 `96.9436` vs 离线缓存 `96.9436`（差值 `-0.0000`），均值 `175.2 ms/圈`，max `343.1 ms`
  - 全局均值 **`178.6 ms/圈`**（P95 **`202.1 ms`**，max **`393.7 ms`**，远低于 29,000 ms TCP 时限），峰值显存 **`224.6 MB`**，多次重置 `state={}` 回放 719 圈逐位一致（详见 [`dev30_results.json`](dev30_results.json)）。

---

## 核心技术升级（V18-A/B/C + V19.5）

1. **V18-A：全覆盖 12 圈联合训练主干 + 50% 重叠方位角扇区 TTA**
   - `Houyi-V18-Joint-12scan` 通过旋转步长采样（`stride=3 × 3 epoch`）覆盖 100% 雷达圈，训练时注入密集杂波（`+12%~38%`）、历史僚机丢点、温和传感扰动与 `360°` 方位角旋转等变增强。
   - 推理侧在 `HouyiScorer` 对 `8-scan Houyi-Dense-v3`（权重 `0.30`）和 `12-scan Houyi-V18-Joint`（权重 `0.70`）均启用半扇区偏移（`+0.5 / n_sectors`）融合（`T=0.90`），消除扇区边缘边界效应。
2. **V18-B：12 圈硬负样本 `HouyiPairHead` + `BackwardMotionHead`**
   - 共享 `Houyi-V18-Joint` 编码特征，在单次前向中同步输出 12 圈跨圈点对亲和概率（紧编队难样本 `AUC = 0.9938`）和 CenterTrack 风格逐点后向运动残差（`MAE = 0.1720`），显存与主干前向完全复用。
3. **V18-C：24 圈轨迹 GRU + 二分冲突图注意力 (`TrackSeqCrossAttnNet`)**
   - `TrackSeqMemory` 为每条航迹维护最近 24 圈时序状态（相对位置、速度、转弯率、IMM 模式、幅值/信噪比/观测掩码），联合 40 维竞品边与 V18 后向运动残差特征，经关系偏置多头自注意力（`self / same_track / same_det / peer`）和 LBFGS 温度校准后，未见场景冲突图准确率达 `AUC = 0.9952, P@0.95 = 0.9942, ECE = 0.0098`。
4. **V19.5：法医级反串轨双门限升级（`D1b + D2c`）**
   - **`D1b` 上向自适应编队方差**：保持干净场景精度下界不变，允许编队置换与模板误差中的 `(az, el, r)` 标准差依据跟踪残差 EMA 自适应上浮，在高噪声压力测试 `STRESS` 中大幅避免误拆编队（贡献 `+2.4+` 域外抗噪收益）。
   - **`D2c` 编队杂波劫持剔除门**：精准识别紧双机单检漏检时另一杂波/入侵点劫持匹配的畸变构型，及时踢除离群伪匹配并回落至单检护航与质心滑行（贡献 `dev_30 Frag 93.30 -> 94.13` 且 `STRESS -> 94.4637`）。

---

## 一、环境准备

推荐 Python 3.10+ 与 CUDA 12.x（PyTorch 2.7+）：

```bash
cd b02-starter
pip install -r experiments/master_v19/requirements-master.txt
```

需要从赛事官方渠道获取两个本地静态包（不入 Git）：
1. **后羿基础模型包**：`HouyiBypassV2/`（含 `code/model_association/` 与 `Model/{model_config.json, data_config.json, model.pth}`）
2. **初赛训练数据集**：`智能体应用赛题B02--目标检测跟踪--初赛训练数据集.zip`

---

## 二、快速推理与可视化（已有权重时）

将权重放入 `b02-starter/assets/`（或使用环境变量 `B02_CHECKPOINT` / `B02_PAIR_CHECKPOINT` / `B02_SEQ_CHECKPOINT`）并设置 `B02_HOUYI_DIR` 指向 `HouyiBypassV2/`，即可自动激活完整 V19.5 管线：

```bash
# 设置权重路径（如未放到 assets/ 下，Windows PowerShell 示例）
$env:B02_HOUYI_DIR = "path/to/HouyiBypassV2"
$env:B02_CHECKPOINT = "output/houyi_dense_170/houyi-dense-v3-e02.pt,output/houyi_v18_170/houyi-v18-joint-12scan.pt"
$env:B02_PAIR_CHECKPOINT = "output/houyi_v18_170/houyi-v18-joint-12scan.pt"
$env:B02_SEQ_CHECKPOINT = "output/houyi_v18_170/track_seq_cross_attn_v18.pt"

# 1. 单场景回放 + 生成 HTML 查看器
python -m b02 replay "你的训练数据.zip" --tracker master --scene train_015 --out runs/master-v19-015
python -m b02 visualize "你的训练数据.zip" --run runs/master-v19-015 --scene train_015 --out runs/master-v19-015/viewer.html --replace

# 2. 完整 30 场景无泄漏开发集 (dev_30) 复现
python -m b02 replay "你的训练数据.zip" --tracker master --split dev --out runs/master-v19-dev30
```

---

## 三、从零复现三阶段训练（默认严格排除 `dev_30`）

所有训练脚本默认排除 `splits/v1.json` 中的 30 个 `dev` 场景（仅用 170 个非 dev 场景）。**最终全量 200 场景冲榜时**在 Step 1 / Step 2 / Step 3 命令末尾追加 `--all-scenes` 即可：

```bash
# Step 1：训练 8 圈密集点模型 Houyi-Dense-v3 (Stage 1 -> Stage 2)
python experiments/master_v19/train_houyi_dense.py \
  --stage 1 --data "你的训练数据.zip" --houyi-dir "path/to/HouyiBypassV2" \
  --init-ckpt "path/to/HouyiBypassV2/Model/model.pth" --out-dir output/houyi_dense_170

python experiments/master_v19/train_houyi_dense.py \
  --stage 2 --data "你的训练数据.zip" --houyi-dir "path/to/HouyiBypassV2" \
  --init-ckpt output/houyi_dense_170/houyi-dense-e02.pt --out-dir output/houyi_dense_170

# Step 2：联合训练 12 圈多任务模型 Houyi-V18-Joint (Point Cls + BackwardMotionHead + 12s PairHead)
python experiments/master_v19/train_houyi_v18_joint.py \
  --data-zip "你的训练数据.zip" --houyi-dir "path/to/HouyiBypassV2" \
  --init-ckpt output/houyi_dense_170/houyi-dense-v3-e02.pt \
  --out-dir output/houyi_v18_170 --epochs 3 --stride 3

# Step 3a：闭环 DAgger 采集 30 场景二分冲突图与 24 圈轨迹特征
python experiments/master_v19/collect_dagger_v18.py \
  --data-zip "你的训练数据.zip" --houyi-dir "path/to/HouyiBypassV2" \
  --v18-ckpt output/houyi_v18_170/houyi-v18-joint-12scan.pt \
  --out output/houyi_v18_170/dagger_v18_seq_comps30.pkl --num-scenes 30

# Step 3b：训练并 LBFGS 校准 V18-C 序列图注意力关联器 TrackSeqCrossAttnNet
python experiments/master_v19/train_seq_associator_v18.py \
  --data output/houyi_v18_170/dagger_v18_seq_comps30.pkl \
  --out output/houyi_v18_170/track_seq_cross_attn_v18.pt
```

---

## 四、官方 `serve.exe` 联调与 A 榜结果包生成

1. 将 [`algorithm_submission.py`](algorithm_submission.py) 复制覆盖到官方 `participant_demo/src/algorithm.py`。
2. 一键运行官方 `serve.exe` TCP 联调并打包上传 ZIP：

```bash
python experiments/master_v19/pack_a_board.py \
  --demo-dir "path/to/participant_demo" \
  --out-dir runs/a_board_master_v19 \
  --zip-out runs/B02_A_master_v19.zip \
  --houyi-dir "path/to/HouyiBypassV2" \
  --checkpoint "output/houyi_dense_170/houyi-dense-v3-e02.pt,output/houyi_v18_170/houyi-v18-joint-12scan.pt" \
  --pair-checkpoint "output/houyi_v18_170/houyi-v18-joint-12scan.pt" \
  --seq-checkpoint "output/houyi_v18_170/track_seq_cross_attn_v18.pt"
```

---

## 五、核心文件清单

- [`b02/master_tracker.py`](../../b02/master_tracker.py)：自包含 V19.5 核心引擎（`HouyiScorer`、50% 重叠扇区 TTA、`HouyiPairHead`、`BackwardMotionHead`、`TrackSeqMemory`、`TrackSeqCrossAttnNet`、`FastSeqCrossAttnScorer`、`FormLink`、`MasterTracker`）
- [`experiments/master_v19/train_houyi_dense.py`](train_houyi_dense.py)：Stage-1 / Stage-2 8 圈密集点模型训练脚本
- [`experiments/master_v19/train_houyi_v18_joint.py`](train_houyi_v18_joint.py)：12 圈三头联合训练脚本（含 A 榜对齐域随机化）
- [`experiments/master_v19/collect_dagger_v18.py`](collect_dagger_v18.py)：闭环 DAgger 40 维图+24 圈时序样本采集脚本
- [`experiments/master_v19/train_seq_associator_v18.py`](train_seq_associator_v18.py)：`TrackSeqCrossAttnNet` 训练与 LBFGS 概率校准脚本
- [`experiments/master_v19/algorithm_submission.py`](algorithm_submission.py)：官方 TCP / Docker `algorithm.py` 即插即用入口
- [`experiments/master_v19/pack_a_board.py`](pack_a_board.py)：Windows 官方 `serve.exe` 全链路校验及结果 ZIP 自动打包脚本
- [`experiments/master_v19/dev30_results.json`](dev30_results.json)：严格 170 场景权重 `dev_30`、`STRESS` 及 RTX 5070 实测基准记录
