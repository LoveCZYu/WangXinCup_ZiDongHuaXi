# MasterTracker V14 + Houyi-Dense-v3 复现指南

本目录包含 **MasterTracker V14 + Houyi-Dense-v3** 完整可复现实现（100% 逐圈因果在线跟踪，零未来帧泄漏）。

## 成绩总览

| 版本 | `dev_30` 本地总分 | 掌握率 (30%) | 虚警分 (10%) | 零碎度 (20%) | 及时性 (20%) | 连续跟踪 (20%) | 官网 A 榜 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Starter 基线 (`point-scorer-8000`) | 79.40 | 91.11 | 93.90 | 57.92 | 82.08 | 73.41 | 75.24 |
| `V13 + Houyi-Dense-e02` | 93.70 | 94.97 | 99.26 | 85.43 | 96.17 | 94.82 | — |
| `V14 + Houyi-Dense-e02` | 94.36 | 94.95 | 99.29 | 88.34 | 96.12 | 95.28 | — |
| **`V14 + Houyi-Dense-v3-e02` (`T=1.15`)** | **94.76** | **95.30** | **99.28** | **88.86** | **96.76** | **95.59** | **90.62** |

- 完整 `dev_30` 诊断指标保存在 [`dev30_results.json`](dev30_results.json)（点命中率 `96.72%`，漏检航迹 `0` 条，`79.1%` 航迹零断链）。
- 推理速度（RTX 5070 Laptop，bf16 SDPA）：单圈均值 **27.3 ms** / P95 **32.4 ms**，GPU 峰值显存 **123 MB**。

---

## 一、环境准备

推荐 Python 3.10+ 与 CUDA 12.x（PyTorch 2.7+）：

```bash
cd b02-starter
pip install -r experiments/master_v14/requirements-master.txt
```

> Windows 下如遇到 OpenMP 冲突，代码内已自动设置 `KMP_DUPLICATE_LIB_OK=TRUE`。

额外需要从赛事官方渠道获取两个本地静态包（不入 Git）：
1. **后羿基础模型包**：`HouyiBypassV2/`（需包含 `code/model_association/` 与 `Model/{model_config.json, data_config.json}`；从头训练时还需 `Model/model.pth`）
2. **初赛训练数据集**：`智能体应用赛题B02--目标检测跟踪--初赛训练数据集.zip`

---

## 二、快速复现推理与可视化（已有权重时）

将权重 `houyi-dense-v3-e02.pt` 放到 `b02-starter/assets/`（或通过 `--checkpoint` 指定），并通过 `--houyi-dir`（或环境变量 `B02_HOUYI_DIR`）指向 `HouyiBypassV2`：

### 1. 单场景回放 + 离线 HTML 查看器
```bash
python -m b02 replay "你的训练数据.zip" \
  --tracker master \
  --houyi-dir "path/to/HouyiBypassV2" \
  --checkpoint "path/to/houyi-dense-v3-e02.pt" \
  --scene train_015 \
  --out runs/master-v14-015

python -m b02 visualize "你的训练数据.zip" \
  --run runs/master-v14-015 \
  --scene train_015 \
  --out runs/master-v14-015/viewer.html --replace
```

### 2. 完整 30 场景无泄漏开发集 (`dev_30`) 复现
```bash
python -m b02 replay "你的训练数据.zip" \
  --tracker master \
  --houyi-dir "path/to/HouyiBypassV2" \
  --checkpoint "path/to/houyi-dense-v3-e02.pt" \
  --split dev \
  --out runs/master-v14-dev30
```

---

## 三、从零复现模型训练（严格排除 `dev_30`）

训练脚本 [`train_houyi_dense.py`](train_houyi_dense.py) 默认严格排除 `splits/v1.json` 中的 30 个 `dev` 场景（仅用 170 个非 dev 场景），采用两阶段不相交密集采样 + 起点 `obs1..3` 3× 加权 + `label_smoothing=0.01`：

```bash
# Stage 1：采样 6.5 万扇区 (circle % 22 == 14)，2 epoch，lr 4e-5 -> 4e-6
python experiments/master_v14/train_houyi_dense.py \
  --stage 1 \
  --data "你的训练数据.zip" \
  --houyi-dir "path/to/HouyiBypassV2" \
  --init-ckpt "path/to/HouyiBypassV2/Model/model.pth" \
  --out-dir runs/houyi_dense_170

# Stage 2：采样 12.8 万不相交扇区 (circle % 11 == 5)，2 epoch，lr 2e-5 -> 2e-6
python experiments/master_v14/train_houyi_dense.py \
  --stage 2 \
  --data "你的训练数据.zip" \
  --houyi-dir "path/to/HouyiBypassV2" \
  --init-ckpt runs/houyi_dense_170/houyi-dense-e02.pt \
  --out-dir runs/houyi_dense_170
```

> 正式冲榜如需吃满全部 200 个场景，在上述命令末尾追加 `--all-scenes` 即可（仅用于提交，不用于本地 `dev_30` 评估）。

---

## 四、官方 `serve.exe` 联调与 A 榜结果包生成

1. 将 [`algorithm_submission.py`](algorithm_submission.py) 复制覆盖到官方 `participant_demo/src/algorithm.py`。
2. 一键运行官方 `serve.exe` TCP 联调并自动打包 A 榜上传用 ZIP：

```bash
python experiments/master_v14/pack_a_board.py \
  --demo-dir "path/to/participant_demo" \
  --out-dir runs/a_board_master_v14 \
  --zip-out runs/B02_A_master_v14.zip \
  --houyi-dir "path/to/HouyiBypassV2" \
  --checkpoint "path/to/houyi-dense-v3-e02.pt"
```

---

## 五、核心代码结构

- [`b02/master_tracker.py`](../../b02/master_tracker.py)：自包含 `MasterTracker V14` 与 `HouyiScorer`（支持单权重与多权重 logit 集成、`T=1.15` 温度校准）
  - 竞争安全 Level-0 原子编队模板匹配（`k=2`、`k=3` 及 `2-of-3` 子对）
  - 已确认-新生僚机跨龄预锁（`co=1`）+ 编队内低分杂波隔离
  - 不缩门限的距离精权代价项（`r_cost_w=0.40, r_sigma=0.005, mc_add=0.30`）
  - 有界紧编队单点歧义守卫 + 僵尸重复轨裁剪 + 活跃漏检/休眠双层 3D OLS 断链重连（TSA）
- [`b02/__main__.py`](../../b02/__main__.py)：CLI 新增 `--tracker master` 与 `--houyi-dir`
