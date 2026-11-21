# WangXinCup_ZiDongHuaXi

## 最新突破：MasterTracker V19.5 (`b02-master-v19.5`)

- **架构升级**：`0.30 × Houyi-Dense-v3 (8-scan)` + `0.70 × Houyi-V18-Joint (12-scan)` + **50% 重叠方位角扇区 TTA** + **V18-B 跨圈亲和头与后向运动残差头** + **V18-C 24 圈轨迹 GRU & 二分冲突图注意力 (`TrackSeqCrossAttnNet`)** + **V19.5 上向自适应编队方差 & 杂波劫持踢除门 (`D1b + D2c`)**。
- **严格无泄漏本地测试（170 场景训练 / 30 场景验证）**：
  - `dev_30`：**96.7147**（`Recall=96.27`, `Clutter=99.34`, `Frag=94.13`, `Timeliness=98.24`, `Continuous=97.13`）
  - 高杂波大噪声域外压力测试 `STRESS`：**94.4637**（较 V17 基线提升 **+3.8715**）
- **实测推理效率（RTX 5070 Laptop，bf16 SDPA）**：单圈均值 **178.6 ms**（P95 `202.1 ms`，max `393.7 ms`，远低于 29,000 ms 通信限额），峰值显存 **224.6 MB**，多包 `state={}` 边界完全确定性复现。
- **完整代码、训练流程与打包说明**：见 [MasterTracker V19.5 复现指南](b02-starter/experiments/master_v19/README.md)（历史 V14 初版 A 榜 `90.62` 记录见 [master_v14](b02-starter/experiments/master_v14/README.md)）。

---

## 团队早期基线 (`b02-association-calibration`)

当前 B02 点模型 v1 的参数、完整开发集结果和验证记录见 [当前成果](b02-starter/CURRENT.md)。队友运行请从 [统一入口说明](b02-starter/README.md) 开始。

代码包含约 40 KB 的推理权重，原团队共享分支为 `b02-association-calibration`。原始赛事数据和运行产物由队员各自获取或生成。
