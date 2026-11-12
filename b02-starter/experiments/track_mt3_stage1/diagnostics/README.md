# 零航迹诊断实验

这些脚本是可保留的诊断工具，不是比赛推理入口。按 `../README.md` 建好云端环境后，把本目录上传到共享存储。工作目录保持为 CNSF 源码根目录，使用已安装该项目的 Python。

## 复现与筛选检查

```bash
python repro_empty.py --checkpoint /path/to/resumed.pt
python repro_empty.py --checkpoint /path/to/resumed.pt --frames 21 --start 20
python diagnose_model.py --checkpoint /path/to/resumed.pt --out /path/to/new-report --sweep
```

实际脚本路径应为上传目录中的绝对路径。`repro_empty.py` 对“零航迹”故意返回失败状态；它只复现症状，不代表模型质量验收。`diagnose_model.py` 从检查点重建配置，测三个固定独立模拟序列（20261111–20261113），每个 100 帧，汇总时评分帧为 20–99。阈值扫描只是定位过滤层的实验，不是选择正式阈值。

GOSPA 使用上游实现，cutoff=2、order=1、alpha=2，越低越好。这里空预测器每帧的 GOSPA 等于该帧真目标数。B02 的官方五项加权指标与此不同。

## 学习对照

```bash
python learning_probe.py --checkpoint /path/to/resumed.pt --out /path/to/fixed --mode fixed --steps 300
python learning_probe.py --checkpoint /path/to/resumed.pt --out /path/to/random --mode random --steps 2000
python learning_probe.py --checkpoint /path/to/resumed.pt --out /path/to/tfzero --mode random --steps 500 --teacher-forcing-zero
```

三组均从同一个第 401 步检查点开始，不串联训练。

- fixed：固定种子 90119 生成两个片段（各两个窗口），反复训练同一组数据；评估也只在这组数据上，不能说明泛化。
- random：保持原短训练配置不变，继续在线随机模拟；每 500 步在种子 20261111 的固定独立序列上检查，评估不会参与梯度更新。
- tfzero：与 random 的前 500 步对照，仅把 teacher forcing 起止概率设成 0；相同的模拟样本生成序列，但训练路径和梯度会改变。禁止把原随机组 2000 步与本组 500 步当成等预算对照。

`query_probe.py` 是锁定本次云端第 401 步检查点的取证脚本，包含单窗口/批量路径一致性、去真值标签、量测全零扰动，以及真值辅助传播的对照。**真值辅助仅用于诊断训练机制，不能进入部署推理。** 默认路径需按其他环境修改。

所有脚本不修改上游源码或原始检查点；输出目录必须使用新目录（锁定查询探针除外）。云端运行建议加 timeout 并保存日志。结论应同时看置信度、定位误差、误报和是否依赖真值辅助，不能只看损失下降或输出非空。
