# 身份竞争前置诊断

这一步在固定哈希顺序选出的原 fit64 内八文件上，只运行当前候选并记录身份竞争与拒绝问题。不训练、不改变状态更新，不运行校准、新工作验证、确认或封存文件；结果不是泛化成绩。

运行前协议 `preflight-protocol.json` 固定源码摘要、父阶段协议、八文件和四个单线程 CPU worker 各最多 900 秒的预算。协议使用 LF 保存，上传后检查源码 SHA。`preflight.py` 复用已有点缓存和三个冻结模型组件；Probe 与上阶段相同，只记录候选边。

诊断有两种不同参考：

- **历史主 ID**：对每个真目标，从本圈以前已发布的命中里选次数最多的 ID，至少三次；并列取较小 ID。判断本圈点是否交给了该历史 ID，以及它是否有该点的记录候选边。此参考不保证是唯一正确的身份归属，也不是满分算法。
- **已有 ID 的历史身份锚**：在该输出 ID 本圈以前的已知标签里，至少三次命中同一真目标，且其比例至少 80%；已知杂点计入分母，未知 −1 不计。只对满足此条件的记录行，统计正确候选、其他目标、已知杂点、未知点和不匹配。当前标签不能定义自己的身份锚。

普通关联 top-k 边、当时实际生成的退役边和出生前转接提议共同构成“记录候选”。这些边受到不同匹配阶段及优先级影响。ID 在调用前的活跃／退役集合里、却没有记录边，**不能直接归因为几何门太小**；也可能是点已被占用、门限、top-k、保留期限或候选生成条件。仅凭这些统计不承诺可恢复分数。

`trusted_update_*` 只统计实际进入可信长期记忆的关联，不能当作全部物理状态更新量。`row_selected_*` 是内部关联，很多点会被后续输出门过滤。`with_correct_candidate_*` 与 `without_correct_candidate_*` 按该行是否存在正确身份点分组；没有正确候选的错误接受提示拒绝监督需求，但不证明训练已解决。

调用示例，在 `b02-starter` 内替换大写路径；输出目录必须不存在：

```text
python experiments/identity_competition/preflight.py --data FIT64_CAL8.zip --cache CACHE64 --prior MEMORY.pt --affinity AFFINITY.pt --head HEAD.pt --fusion FUSION.json --freeze experiments/robust_validation/freeze.json --protocol experiments/identity_competition/preflight-protocol.json --out runs/identity-preflight/part-0
```

分片时给每个 worker 使用相同 `--shards 4` 和不同 `--index 0..3`、输出目录。只导出逐场景计数、缓存和协议摘要，不导出原始点、标签数组或新模型权重。

八个分片场景下载到同一个 `runs/identity-preflight` 后，运行 `python experiments/identity_competition/summarize.py runs/identity-preflight` 生成可复核 JSON 和结果 HTML。完整来源与解释见 `RESULTS_20261117.md`；模型实际输出方案未改变，因此沿用上阶段二维回放。
