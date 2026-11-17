"""Render all preregistered history/set ablations without selecting a winner."""
import argparse
from html import escape
import json
from pathlib import Path


def summarize(root):
    report = json.loads((root/'models/report.json').read_text(encoding='utf-8'))
    assert report['complete']
    identity_rows, rejection_rows = [], []
    for name, record in report['identity'].items():
        m = record['selected']
        identity_rows.append([name,str(m['step']),f"{m['binary_loss']:.6f}",
            str(m['fixed_coverage_column']['wrong']), f"{100*m['positive_row_match_recall']:.3f}",
            f"{100*m['absent_row_false_accept_rate']:.3f}"])
    for name, record in report['rejection'].items():
        m = record['validation_calibrated']
        rejection_rows.append([name,str(record['selected']['step']),
            f"{record['offset_training_only']:.6f}", f"{100*m['positive_match_recall']:.3f}",
            f"{100*m['absent_false_accept_rate']:.3f}", str(m['wrong_target_on_present_rows'])])
    identity_headers = ['身份头 / 种子','追加步数','验证 BCE','固定接受量列误配','正确行召回 %','无正确候选错误接受 %']
    null_headers = ['拒配头 / 种子','步数','训练拟合偏移','正确行召回 %','无正确候选错误接受 %','有正确候选却接错次数']
    def markdown(headers, rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
            ['| '+' | '.join(row)+' |' for row in rows])
    text = '''# 完整历史与候选集合对照 · 2026-11-17

本轮 12 次训练完成，两项预注册能力门槛均未通过。保留全部失败记录，不进入整场回放，不替换成熟 v2 候选或正式 v1.1，不称 v3。此处没有新的官网成绩。

## 固定条件与数据角色

复用上一阶段的 24 个训练文件和 8 个内部头验证文件，不重新采样场景。训练步数上限各 1200，固定种子 20261118 / 20261119，检查点 0 / 200 / 600 / 1200。代码与门槛在运行前冻结，完整协议在 [protocol.json](protocol.json)。

这 8 个场景属于父模型 fit64，也已用于开发。它们不是整套系统未见数据。原校准八文件、新工作验证 16、确认集及正式封存 29 本轮均未读取。样本只包含过去输出形成的可靠身份锚，纯杂点或含混历史不在此训练分布内，不能据此推断完整运行表现。

## A：完整历史，还是仅仅多训练？

此前配对头只直接选取轨迹 98 维中的 8 维，连同点边 85 维和发布元数据 3 维形成 96 维输入。本轮将全部 98 维轨迹输入恢复，形成 186 维；这是已有因果特征的恢复，没有加入未来帧或真值 ID。

两组都从各自种子的旧二元头出发，并在初始化时精确保留打分函数。`continued96` 与 `full186` 接受相同的追加步数、损失、优化器和抽样 ID；检查点只按已规定的验证 BCE 选择。`full186` 增加了输入及参数，不能视为单一语义因素的完美消融。

'''+markdown(identity_headers,identity_rows)+'''

固定接受量为 64417 个强争抢列；这是冻结历史上的局部身份判断，不是整场点命中或官网成绩。

两种子列误配分别由 1259→1266、1178→1172，未达到预设减少至少 5%。第二种子的错误接受也未满足保护条件。结论是：**在本轮容量、预算和监督条件下，直接补全已有历史输入没有稳定收益**，不能外推成“历史无用”或“架构上限”。继续训练的两组自身也发生工作点变化，不把它们当作旧方案的直接升级。

## B：拒配是否需要看见候选集合？

冻结旧二元配对打分器，单独训练拒配头。历史对照读取轨迹 101 维；候选组额外对每条 96 维边编码，按真实候选列表做均值、最大值池化并加入候选数量。两组使用同尺寸解码器；历史组的上下文输入固定为零。候选编码器的有效参数只在候选组使用，因此这不是完全等有效参数量对照。

集合输入只依赖几何候选成员。未知标签候选仍进入输入；含未知候选的行不参与本轮拒配损失与评估。训练行均匀抽样，以及正确候选存在／不存在各一半抽样，组成 2×2 对照。主比较在运行前固定为 balanced 候选组对 balanced 历史组。

全部模型按平衡验证行损失选检查点；随后仅用训练 24 文件拟合一个 null 偏移，使正确匹配接受数量保留在旧二元头水平。验证标签没有用于选择该偏移。

'''+markdown(null_headers,rejection_rows)+'''

均匀抽样四个模型均选中第 0 步。平衡抽样两种子的两组均选中 600 步。这支持“拒配样本稀少需要单独处理”的研究方向，但单一选型结果不等于训练永远无效。

在训练拟合工作点上，候选组的无正确候选错误接受相对历史组由 50.060%→44.418%、47.129%→43.122%；但正确候选存在时，接错点次数由 314→452、324→440。第一个种子达到至少 10% 的拒错改善，第二个没有；两个种子均未满足错点接受保护条件。预注册门槛均未通过。

**机制边界可以确认，改进机制尚未证明：**本实验故意冻结了配对排序，拒配头只能在现有最优候选与 null 之间取舍，无法把排错的正确点提到首位。当前结果表明它改变了接受人群，不能称为已经解决身份关联。上述行指标也不能外推成零碎度或连续跟踪子分。

## 收敛结论与下一项问题

本轮停止这两项组件的整场扩展，不用次要均匀组替换主比较，不用最好种子覆盖失败门槛。原二维整场回放保留在 `runs/competition-learning/index.html`；没有给本轮未做整场的模型编造回放或估分。

下一项优先验证的结构能力是：**在同一次决策中，同时让边评分看见其他候选点，以及争抢该点的其他轨迹，再联合学习不匹配。**可参考二部图消息传递，而非立即更换图像骨干或下载视觉预训练权重。其动机与局限见 [开源架构一手资料](../../docs/TRACKING_ARCHITECTURE_RESEARCH_20261117.md)。这是一项新的待验证假设，本轮没有训练或证明该结构。

若推进，先在相同 24/8 上设独立协议，对比“逐边残差”和“一层行列上下文残差”，保持候选构造、运动模型、点分数与预算相同；先证明困难争抢下的排序与拒配能共同改善，再按固定条件接入整场。不能因局部指标好看就放弃连续、混轨和命中保护条件。

## 复核与复现

新代码位于 `experiments/set_association/`；旧模型源码未改。新测试覆盖原函数保留、集合顺序与填充不变性、标签可见性不改变集合输入、拒配训练不修改配对器。云端 Python3.10 / PyTorch2.2 的 12 项前置测试通过，本地完整 92 项通过。

云端原始证据目录：`/root/shared-nvme/b02/v2-history-set-20261117`；本地 `runs/set-association/models/` 保留 12 个权重、全部历史与报告。权重不进入 Git。样本复用 `/root/shared-nvme/b02/v2-competition-learning-20261117/samples`，原样本协议摘要与旧权重摘要均受检查。

入口示例（路径替换为实际资源；必须提供协议匹配的旧模型及完整样本）：

```bash
python experiments/set_association/train.py --samples OLD_SAMPLES --sample-protocol experiments/competition_learning/protocol.json --prior-models OLD_MODELS --protocol experiments/set_association/protocol.json --prior PARENT_MEMORY.pt --affinity OLD_AFFINITY.pt --out NEW_OUTPUT
python -m unittest tests.test_set_association -v
python experiments/set_association_report.py runs/set-association
```

本研究以作者方法为参考，并未复现 MPNTrack、MOTR 或 Deep Sets 的完整论文系统。
'''
    seconds = report['elapsed_seconds'];gib = report['peak_cuda_bytes']/2**30
    text += f'\n12 次模型训练、验证与训练偏移计算合计约 {seconds:.1f} 秒，峰值 CUDA 张量分配约 {gib:.2f} GiB；不含样本加载、进程启动和云平台账单开销。协议 SHA256：`{report["protocol_sha256"]}`。\n'
    target = Path(__file__).parent/'set_association'
    (target/'RESULTS_20261117.md').write_text(text,encoding='utf-8',newline='\n')
    (target/'RESULTS_20261117.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    def htmltable(headers, rows):
        return '<div class="scroll"><table><tr>'+''.join('<th>'+escape(s)+'</th>' for s in headers)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+escape(s)+'</td>' for s in row)+'</tr>' for row in rows)+'</table></div>'
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B02 历史与集合对照</title>
<style>body{background:#101824;color:#e5edf7;font:16px/1.7 system-ui;margin:0}main{max-width:1350px;margin:auto;padding:24px}section{padding:20px;background:#192638;border:1px solid #34465d;border-radius:8px;margin:20px 0}h1{font-size:28px}h2{font-size:21px}a{color:#8ee7d2}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px;border-bottom:1px solid #34465d;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}.warning{color:#ffdc89}nav{display:flex;gap:20px;flex-wrap:wrap}</style>
<main><h1>完整历史与候选集合 · 2026-11-17</h1><section><h2>12 次对照完成，未进入整场扩展</h2><p class="warning">两项预注册能力门槛均未通过。没有本轮整场估分或官网新分数，不升级 v3。</p><p>仍使用原 24 个训练场景与 8 个内部头验证场景。父模型见过这些数据，属于内部开发。原校准八文件、新工作验证 16、确认集与正式封存 29 本轮未读取。</p></section>
<section><h2>A · 恢复完整历史 vs 等步数继续训练</h2><p>两组均从旧二元头初始化，初始函数相同，抽样 ID 与预算相同。检查点按验证 BCE 选择。固定列接受量 64417；下表不是整场成绩。</p>'''
    html += htmltable(identity_headers,identity_rows)+'''<p>列误配变化约 +0.6% / −0.5%，未达到双种子至少减少 5% 的门槛。不能据此断言历史无用。</p></section><section><h2>B · 历史拒配 vs 候选集合拒配</h2><p>配对排序保持冻结。下面的偏移仅用训练集拟合，验证不搜索阈值；含未知候选的行排除评估。</p>'''
    html += htmltable(null_headers,rejection_rows)+'''<p>候选组减少无正确候选时的错误接受，但两种子在有正确候选时放行了更多错误点，未通过保护条件。拒配头无法修复被冻结的候选排序。</p></section><section><h2>下一项结构问题</h2><p>让边评分同时看到当前行的其他候选、以及同一个点的竞争轨迹，并与拒配联合学习。先证明局部能力，再做整场；本轮没有训练或证明该新结构。</p><nav><a href="../../docs/TRACKING_ARCHITECTURE_RESEARCH_20261117.md">开源架构一手资料</a><a href="../../experiments/set_association/RESULTS_20261117.md">完整结果与复现</a><a href="models/report.json">全部训练与检查点</a><a href="../../experiments/set_association/protocol.json">运行前协议</a><a href="../competition-learning/index.html">上阶段原二维整场回放</a></nav></section>'''
    html += f'<p>模型训练／验证及偏移计算约 {seconds:.1f} 秒，CUDA 张量峰值约 {gib:.2f} GiB。全部失败记录保留。</p></main></html>'
    (root/'index.html').write_text(html,encoding='utf-8',newline='\n')
    print('wrote',target/'RESULTS_20261117.md',root/'index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    summarize(parser.parse_args().root)
