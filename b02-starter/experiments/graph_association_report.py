"""Summarize frozen graph ablations, including all seeds and failed gates."""
import argparse
import hashlib
from html import escape
import json
from pathlib import Path


def render(root):
    target = Path(__file__).parent/'graph_association'
    root.mkdir(parents=True,exist_ok=True)
    path = root/'models/report.json'
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((target/'RESULTS_20261117.json').read_bytes())
    report = json.loads(path.read_text(encoding='utf-8'))
    protocol_path = target/'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    assert report['complete']
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == report['protocol_sha256']
    diagnostic_path = root/'diagnostic.json'
    if not diagnostic_path.exists():diagnostic_path.write_bytes((target/'DIAGNOSTIC_20261117.json').read_bytes())
    diagnostic = json.loads(diagnostic_path.read_text(encoding='utf-8'))
    dp = Path(__file__).with_name('graph_association_diagnostic.json')
    plan = json.loads(dp.read_text(encoding='utf-8'))
    assert hashlib.sha256(dp.read_bytes()).hexdigest() == diagnostic['diagnostic_protocol_sha256']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == plan['training_report_sha256']
    rows = []
    for name,record in report['models'].items():
        c = record['validation_calibrated']
        rows.append([name,str(record['selected']['step']),str(record['selected']['fixed_coverage_column']['wrong']),
            f"{100*c['positive_match_recall']:.3f}" if c else '不可校正',
            f"{100*c['absent_false_accept_rate']:.3f}" if c else '不可校正',
            str(c['wrong_target_on_present_rows']) if c else '不可校正',
            f"{record['offset_training_only']:.6f}" if c else '不可校正'])
    headers = ['关联头 / 种子','选中步数','固定接受量列误配','正确行召回 %','无正确候选错误接受 %','有正确候选却接错','训练拟合偏移']
    table = '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(row)+' |' for row in rows])
    verdict = '通过' if report['offline_passed'] else '未通过'
    md = f'''# 行列上下文联合关联对照 · 2026-11-17

两种子四次模型训练完成，预注册离线门槛**{verdict}**。主比较始终是同种子的上下文头对逐边头，不事后选择最好种子或改变接受条件。

## 问题与实现

前一轮候选感知拒配能减少部分错误接受，但固定配对排序无法修复“正确候选存在却选错点”。本轮让边评分与拒配共同学习，同时检验行、列上下文的增益。

每条 96 维候选边先编码为 32 维；对同轨迹候选行、争抢同一点的轨迹列分别计算均值／最大值／数量摘要。边残差读取自身编码与两侧摘要，null 读取轨迹历史 101 维与候选行摘要。一跳关系聚合后输出，没有未来帧或真值作为特征。

逐边对照具有相同解码器尺寸，关系摘要置零。两组均从对应种子的冻结二元头出发，新边残差与 null 输出层从零开始，初始打分函数完全一致。随机数在初始化后重置，抽样 ID 一致。有效上下文参数量不同，不能称完全等有效容量。

这是受 [MPNTrack](https://arxiv.org/abs/1912.07515) 和 [SuperGlue](https://arxiv.org/abs/1911.11763) 启发的小型雷达特征实现；没有复制上游代码／权重，没有复现完整论文。原 MPNTrack 的跨帧图不能直接当作比赛的因果实现。

## 固定训练与数据边界

仍用原缓存训练 24 文件／内部验证 8 文件，固定种子 20261118 / 20261119，各最多 2000 步，检查点为 0 / 200 / 600 / 1200 / 2000。

冻结旧二元头，只训练新共享边编码器、边残差和 null。两组相同 AdamW、步数及损失：已知边 BCE + 0.5×平衡行 set-CE（含 null）+ 0.5×列 set-CE（含零分未匹配槽）。行抽样为正确候选存在／不存在各 128；列目标优先历史主 ID，边 BCE 仍按语义同一目标监督。

特征池化依据完整几何成员，未知标签不改变上下文。含未知候选的行与列不参与相应集合损失。检查点按验证 BCE + 0.5×平衡行 CE + 0.5×列 CE 选择，行为指标和整场分数不选检查点。

随后只用训练 24 文件拟合 null 偏移，保持旧二元头正确匹配接受量；若新排序使这个接受量无法达到，记录校正不可行并判失败。验证不搜索阈值。

这些场景均在父模型 fit64 内，内部验证八文件已反复用于开发。不是整套系统未见验证。原校准八文件、新工作验证 16、确认集及正式封存 29 本轮未读取。缓存只保留可靠历史身份锚，忽略含混／纯杂点历史，实际运行分布更广。

## 全部主比较结果

{table}

固定列接受量为 64417 个强争抢列。正确行召回、错误接受及错接是冻结历史上的局部指标，不能视为整场命中率、零碎度或官网成绩。

## 运行前门槛及判定

两个种子均须满足：列误配至少减少 5%；正确匹配召回下降不超过 0.3 个百分点；无正确候选的错误接受至少减少 5%；有正确候选时的错接不增加超过 5%；两类错误接受合计至少减少 5%。完整 [协议](protocol.json) 保留实际数值和停止条件。

'''
    for seed,checks in report['comparisons'].items():
        md += f'- {seed}：'+ '；'.join(k+('通过' if v else '未通过') for k,v in checks.items())+'。\n'
    if report['offline_passed']:
        md += '\n达到组件推进条件，接下来在相同八文件自由运行，与冻结基准、旧二元列内方案和逐边联合方案对照。局部门槛通过本身不证明整场收益。\n'
    else:
        md += '\n按协议停止整场扩展，保留四个模型、全部检查点历史及失败条件，不采用次要指标重新定义成功，不更换正式版本。当前结果不足以证明完整图架构无效；也不支持仅增加这层上下文即可突破身份瓶颈。\n'
    audit = diagnostic['target_feasibility']
    md += f'''
## 追加只读诊断：核对解释，不改变门槛

这项诊断在看到本轮失败后另立 [协议](../graph_association_diagnostic.json)，只读同一内部验证八文件的已缓存候选；无训练、无阈值搜索、不产生部署候选，也不覆盖原失败结论。

**标签冲突不是本批主要原因。**审计假设是“多个旧 ID 同时被行标签要求接同一点，而一对一列约束不允许”。{audit['supervised_positive_rows']} 条已知正例行中，冻结候选图最大可同时正确匹配 {audit['maximum_simultaneous_correct_rows']} 条，缺口 {audit['minimum_positive_rows_unsatisfied_by_one_to_one']} 条（约 {100*audit['minimum_positive_rows_unsatisfied_by_one_to_one']/audit['supervised_positive_rows']:.3f}%）；没有任一列优先正边的正例行 {audit['positive_rows_without_any_preferred_column_edge']} 条。这个规模不足以支撑“标签冲突是主要瓶颈”的解释。旧二元头局部正确接受量也没有超过上述匹配容量。这只排除了本批缓存上的一种主要原因，不能推断所有运行状态都无目标冲突。

**行内排序变化很小，接受边界仍有取舍。**以下交换诊断固定已训练的边函数、null 函数及各自训练拟合偏移，不重新校正。联合训练函数被拆开后可能产生尺度差异，因此只用于数值归因，不作为因果证明或新候选。

| 种子 | 错序→正序 | 正序→错序 | 正序净增 | 两类错误接受合计：逐边→上下文 |
| --- | ---: | ---: | ---: | ---: |
'''
    for seed,entry in diagnostic['seeds'].items():
        t = entry['paired_transitions'];v = entry['variants']
        md += f'| {seed} | {t["wrong_to_correct_top"]} | {t["correct_to_wrong_top"]} | {t["wrong_to_correct_top"]-t["correct_to_wrong_top"]} | {v["independent"]["wrong_accepted"]}→{v["context"]["wrong_accepted"]} |\n'
    md += '\n| 种子／交换方案 | 正确行召回 % | 无正确候选错误接受 % | 有正确候选却接错 |\n| --- | ---: | ---: | ---: |\n'
    for seed,entry in diagnostic['seeds'].items():
        for name,v in entry['variants'].items():
            md += f'| {seed} / {name} | {100*v["positive_match_recall"]:.3f} | {100*v["absent_false_accept_rate"]:.3f} | {v["wrong_target_on_present_rows"]} |\n'
    md += '''
完整上下文头只有很小的正序净增；边／拒配交换的改善方向也不能在两种子间稳定保持，因此不挑选一个表中较好的混合方案替代主比较。

本批错误接受约八成发生在没有正确候选的行。仅调整排序无法在这些行中选回正确点，但仍可通过拒配减少错接；这不意味着候选缺失已被证明来自运动模型。下一项先用冻结原方案追溯“第一次缺失”：原始输入有没有该身份点、是否被门限／前一关联阶段排除、内部状态此前何时偏离。按证据再单独试观测更新或候选构造，暂不继续堆关联头和训练步数。

四个模型均按既定选型规则选中 600 步；本轮增加至 2000 步没有得到更低的选型损失。此结果不能推出所有更长训练都无效。
'''
    md += '''
## 工程故障与保留记录

首轮真实缓存的分组索引为 int32，CUDA 的 `scatter_reduce` 要求 int64，故在上下文组第 0 步整图验证时报错停止。已在聚合入口统一为 int64 并补测试；修复后按原数据、步数、种子、优化器、检查点与门槛完整重跑，未根据已看到的逐边对照调整研究条件。

协议保留修订前摘要和修订理由。`runs/graph-association/attempt01/` 保留初次启动的日志、退出状态、旧协议和已有逐边模型记录；完整训练位于 `models/`。工程失败不混作架构效果。

## 复现入口与验证

```bash
python experiments/graph_association/train.py --samples OLD_SAMPLES --sample-protocol experiments/competition_learning/protocol.json --prior-models OLD_MODELS --protocol experiments/graph_association/protocol.json --prior PARENT_MEMORY.pt --affinity OLD_AFFINITY.pt --out NEW_OUTPUT
python -m unittest tests.test_graph_association -v
python experiments/graph_association_report.py runs/graph-association
```

新测试覆盖：旧函数保持、整图／抽样一致、候选顺序与填充不变性、分组类型兼容、未知标签仅影响损失资格、联合训练冻结旧头，以及运行时的父模型代价尺度。训练前云端 PyTorch2.2 的 11 项前置测试通过。

同步后的团队仓库完整 101 项测试通过；首轮工程修复前后的逐边默认种子选中指标逐项精确一致。

队友只有仓库时，也可用上述报告入口从已提交的汇总 JSON 重建结果页，不需要训练权重或原始数据；原始模型复现仍须协议匹配的旧权重与样本。报告、诊断的完整数值和失败门槛均进入 Git。

云端证据目录：`/root/shared-nvme/b02/v2-graph-association-20261117`，完整运行 `source02/models02`，失败尝试 `source01/models`。旧权重与样本协议在入口校验。权重、原始点迹和回放不进入 Git。
'''
    md += f'\n完整模型阶段（含训练、验证与训练偏移拟合）约 {report["elapsed_seconds"]:.1f} 秒；CUDA 张量峰值约 {report["peak_cuda_bytes"]/2**30:.2f} GiB，不含首轮工程失败及云平台计费开销。协议 SHA256：`{report["protocol_sha256"]}`。\n'
    (target/'RESULTS_20261117.md').write_text(md,encoding='utf-8',newline='\n')
    (target/'RESULTS_20261117.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    (target/'DIAGNOSTIC_20261117.json').write_bytes(diagnostic_path.read_bytes())
    htmltable = '<div class="scroll"><table><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+escape(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B02 行列上下文联合关联</title><style>body{background:#101824;color:#e5edf7;font:16px/1.7 system-ui;margin:0}main{max-width:1350px;margin:auto;padding:24px}section{padding:20px;background:#192638;border:1px solid #34465d;border-radius:8px;margin:20px 0}h1{font-size:28px}h2{font-size:21px}a{color:#8ee7d2}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px;border-bottom:1px solid #34465d;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}.warning{color:#ffdc89}nav{display:flex;gap:20px;flex-wrap:wrap}</style><main><h1>行列上下文联合关联 · 2026-11-17</h1>'''
    html += f'<section><h2>预注册能力门槛：{verdict}</h2><p>逐边残差与一跳行列上下文，双种子，相同初始化／损失／预算，各最多2000步。以下全部为冻结历史上的局部指标。</p><p class="warning">原24个训练／8个内部验证场景，属于父模型拟合池且已反复开发，不是整套系统未见验证。原校准八文件、新工作验证16与正式封存29未动。</p></section>'
    html += '<section><h2>完整主比较</h2>'+htmltable+'<p>固定列接受量64417；偏移只用训练24文件拟合。存在正确候选时的错接、缺席时错误接受必须一起检查。</p></section><section><h2>每个种子的门槛</h2>'
    for seed,checks in report['comparisons'].items():
        html += '<p>'+seed+'：'+ '；'.join(escape(k)+('通过' if v else '未通过') for k,v in checks.items())+'</p>'
    html += f'</section><section><h2>追加诊断：排除一种猜测</h2><p>142443条正例行的一对一匹配容量缺口仅{audit["minimum_positive_rows_unsatisfied_by_one_to_one"]}条，不能用标签冲突解释主要退化。上下文头两种子在拒配前的正序净增仅3／9条；当前错误接受约八成发生在正确点不在候选列表时。下一项先追溯候选缺失，不能直接认定是运动模型问题。</p><a href="diagnostic.json">全部交换与标签可行性诊断</a></section><section><h2>证据与回放</h2><nav><a href="../../experiments/graph_association/RESULTS_20261117.md">完整报告</a><a href="models/report.json">全部训练记录</a><a href="../../experiments/graph_association/protocol.json">运行前协议与兼容修订</a>'
    if (root.parent/'competition-learning/index.html').exists():
        html += '<a href="../competition-learning/index.html">此前原二维整场回放</a>'
    html += '</nav>'
    html += '<p>'+('组件门槛通过，须继续检查整场。' if report['offline_passed'] else '按协议未进入整场扩展，没有本轮整场分数或新轨迹回放，不升级版本。')+'</p></section></main></html>'
    (root/'index.html').write_text(html,encoding='utf-8',newline='\n')
    print(json.dumps({'offline_passed':report['offline_passed'],'comparisons':report['comparisons'],'rows':rows},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    render(p.parse_args().root)
