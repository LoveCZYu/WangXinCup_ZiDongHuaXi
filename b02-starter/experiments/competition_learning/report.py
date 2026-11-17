"""Render head metrics and whole-scene research results as separate tables."""
import argparse
from html import escape
import json
from pathlib import Path


def scene_table(models, labels):
    rows = []
    for name, row in models.items():
        m = row['metrics']
        rows.append(f'<tr><th>{labels[name]}</th>'+''.join(f'<td>{v:.4f}</td>' for v in
            (m['weighted_total'], m['fragmentation_score'], m['continuous_score'], m['point_recall']))+
            f'<td>{m["clutter_outputs"]}</td><td>{row["identity"]["mixed_minority_truth_points"]}</td>'
            f'<td>{row["short_ids_1or2"]}</td><td>{row["strict_one_to_one_recall"]:.4f}</td></tr>')
    return '<div class="scroll"><table><tr><th>方案</th><th>本地总分</th><th>零碎度</th><th>连续</th><th>点命中 %</th><th>已知杂点</th><th>混轨少数身份点</th><th>一两点额外 ID</th><th>一对一影子 %</th></tr>'+''.join(rows)+'</table></div>'


def render(root):
    training = json.loads((root/'models/report.json').read_text(encoding='utf-8'))
    calibrated = json.loads((root/'reject-calibration.json').read_text(encoding='utf-8'))
    pilot = json.loads((root/'pilot-comparison.json').read_text(encoding='utf-8'))
    transfer = json.loads((root/'transfer-comparison.json').read_text(encoding='utf-8'))
    replication = json.loads((root/'replication-comparison.json').read_text(encoding='utf-8'))
    training_rows = []
    for name, model in training['models'].items():
        s = model['selected']
        training_rows.append(f'<tr><th>{escape(name)}</th><td>{s["step"]}</td><td>{s["fixed_coverage_column"]["wrong"]}</td>'
            f'<td>{100*s["positive_row_match_recall"]:.3f}</td><td>{100*s["absent_row_false_accept_rate"]:.3f}</td></tr>')
    calibration_rows = []
    for seed, row in calibrated['seeds'].items():
        b, c = row['binary_validation'], row['joint_validation_calibrated']
        calibration_rows.append(f'<tr><th>{seed}</th><td>{row["offset_fitted_on_training24_only"]:.6f}</td>'+
            ''.join(f'<td>{100*v:.3f}</td>' for v in (b['positive_match_recall'], c['positive_match_recall'],
                b['absent_false_accept_rate'], c['absent_false_accept_rate']))+'</tr>')
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>B02 新关联训练研究</title><style>body{margin:0;background:#101824;color:#e5edf7;font:16px/1.7 system-ui}main{max-width:1350px;margin:auto;padding:24px}section{padding:20px;background:#192638;border:1px solid #34465d;border-radius:8px;margin:20px 0}h1{font-size:28px}h2{font-size:21px}a{color:#8ee7d2}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px;border-bottom:1px solid #34465d;text-align:right;white-space:nowrap}th:first-child{text-align:left}.scroll{overflow:auto}.warning{color:#ffdc89}nav{display:flex;gap:20px;flex-wrap:wrap}</style>
<main><h1>跨 ID 竞争与拒绝训练研究 · 2026-11-17</h1>
<section><h2>有身份保持信号，仍为研究结果</h2><p>新模型已完成真实样本训练；联合目标未通过原离线门槛。后续工作点与整场对照是追加的研究诊断，没有覆盖原失败结论或替换冻结提交版。</p>
<p class="warning">24 个训练场景、8 个新头内部验证场景均来自父模型 fit64。底层模型及旧头选型见过这些数据；不能称为整套系统未见验证。原校准八文件、新工作验证 16、确认集与正式封存 29 本轮未读取。</p></section>
<section><h2>训练对照：相同数据与初始化</h2><p>605899 条明确历史身份行中，314710 条只有一个候选；保留拒绝样本，未知 −1 不当负例。二元与联合目标各两个固定种子、最多 1600 步；原 93 维判别函数保留，新增三个元数据维度从零权重开始。</p>
<p>以下是冻结历史上的局部指标，不是整场点命中率或官网成绩。固定接受量为 64417 个强争抢列。</p><div class="scroll"><table><tr><th>模型与种子</th><th>选中步数</th><th>列内误配</th><th>正确行匹配召回 %</th><th>无正确候选的错误接受 %</th></tr>'''
    html += ''.join(training_rows)+'</table></div><p class="warning">原门槛要求两个种子同时改善误配和拒绝，并保持正确行召回；本轮未通过。平均损失下降不能替代这些行为条件。</p></section>'
    html += '<section><h2>追加诊断：偏移只从训练数据拟合</h2><div class="scroll"><table><tr><th>种子</th><th>训练偏移</th><th>二元正确召回 %</th><th>校正联合正确召回 %</th><th>二元错误接受 %</th><th>校正联合错误接受 %</th></tr>'+''.join(calibration_rows)+'</table></div><p>没有用验证分数搜索偏移。此诊断排除含未知候选的行，分母与上表略有差异。它支持工作点解释部分退化，但正确候选存在时，校正联合的错误点选择仍更多。</p></section>'
    html += '<section><h2>内部验证八文件整场：绝对评分与拒绝接入</h2>'
    html += scene_table(pilot['models'], {'baseline':'原关联候选', 'binary_absolute':'新二元绝对评分',
        'joint_raw':'新联合原工作点', 'joint_calibrated':'新联合训练拟合工作点'})
    html += '<p>这是自由运行的全部轨迹，状态会随新判断改变。联合原工作点改善零碎度与混轨，但点命中下降；更保守的工作点又损失了更多命中与连续性。因此冻结行指标不能直接外推成整场收益。</p></section>'
    html += '<section><h2>同批整场：保留原规则，仅转移跨 ID 证据</h2>'
    html += scene_table(transfer['models'], {'baseline':'原关联候选', 'old_column':'旧身份头列内比较',
        'binary_column':'新二元头列内比较', 'joint_column':'新联合头列内比较（预定主比较）'})
    checks = '，'.join(escape(k)+('通过' if v else '未通过') for k,v in transfer['primary_checks'].items())
    html += f'<p>预先固定的研究检查：{checks}。单项结果不用于事后换赢家，仍不升级版本。</p>'
    html += '<p>“保留原规则”指候选、行内评分与 null 规则保持原状，只增加成熟 ID 争强点时的列内代价；全局匹配耦合仍可能改变点集合。所有表格的原候选是上阶段成熟方案，不是最初简单基线。</p></section>'
    html += '<section><h2>探索性二元列内信号：第二种子复核</h2>'
    html += scene_table(replication['seeds'], {'20261118':'默认种子 20261118','20261119':'第二种子 20261119'})
    html += '<p>两种子相对原候选同向改善，但第二个的零碎度增幅约 0.959，没有达到预设的至少 +1.0，不能称复核门槛全部通过。继续保留默认种子，不按某个分项选择种子。</p><p>相较旧头列内对照，新二元头只增加约 0.05～0.15 本地总分，混轨及严格影子归属也没有全面胜出。收益并非全部来自新训练；完整对照保留这些取舍。</p></section>'
    html += '<section><h2>原二维回放 · train_152</h2><nav><a href="train_152-baseline.html">原候选</a><a href="train_152-joint-raw.html">新联合绝对评分</a><a href="train_152-binary-column.html">新二元列内比较（探索）</a><a href="train_152-joint-column.html">新联合列内比较</a></nav><p>train_152 是内部验证名单首文件，整场对照前已固定；没有按表现挑选。回放评分表列出本批全部八文件成绩。与上轮不同场景的 85.28／84.31 不跨表比较。</p></section>'
    html += '<section><h2>证据</h2><nav><a href="models/report.json">原训练与失败门槛</a><a href="reject-calibration.json">训练拟合偏移诊断</a><a href="pilot-comparison.json">绝对接入完整对照</a><a href="transfer-comparison.json">列内接入完整对照</a><a href="replication-comparison.json">双种子复核</a><a href="../../experiments/competition_learning/RESULTS_20261117.md">解释、限制与下一步</a></nav><p>四次训练／验证合计约 40 秒，峰值 CUDA 分配约 0.64 GiB；CPU 采样及整场回放另计。结果仅为本地估分与内部开发证据，没有官网新成绩。</p></section></main></html>'
    (root/'index.html').write_text(html, encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    render(p.parse_args().root)
