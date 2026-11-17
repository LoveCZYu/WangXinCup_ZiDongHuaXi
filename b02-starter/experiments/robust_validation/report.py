"""Render the frozen comparison separately from truth-assisted diagnostics."""
import argparse
from html import escape
import json
from pathlib import Path


LABELS = {'wide': '宽方案对照', 'memory': '单记忆检查', 'candidate': '接续评分＋记忆检查'}
DIAGNOSTICS = {
    'frozen_candidate': '实际候选输出', 'actual_graph': '真值辅助：已有候选边',
    'relaxed_geometry': '真值辅助：放宽几何边', 'perfect_identity': '真值辅助：理想身份',
    'clean_known_clutter': '真值辅助：删除已知杂点',
    'perfect_identity_clean': '真值辅助：理想身份＋删除已知杂点',
}


def render(root):
    validation = json.loads((root / 'validation-report.json').read_text(encoding='utf-8'))
    diagnosis = json.loads((root / 'diagnostic-report.json').read_text(encoding='utf-8'))
    rows = []
    for name, row in validation['models'].items():
        m = row['metrics']
        rows.append('<tr><th>' + escape(LABELS[name]) + '</th>' + ''.join(
            f'<td>{v:.3f}</td>' for v in (m['weighted_total'], m['fragmentation_score'],
                m['continuous_score'], m['point_recall'], m['timeliness_score'])) +
            f"<td>{m['clutter_outputs']:,}</td><td>{row['identity']['mixed_minority_truth_points']:,}</td>"
            f"<td>{row['short_ids_1or2']}</td></tr>")
    diagnostic_rows = []
    for name, row in diagnosis['variants'].items():
        m = row['metrics']
        diagnostic_rows.append(f'<tr><th>{DIAGNOSTICS[name]}</th><td>{m["weighted_total"]:.3f}</td>'
            f'<td>{m["fragmentation_score"]:.3f}</td><td>{m["continuous_score"]:.3f}</td></tr>')
    intervals = []
    for name, label in [('weighted_total', '本地总分差'), ('fragmentation_score', '零碎度子分差'),
                        ('continuous_score', '连续子分差'), ('mixed_rate_percent', '混轨少数身份点率差（百分点）')]:
        ci = validation['paired_scene_bootstrap_95'][name]
        intervals.append(f'<tr><th>{label}</th><td>[{ci["lower"]:+.4f}, {ci["upper"]:+.4f}]</td></tr>')
    scene_rows = []
    for name, row in validation['models']['candidate']['scenes'].items():
        scene_rows.append(f'<tr><th>{escape(name)}</th>' + ''.join(f'<td>{row[k]:+.4f}</td>' for k in
            ('weighted_total', 'fragmentation_score', 'continuous_score', 'point_recall')) + '</tr>')
    decision = {'support': '支持预定假设', 'regression': '存在预定判据内的退化',
                'insufficient_evidence': '证据不足：尚未证明稳定改善'}[validation['decision']]
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>B02 冻结验证与诊断</title>
<style>body{margin:0;background:#101824;color:#e5edf7;font:16px/1.7 system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:24px}section{background:#192638;border:1px solid #34465d;padding:20px;margin:20px 0;border-radius:8px}
h1{font-size:28px}h2{font-size:21px}a{color:#8ee7d2}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:9px;border-bottom:1px solid #34465d;text-align:right;white-space:nowrap}th:first-child{text-align:left}
.scroll{overflow:auto}.note{color:#b8c8da}.verdict{color:#ffdc89}nav{display:flex;gap:20px;flex-wrap:wrap}</style>
<main><h1>冻结验证与错误归因 · 2026-11-17</h1>'''
    html += f'<section><h2 class="verdict">{decision}</h2><p>三个方案、权重、主比较和采用条件均在新场景结果出现前冻结。本轮没有训练或调整阈值。</p>'
    html += '<p>下表是当前路线此前未见的 16 文件工作验证，本地估分不是官网成绩。候选减少约 8% 的混轨少数身份点，但置信区间跨零，短片段更多；全部保护条件通过，仍未满足完整采用判据。保留为 v2 实验。</p></section>'
    html += '<section><h2>新增 16 文件：实际可运行方案</h2><div class="scroll"><table><tr><th>方案</th><th>总分</th><th>零碎度</th><th>连续</th><th>点命中率 %</th><th>起批子分</th><th>已知杂点</th><th>混轨少数身份点</th><th>一两点额外 ID</th></tr>'
    html += ''.join(rows) + '</table></div><p class="note">宽方案是上一轮的成熟对照，不是最初 60 多分的简单基线。混轨少数身份点指同一输出 ID 的非主真目标点数；该项与额外 ID 数不是同一概念。</p>'
    html += '<h3>候选减宽方案：配对场景自助重采样 95% 区间</h3><table>' + ''.join(intervals) + '</table><p class="note">10,000 次重采样，固定种子。仅 16 场景，且无法保证模拟场景彼此独立，区间不是严格泛化保证。</p></section>'
    html += '<section><h2>原二维查看器</h2><nav><a href="train_011-candidate.html">候选回放 train_011</a><a href="train_011-wide.html">同场景宽方案</a></nav><p class="note">train_011 是看结果前按固定列表选出的首个预览场景，没有按表现挑选。回放评分表同时列出这 16 文件的整批成绩。单记忆方案保留在上表和结果文件。</p></section>'
    html += '<section><h2>新增 16 文件：候选相对宽方案逐场景差</h2><div class="scroll"><table><tr><th>场景</th><th>总分差</th><th>零碎度差</th><th>连续差</th><th>点命中百分点差</th></tr>' + ''.join(scene_rows) + '</table></div></section>'
    html += '<section><h2>旧八文件校准：真值辅助诊断</h2><p class="verdict">以下修复使用真实身份及事后信息，不是可提交模型，不与上面的新场景成绩混排，也不是架构上限。</p><table><tr><th>诊断</th><th>总分</th><th>零碎度</th><th>连续</th></tr>' + ''.join(diagnostic_rows) + '</table>'
    html += '<p>保持实际输出点不变，以事后主 ID 在当前候选边内重分配身份，仍受到原始状态前缀与同圈 ID 唯一性约束。扩大候选仅增加有限诊断收益，支持进一步研究已有 ID 的身份竞争，但不能证明学习器能实现这些诊断分数。</p></section>'
    html += '<section><h2>证据与数据角色</h2><nav><a href="validation-report.json">完整验证 JSON</a><a href="validation-comparison.csv">三方案 CSV</a><a href="diagnostic-report.json">旧校准诊断 JSON</a><a href="../../experiments/robust_validation/RESULTS_20261117.md">结论与后续假设</a></nav><p>正式封存 29 文件未读取。新增 16 文件现在是已评估工作验证，后续不能再称盲测；另 36 个候选只做过无标签输入前缀筛查。筛查未证明模拟轨迹同源关系。后羿不属于当前权重谱系，其另一条路线的使用情况未知，不阻断本轮。</p></section></main></html>'
    (root / 'index.html').write_text(html, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    render(parser.parse_args().root)
