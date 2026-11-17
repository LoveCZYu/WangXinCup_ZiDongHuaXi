"""Present onset mechanisms and complete four-way results without promotion."""
import argparse
from html import escape
import json
from pathlib import Path

LABELS = {'baseline': '原关联候选（本轮对照）', 'identity': '仅跨 ID 身份证据',
          'volume': '仅宽门体积代价', 'identity_volume': '两者组合（预定主比较）'}


def render(root):
    report = json.loads((root/'fitting-comparison.json').read_text(encoding='utf-8'))
    gap = json.loads((root/'training-gap.json').read_text(encoding='utf-8'))
    rows = []
    for name, row in report['models'].items():
        m = row['metrics']
        values = [m['weighted_total'], m['fragmentation_score'], m['continuous_score'], m['point_recall']]
        rows.append(f'<tr><th>{LABELS[name]}</th>' + ''.join(f'<td>{v:.4f}</td>' for v in values) +
            f'<td>{m["clutter_outputs"]}</td><td>{row["identity"]["mixed_minority_truth_points"]}</td>'
            f'<td>{row["short_ids_1or2"]}</td><td>{row["strict_one_to_one_recall"]:.4f}</td></tr>')
    scenes = ''.join('<tr><th>'+escape(name)+'</th>'+''.join(f'<td>{delta[k]:+.4f}</td>' for k in
        ('weighted_total', 'fragmentation_score', 'continuous_score', 'point_recall'))+'</tr>'
        for name, delta in report['models']['identity_volume']['scenes'].items())
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>B02 首次换号研究</title><style>body{margin:0;background:#101824;color:#e5edf7;font:16px/1.7 system-ui}main{max-width:1300px;margin:auto;padding:24px}section{padding:20px;background:#192638;border:1px solid #34465d;border-radius:8px;margin:20px 0}h1{font-size:28px}h2{font-size:21px}a{color:#8ee7d2}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px;border-bottom:1px solid #34465d;text-align:right;white-space:nowrap}th:first-child{text-align:left}.scroll{overflow:auto}.warning{color:#ffdc89}nav{display:flex;gap:20px;flex-wrap:wrap}</style>
<main><h1>首次换号研究 · 2026-11-17</h1><section><h2>已复现机制，尚未通过扩大实验门槛</h2>
<p>两个真实换号案例可重复重放；其中一个可缩减到两条已有轨迹和一个点。组合修正能修复这两个案例，局部输出点集合不变。</p>
<p class="warning">完整八文件的零碎度增加 0.990993，未达到事先固定的至少 +1.0。其余聚合保护条件通过，仍按协议停止，不扩大到校准或新工作验证。不能四舍五入成通过，也不能据此认定方向无效。</p></section>
<section><h2>找到的能力限制</h2><p>当前主关联器按轨迹行内排序；每条轨迹只有一个候选时，学习修正为零。它没有直接学习“同一点该归哪个 ID”，也没有训练当前 null 拒绝输出层。</p>'''
    html += f'<p>实际训练元数据：{gap["fit_counts"]["all_rows"]:,} 条记录中保留 {gap["fit_counts"]["rank_rows"]:,} 条排序行（{100*gap["used_row_fraction"]:.2f}%）；单候选及不匹配目标行被排除。null 最后层权重和偏置绝对值和均为 0。</p>'
    html += '<p>暂失配 ID 扩大门后，归一化距离会被折扣。门体积代价能修复一个案例，却不证明它适合所有真实失配；当前门也不是已校准的噪声方差。</p></section>'
    html += '<section><h2>原拟合池八文件：四格完整对照</h2><p class="warning">已见数据，本地估分，不是独立验证或官网成绩。没有训练新模型。</p><div class="scroll"><table><tr><th>方案</th><th>总分</th><th>零碎度</th><th>连续</th><th>点命中 %</th><th>已知杂点</th><th>混轨少数身份点</th><th>一两点额外 ID</th><th>一对一影子命中 %</th></tr>'
    html += ''.join(rows) + '</table></div><p>“原关联候选”是上阶段的成熟候选，不是最初 60 多分的简单基线。影子归属不是官方未公开的匹配脚本。</p>'
    html += '<p>组合在零碎度上略高于仅身份方案，但总分仅多约 0.0012，连续性和一对一影子归属反而较低。单独体积代价也增加了混轨少数身份点。因此不能为了让两个演示都通过，就直接把该规则合入正式方案。</p></section>'
    html += '<section><h2>原二维查看器 · train_093</h2><nav><a href="train_093-baseline.html">原候选回放</a><a href="train_093-identity_volume.html">组合研究回放</a></nav><p>此场景在整场实验前固定，用于观察第 47、78 圈附近的真实换号。整场干预会改变后续状态，不能把保存单步状态的反事实等同于完整重跑。</p></section>'
    html += '<section><h2>预定主比较逐场景差</h2><div class="scroll"><table><tr><th>场景</th><th>总分差</th><th>零碎度差</th><th>连续差</th><th>点命中百分点差</th></tr>' + scenes + '</table></div></section>'
    html += '<section><h2>可复核证据与下一步</h2><nav><a href="fitting-comparison.json">完整四格 JSON</a><a href="training-gap.json">实际权重与训练覆盖核对</a><a href="../../experiments/identity_competition/ONSET_RESEARCH_20261117.md">原因与方法报告</a><a href="../v2-validation/index.html">上一轮新增 16 文件验证</a></nav><p>下一条结构路线是从真实竞争中训练跨 ID 关系与不匹配项，而不是仅增加旧排序模型的步数。先用拟合池内部场景验证身份判别，再接整场。正式 29 文件未读取，新增 16 工作验证本轮没有重跑；冻结提交版未替换，仍不称 v3。</p></section></main></html>'
    (root/'index.html').write_text(html, encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    render(p.parse_args().root)
