"""Aggregate fitting-pool preflight counts without turning them into scores."""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.v2_association.run_stage1 import save, sha


def summarize(root):
    protocol_path = Path(__file__).with_name('preflight-protocol.json')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    scenes = {}
    totals = Counter()
    for path in sorted(root.glob('part-*/train_*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        assert row['protocol_sha256'] == sha(protocol_path)
        assert row['scene'] not in scenes
        scenes[row['scene']] = row
        totals.update(row['counts'])
    assert set(scenes) == set(protocol['fit_preflight8'])
    c = totals
    statuses = ('correct', 'clutter', 'other_target', 'unknown')
    assert sum(c['row_selected_' + k] for k in statuses) + c['null_with_correct_candidate'] + c['null_without_correct_candidate'] == c['anchored_recorded_rows']
    assert c['prefix_canonical_hits'] + c['prefix_canonical_misses'] == c['prefix_canonical_points']
    assert sum(c['prefix_miss_' + k] for k in ('recorded_edge', 'geometry_only', 'in_bank_no_recorded_edge', 'outside_banks')) == c['prefix_canonical_misses']
    ratios = {
        'anchor_present_candidate_coverage': c['anchor_candidate_rows'] / c['anchor_present_rows'],
        'wrong_target_without_correct_candidate_fraction': c['without_correct_candidate_other_target'] / c['row_selected_other_target'],
        'wrong_target_high_point_confidence_fraction': c['row_selected_other_target_enter'] / c['row_selected_other_target'],
        'canonical_miss_recorded_edge_fraction': c['prefix_miss_recorded_edge'] / c['prefix_canonical_misses'],
        'canonical_miss_outside_banks_fraction': c['prefix_miss_outside_banks'] / c['prefix_canonical_misses'],
    }
    report = {'format': 'b02-identity-preflight-results-v1', 'counts': dict(c), 'ratios': ratios,
        'scenes': scenes, 'protocol': protocol, 'provenance': json.loads((root/'provenance.json').read_text()),
        'sufficient_wrong_target_cases_for_design': c['row_selected_other_target'] >= 100,
        'new_training': False, 'independent_validation': False, 'sealed_test_used': False,
        'interpretation': ['High point confidence is not same-ID confidence.',
            'Wrong-target edges without the anchored truth in candidates motivate null/rejection supervision as well as candidate audit.',
            'Historical-canonical point misses can accumulate after an earlier ID loss; outside-bank point counts do not locate the first cause.',
            'No deployable score improvement or architecture ceiling is inferred.']}
    save(root/'report.json', report)
    rows = ''.join(f'<tr><th>{escape(name)}</th><td>{value:,}</td></tr>' for name, value in (
        ('有历史主 ID 参考的真目标点', c['prefix_canonical_points']),
        ('未交给历史主 ID 的点', c['prefix_canonical_misses']),
        ('上述点：历史主 ID 有记录候选边', c['prefix_miss_recorded_edge']),
        ('上述点：只有几何候选', c['prefix_miss_geometry_only']),
        ('上述点：调用前仍在集合，但无记录边', c['prefix_miss_in_bank_no_recorded_edge']),
        ('上述点：调用前已不在活跃／退役集合', c['prefix_miss_outside_banks'])))
    row_table = ''.join(f'<tr><th>{label}</th><td>{c["with_correct_candidate_"+status]:,}</td><td>{c["without_correct_candidate_"+status]:,}</td></tr>'
        for status, label in (('correct','匹配正确身份'),('other_target','匹配其他真目标'),('clutter','匹配已知杂点'),('unknown','匹配未知标签点')))
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>B02 身份竞争前置诊断</title>
<style>body{background:#101824;color:#e5edf7;font:16px/1.7 system-ui;margin:24px auto;max-width:1000px;padding:16px}a{color:#8ee7d2}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #34465d;text-align:right}th:first-child{text-align:left}.note{color:#ffdc89}</style>
<h1>身份竞争前置诊断 · 原拟合池八文件</h1><p class="note">没有训练、没有新模型成绩。以下是原 fit64 内的已见数据诊断，不能称作泛化验证。</p>'''
    html += '<h2>已有身份锚的记录行：排序与拒绝</h2><table><tr><th>内部匹配结果</th><th>存在正确候选</th><th>没有正确候选</th></tr>' + row_table + '</table>'
    html += f'<p>其他真目标的错误接受有 {c["row_selected_other_target"]:,} 次，其中 {100*ratios["wrong_target_without_correct_candidate_fraction"]:.1f}% 没有本身份的正确候选；{100*ratios["wrong_target_high_point_confidence_fraction"]:.1f}% 的点分数达到起始门槛。提升点分数阈值不能替代身份判别和拒绝。</p>'
    html += f'<p>实际可信记忆更新中，{c["trusted_update_other_target"]:,} 次写入其他真目标，{c["trusted_update_clutter"]:,} 次写入已知杂点；未知标签的更新单列，不能当作错误。</p>'
    html += '<h2>历史主 ID 参考：逐点连续性缺口</h2><table>' + rows + '</table><p class="note">这是点数，不是碎片产生次数。旧 ID 首次丢失后可连续产生许多“集合外”点，因此不能据此认定保留时间就是首因，或承诺延长保留会恢复这些点。</p>'
    html += '<p>下一步需回溯身份丢失的开端，再设计真实竞争与拒绝监督；不按此统计直接放大所有几何门或延长所有 ID 寿命。</p><p><a href="report.json">完整计数与协议</a> · <a href="../../experiments/identity_competition/RESULTS_20261117.md">解释与限制</a> · <a href="../v2-validation/index.html">新增 16 文件冻结验证与原二维回放</a></p></html>'
    (root/'index.html').write_text(html, encoding='utf-8')
    print(json.dumps({'ratios': ratios, 'scenes': len(scenes), 'wrong_target': c['row_selected_other_target']}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    summarize(parser.parse_args().root)
