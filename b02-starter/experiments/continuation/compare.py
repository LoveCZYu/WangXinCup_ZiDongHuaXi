"""Join replay evidence, identity cost, and genuine-target first-hit changes."""
import argparse
from collections import Counter
import csv
from html import escape
import json
from pathlib import Path

NAMES={'wide_shadow':'冻结宽容差对照','narrow_shadow':'窄准入＋窄评分',
       'gate_only':'只扩大准入','score_only':'只放宽评分归一化','memory_guard':'长期记忆更新检查',
       'learned-20261116':'接续模型 · 种子 20261116','learned_safe-20261116':'高精度接续模型 · 种子 20261116',
       'learned-20261117':'接续模型 · 种子 20261117','learned_safe-20261117':'高精度接续模型 · 种子 20261117',
       'fusion-20261116':'评分纠偏 · 种子 20261116','fusion-20261117':'评分纠偏 · 种子 20261117',
       'fusion_memory-20261116':'评分纠偏＋记忆更新检查'}
METRICS={'fragmentation_score':'零碎度','continuous_score':'连续跟踪','point_recall':'点命中率 %',
         'track_recall':'逐轨命中率 %','timeliness_score':'起批及时性','clutter_outputs':'杂点输出','weighted_total':'本地总分'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    root=parser.parse_args().root
    base=json.loads((root/'wide_shadow/report.json').read_text(encoding='utf-8'))
    baseline=next(iter(base['variants'].values()))['metrics']
    delays={}
    for path in root.glob('wide_shadow-*/train_*.json'):
        row=json.loads(path.read_text(encoding='utf-8'))
        delays[row['scene']]=row['birth_audit']['first_truth_delays']
    assert len(delays)==8
    report={};table=[]
    for mode,label in NAMES.items():
        path=root/mode/'report.json'
        if not path.exists():continue
        row=json.loads(path.read_text(encoding='utf-8'))
        metrics=next(iter(row['variants'].values()))['metrics']
        audit,changes=Counter(),Counter()
        scenes={}
        for path in root.glob(mode+'-*/train_*.json'):
            item=json.loads(path.read_text(encoding='utf-8'));scene=item['scene']
            current=item['birth_audit']['first_truth_delays'];prior=delays[scene]
            difference=[current[t]-d for t,d in prior.items() if t in current]
            check={'later':sum(d>0 for d in difference),'later_over_one':sum(d>1 for d in difference),
                   'earlier':sum(d<0 for d in difference),'lost':len(set(prior)-set(current))}
            changes.update(check);audit.update(item['birth_audit']['extra_id_lengths']);scenes[scene]=check
        assert len(scenes)==8,mode
        checks={'point_recall':metrics['point_recall']>=baseline['point_recall']-.1,
                'track_recall':metrics['track_recall']>=baseline['track_recall']-.1,
                'continuous':metrics['continuous_score']>=baseline['continuous_score']-.3,
                'timeliness':metrics['timeliness_score']>=baseline['timeliness_score']-.3,
                'clutter':metrics['clutter_outputs']<=1.02*baseline['clutter_outputs'],
                'total':metrics['weighted_total']>=baseline['weighted_total'],
                'identity_cost_reduced':row['identity']['mixed_minority_truth_points']<base['identity']['mixed_minority_truth_points'],
                'target70':metrics['fragmentation_score']>=70.}
        report[mode]={'name':label,'metrics':metrics,'identity':row['identity'],'plans':row['plans'],
                      'checks':checks,'extra_id_lengths':dict(audit),'extra_ids_1or2':audit['1']+audit['2'],
                      'first_hit_changes':dict(changes),'scenes_first_hit_changes':scenes}
        table.append([label,*[metrics[k] for k in METRICS],audit['1']+audit['2'],row['identity']['mixed_minority_truth_points'],changes['later_over_one']])
    (root/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    headings=['方案',*METRICS.values(),'一两点额外 ID','混轨少数身份点','新目标起批推迟超过一圈']
    with (root/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as target:
        writer=csv.writer(target);writer.writerow(headings);writer.writerows(table)
    rows=''.join('<tr>'+''.join('<td>'+escape(f'{v:.3f}' if isinstance(v,float) else str(v))+'</td>' for v in row)+'</tr>' for row in table)
    links=' · '.join(f'<a href="{escape(path.name,quote=True)}">{escape(path.stem)}</a>' for path in root.glob('train_*.html'))
    (root/'index.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>B02 接续判别与几何对照</title><style>body{{font:15px/1.6 system-ui;background:#f6f8fa;color:#192530;padding:24px}}main{{max-width:1500px;margin:auto}}table{{border-collapse:collapse;background:white}}td,th{{padding:10px;white-space:nowrap;border-bottom:1px solid #dce3e9;text-align:right}}th{{background:#e7eff5}}th:first-child,td:first-child{{text-align:left}}</style>
<main><h1>接续判别与几何对照</h1><p>八文件校准集本地估分。控制组零碎度 69.8870；封存测试、开发 30 文件和确认 16 文件均未使用。</p>
<p><a href="comparison.csv">分数表 CSV</a> · <a href="comparison.json">完整统计 JSON</a> · {links}</p>
<div style="overflow:auto"><table><thead><tr>{''.join('<th>'+escape(s)+'</th>' for s in headings)}</tr></thead><tbody>{rows}</tbody></table></div>
<p>模型的拟合池内验证 16 文件仅用于接续判别头的检查点和门槛选择，与保留的确认 16 文件不同。混轨少数身份点越少越好；它与 ID 数量一起检查。</p></main></html>''',encoding='utf-8')
    print(json.dumps({name:{'fragmentation':row['metrics']['fragmentation_score'],'total':row['metrics']['weighted_total'],
                            'minority':row['identity']['mixed_minority_truth_points'],'checks':row['checks']}
                      for name,row in report.items()},ensure_ascii=False))


if __name__=='__main__':main()
