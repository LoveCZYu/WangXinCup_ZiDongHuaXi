"""Build a local experiment index from downloaded evidence, without raw points."""
import argparse
import csv
from html import escape
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    rows=[]
    for path in sorted(a.run.glob('*/report.json')):
        report=json.loads(path.read_text(encoding='utf-8'))
        if 'metrics' not in report or not report.get('complete'):continue
        rows.append({'run':path.parent.name,'scope':','.join(report['scenes']),
                     **report['metrics'],'switches':report['identity'].get('switches',0),
                     'strategy':report.get('strategy','residual'),'scale':report['scale']})
    development=a.run/'dev04-parallel/report.json'
    dev=json.loads(development.read_text(encoding='utf-8')) if development.exists() else None
    export_rows=list(rows)
    if dev and dev.get('complete'):
        for name,value in dev['results'].items():
            export_rows.append({'run':'dev04-parallel/'+name,'scope':','.join(dev['scenes']),
                **value['metrics'],'switches':value['identity']['switches'],
                'strategy':'frozen_v1' if name=='v1' else 'rank_only','scale':1.})
    (a.run/'comparison.json').write_text(json.dumps(export_rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if export_rows:
        with (a.run/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(export_rows[0]));writer.writeheader();writer.writerows(export_rows)
    def number(value):return f'{value:.3f}' if isinstance(value,float) else str(value)
    keys=('weighted_total','point_recall','clutter_outputs','fragmentation_score','timeliness_score','continuous_score','switches')
    def table(items):
        body=''
        for row in items:
            href=escape(row.get('href',row['run']+'/report.json'),quote=True)
            body+='<tr><td><a href="'+href+'">'+escape(row['run'])+'</a></td>'
            body+=''.join('<td>'+number(row.get(k,'—'))+'</td>' for k in keys)+'</tr>'
        return '<div class="scroll"><table><thead><tr><th>版本／原始报告</th><th>本地总分</th><th>点命中 %</th><th>已知杂点</th><th>零碎子分</th><th>及时子分</th><th>连续子分</th><th>换号</th></tr></thead><tbody>'+body+'</tbody></table></div>'
    sections=''
    stage=a.run/'stage1-01/report.json'
    if stage.exists():
        d=json.loads(stage.read_text(encoding='utf-8'))
        items=[{'run':name,'href':'stage1-01/report.json',**v['metrics'],'switches':v['identity'].get('switches',0)} for name,v in d['results'].items()]
        sections+='<h2>八文件规则对照</h2><p>三个规则候选均未通过采用条件，v1 保持不变。</p>'+table(items)
    scopes=sorted({r['scope'] for r in rows},key=lambda s:(len(s.split(',')),s))
    for scope in scopes:
        sections+='<h2>'+escape(str(len(scope.split(','))))+' 文件学习对照</h2><p>'+escape(scope)+'</p>'
        sections+=table([r for r in rows if r['scope']==scope])
    if dev:
        d=dev
        if d.get('complete'):
            sections+='<h2>冻结后完整 30 文件开发验证</h2><p>两种子、检查点与策略在开发验证前固定。29 个独立保留文件未使用；未据开发结果调参。</p>'
            items=[{'run':name,'href':'dev04-parallel/report.json',**v['metrics'],'switches':v['identity']['switches']} for name,v in d['results'].items()]
            sections+=table(items)
    viewers=sorted(a.run.glob('*.html'))
    viewers=[p for p in viewers if p.name!='index.html']
    links='<p>'+ ' · '.join('<a href="'+escape(p.name,quote=True)+'">'+escape(p.stem)+'</a>' for p in viewers)+'</p>' if viewers else ''
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>B02 v2 实验对照</title><style>
body{font:15px/1.6 system-ui,sans-serif;background:#f2f5f8;color:#172e43;margin:0;padding:28px}
main{max-width:1320px;margin:auto}h1{margin-bottom:8px}h2{margin-top:34px}.notice{background:#fff5dc;border-left:4px solid #db9e2d;padding:16px}
.scroll{overflow:auto;background:white;border:1px solid #d9e2eb;border-radius:8px}table{border-collapse:collapse;width:100%;white-space:nowrap}
th,td{padding:10px 13px;text-align:right;border-bottom:1px solid #e7edf2}th:first-child,td:first-child{text-align:left}th{background:#e9f0f6}
a{color:#12639a}p{color:#4a5e70}.files{margin-top:30px}</style><main>
<h1>B02 v2 实验对照</h1><div class="notice">官网保底仍为 v1 的 75.24 分。此页为本地估算，不同场景范围须分开比较。
实验结果不会自动替换提交版；开发 30 文件须在候选固定后验证，独立保留的 29 文件不在本轮使用。</div>
'''+links+sections+'''<p class="files"><a href="comparison.csv">下载学习对照 CSV</a> · <a href="comparison.json">下载 JSON</a></p>
<p>完整曲线与配置见各训练目录 history.jsonl／manifest.json；点模型分数缓存的关联计时不代表提交性能。</p></main></html>'''
    (a.run/'index.html').write_text(text,encoding='utf-8')
    print(json.dumps({'index':str(a.run/'index.html'),'comparisons':len(export_rows),'calibration_scopes':len(scopes)},ensure_ascii=False))


if __name__=='__main__':main()
