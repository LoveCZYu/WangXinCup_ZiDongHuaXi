"""One bounded output-gate calibration before freezing the release candidate."""
import argparse
from collections import deque
import json
from pathlib import Path
from statistics import mean
import sys
from time import monotonic,perf_counter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project','data','steps-run','reference','combination-run','out'):
        p.add_argument(f'--{name}',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=330)
    a=p.parse_args();a.project=a.project.resolve()
    sys.path[:0]=[str(a.project),str(a.project/'experiments/point_scorer'),
                   str(a.project/'experiments/point_scorer_expanded'),str(Path(__file__).resolve().parent)]
    import torch
    from b02.replay import _scans,open_scenes,replay_frames
    from b02.scorecard import build_scorecard,load_summary,write_scorecard
    from model import PointScorer,HISTORY,point_features
    from policy import Policy,OutputGate
    from interventions import Tracker
    from experiments.association.compare_steps import DiscardTrace,save,sha256
    torch.set_num_threads(4);started=monotonic();a.out.mkdir(parents=True,exist_ok=False)
    steps=json.loads((a.steps_run/'manifest.json').read_text())
    split=json.loads((a.project/'splits/v1.json').read_text())
    prior=json.loads((a.combination_run/'report.json').read_text())
    assert prior['complete'] and all(prior['control_reproduced'].values())
    scenes,seeds=steps['calibration'],steps['seeds']
    assert len(scenes)==8 and set(scenes)<=set(split['train'])
    assert not set(scenes)&set(steps['fit']+split['dev']+split['holdout'])
    assert sha256(a.data)==steps['data_sha256']
    policies=[Policy(f'e{int(round(e*1000)):03d}_m{int(round(m*1000)):03d}',e,m,3)
              for e,m in ((.50,.05),(.50,.075),(.50,.10),(.525,.05),(.475,.075),(.475,.10))]
    manifest={'scenes':scenes,'seeds':seeds,'step':8000,'variant':'combined','policies':{r.name:vars(r) for r in policies},
              'data_sha256':steps['data_sha256'],'recall_tolerance_pp':.10,'soft_seconds':a.seconds,
              'selection':'Each seed clutter <= original8000 and recall loss <=0.10 pp; mean continuity and total improve. Prefer strict no-recall-loss, then mean total. Freeze before dev.',
              'source_sha256':{str(path.relative_to(a.project)):sha256(path) for path in
                 [Path(__file__).resolve(),Path(__file__).with_name('interventions.py'),Path(__file__).with_name('assignment.py'),
                  a.project/'b02/algorithm.py',a.project/'experiments/point_scorer/model.py',
                  a.project/'experiments/point_scorer_expanded/policy.py']},
              'checkpoint_sha256':{str(s):sha256(a.steps_run/f'{s}-step8000.pt') for s in seeds}}
    save(a.out/'manifest.json',manifest)
    report={'complete':False,'by_seed':{},'references':{},'original_combination_reproduced':{}}
    def budget():
        if monotonic()-started>a.seconds:raise TimeoutError('Final calibration soft budget reached')
    def log(**fields):
        row={'seconds':monotonic()-started,**fields};print(json.dumps(row),flush=True)
        with (a.out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    try:
        with open_scenes(a.data) as (_,opener):
            for seed in seeds:
                saved=torch.load(a.steps_run/f'{seed}-step8000.pt',map_location='cpu',weights_only=True)
                assert saved['step']==8000 and saved['seed']==seed
                model=PointScorer().cuda().eval();model.load_state_dict(saved['model'])
                reference=load_summary(a.reference/f'{seed}-enter_0.50_maintain_0.050')
                expected=load_summary(a.combination_run/f'{seed}-combined')
                report['references'][seed]=build_scorecard(reference,'original8000')['overall']['values']
                summaries={r.name:{'kind':'local_diagnostics','config':{'seed':seed,'variant':'combined','policy':vars(r)},'scenes':{}} for r in policies}
                for r in policies:(a.out/f'{seed}-{r.name}').mkdir()
                for scene in scenes:
                    budget()
                    with opener(scene) as src:frames=list(_scans(src))
                    history=deque(maxlen=HISTORY-1);tracker=Tracker('combined');associated=[];durations=[]
                    for circle,points,_ in frames:
                        budget();start=perf_counter()
                        with torch.no_grad():scores=model(torch.as_tensor(point_features(points,list(history)),device='cuda')).sigmoid().cpu().tolist()
                        history.append(points);tracker(points,scores)
                        associated.append((tracker.last_raw,dict(zip((x['PntID'] for x in points),scores))))
                        durations.append((perf_counter()-start)*1000)
                    for rule in policies:
                        gate=OutputGate(rule);iterator=iter(zip(frames,associated));out=a.out/f'{seed}-{rule.name}'
                        sink=((out/f'{scene}.jsonl').open('w',encoding='utf-8') if seed==seeds[0] and scene in ('train_032','train_085') else DiscardTrace())
                        def handler(points,state):
                            (circle,_,_),(raw,scores)=next(iterator)
                            return gate.apply(raw,scores,circle)
                        class Trace:
                            def write(self,line):
                                row=json.loads(line);row['milliseconds']=round(durations[row['circle']],4)
                                sink.write(json.dumps(row)+'\n')
                        try:observed=replay_frames(iter(frames),Trace(),handler)
                        finally:
                            if hasattr(sink,'close'):sink.close()
                        times=sorted(durations);observed['p95_handler_ms']=round(times[(95*len(times)-1)//100],4);observed['max_handler_ms']=round(times[-1],4)
                        if rule==policies[0]:
                            assert all(v==expected['scenes'][scene][k] for k,v in json.loads(json.dumps(observed)).items() if k not in ('p95_handler_ms','max_handler_ms')),(seed,scene)
                        summaries[rule.name]['scenes'][scene]=observed
                        save(out/'summary.json',summaries[rule.name])
                    log(event='scene',seed=seed,scene=scene)
                report['by_seed'][seed]={}
                for rule in policies:
                    out=a.out/f'{seed}-{rule.name}';card=build_scorecard(summaries[rule.name],out.name,reference,'original8000')
                    write_scorecard(out,card);report['by_seed'][seed][rule.name]=card['overall']['values']
                report['original_combination_reproduced'][seed]=True;save(a.out/'report.json',report)
        keys=tuple(next(iter(report['references'].values())))
        report['reference_means']={k:mean(row[k] for row in report['references'].values()) for k in keys}
        report['means']={r.name:{k:mean(rows[r.name][k] for rows in report['by_seed'].values()) for k in keys} for r in policies}
        report['checks']={}
        for rule in policies:
            rows=[(report['by_seed'][s][rule.name],report['references'][s]) for s in seeds]
            avg=report['means'][rule.name];ref=report['reference_means']
            report['checks'][rule.name]={'each_seed_clutter_not_higher':all(v['clutter_outputs']<=b['clutter_outputs'] for v,b in rows),
                'each_seed_recall_within_010pp':all(v['point_recall']>=b['point_recall']-.10 for v,b in rows),
                'mean_continuity_higher':avg['continuous_score']>ref['continuous_score'],
                'mean_total_higher':avg['weighted_total']>ref['weighted_total']}
            report['checks'][rule.name]['strict_recall_not_lower']=all(v['point_recall']>=b['point_recall'] for v,b in rows)
        eligible=[name for name,checks in report['checks'].items() if all(v for k,v in checks.items() if k!='strict_recall_not_lower')]
        selected=max(eligible,key=lambda name:(report['checks'][name]['strict_recall_not_lower'],report['means'][name]['weighted_total']),default=None)
        report['selected']=selected
        selection={'frozen':True,'variant':'combined' if selected else 'control','policy':vars(next(r for r in policies if r.name==selected)) if selected else {'name':'original8000','enter':.5,'maintain':.05,'grace':3},
                   'calibration_selected':selected,'seed_for_shared_runtime':seeds[0],'step':8000,'data_sha256':steps['data_sha256'],
                   'reason':'Selected once on calibration before dev; no dev-based tuning.'}
        save(a.out/'selection.json',selection);report['complete']=True
    except TimeoutError as error:report['budget_stop']=str(error)
    finally:report['elapsed_seconds']=monotonic()-started;save(a.out/'report.json',report)
    log(event='complete',complete=report['complete'],selected=report.get('selected'),means=report.get('means'))


if __name__=='__main__':main()
