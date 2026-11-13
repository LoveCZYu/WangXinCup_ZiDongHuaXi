"""Compare two frozen causal interventions on internal calibration only."""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
from statistics import mean
import sys
from time import monotonic, perf_counter


def identity_diagnostics(frames, outputs):
    last, owners, counts = {}, defaultdict(Counter), Counter()
    for (_, _, labels), result in zip(frames,outputs,strict=True):
        for point in result:
            label, batch = labels[point['PntID']], point['BatchID']
            if label <= 0:
                continue
            if label in last:
                counts['switches'] += last[label] != batch
            last[label] = batch
            owners[batch][label] += 1
    counts['truth_carrying_ids'] = len(owners)
    counts['mixed_ids'] = sum(len(c)>1 for c in owners.values())
    counts['mixed_minority_truth_points'] = sum(sum(c.values())-max(c.values()) for c in owners.values())
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project','data','steps-run','reference','out'):
        parser.add_argument(f'--{name}',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=660)
    args=parser.parse_args()
    args.project=args.project.resolve()
    sys.path[:0]=[str(args.project),str(args.project/'experiments/point_scorer'),
                   str(args.project/'experiments/point_scorer_expanded'),str(Path(__file__).resolve().parent)]
    import torch
    from b02.replay import _scans,open_scenes,replay_frames
    from b02.scorecard import build_scorecard,load_summary,write_scorecard
    from interventions import Tracker
    from model import HISTORY,PointScorer,point_features
    from experiments.association.compare_steps import DiscardTrace,save,sha256

    torch.set_num_threads(4)
    started=monotonic()
    args.out.mkdir(parents=True,exist_ok=False)
    steps=json.loads((args.steps_run/'manifest.json').read_text())
    split=json.loads((args.project/'splits/v1.json').read_text())
    scenes,seeds=steps['calibration'],steps['seeds']
    assert len(scenes)==8 and set(scenes)<=set(split['train'])
    assert not set(scenes)&set(steps['fit']+split['dev']+split['holdout'])
    assert sha256(args.data)==steps['data_sha256']
    assert sha256(Path(__file__).with_name('assignment.py'))==steps['source_sha256']['assignment.py']
    variants=('control','evidence_priority','confidence_velocity')
    manifest={'scenes':scenes,'seeds':seeds,'step':8000,'output_rule':{'enter':.50,'maintain':.05,'grace':3},
              'score_weight':.50,'variants':variants,'data_sha256':steps['data_sha256'],
              'interventions':{'evidence_priority':'Real-edge penalty 0.15*(1-min(prior strong hits,6)/6); strong >=0.50. Other association rules unchanged.',
                               'confidence_velocity':'Scale baseline velocity increment by min(1,point_score/0.50); position, hits, birth and output gate unchanged.'},
              'selection':'Require each seed recall >= control and clutter <= control; mean continuity and local total both improve. No dev evaluation.',
              'viewer_scenes':['train_032','train_085'],'soft_seconds':args.seconds,
              'source_sha256':{str(p.relative_to(args.project)):sha256(p) for p in
                 [Path(__file__).resolve(),Path(__file__).with_name('interventions.py'),Path(__file__).with_name('assignment.py'),
                  args.project/'b02/algorithm.py',args.project/'b02/replay.py',args.project/'b02/scorecard.py',
                  args.project/'experiments/point_scorer/model.py',args.project/'experiments/point_scorer_expanded/policy.py']},
              'checkpoint_sha256':{str(seed):sha256(args.steps_run/f'{seed}-step8000.pt') for seed in seeds}}
    save(args.out/'manifest.json',manifest)
    report={'complete':False,'by_seed':{},'control_reproduced':{}}

    def budget():
        if monotonic()-started>args.seconds:
            raise TimeoutError('Intervention comparison soft budget reached')

    def log(**fields):
        row={'seconds':monotonic()-started,**fields}
        print(json.dumps(row),flush=True)
        with (args.out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')

    try:
        with open_scenes(args.data) as (_,opener):
            for seed in seeds:
                saved=torch.load(args.steps_run/f'{seed}-step8000.pt',map_location='cpu',weights_only=True)
                assert saved['seed']==seed and saved['step']==8000
                assert json.loads(json.dumps(saved['manifest']))==steps
                model=PointScorer().cuda().eval()
                model.load_state_dict(saved['model'])
                expected=load_summary(args.reference/f'{seed}-enter_0.50_maintain_0.050')
                summaries={v:{'kind':'local_diagnostics','config':{'variant':v,'seed':seed,'step':8000},'scenes':{}} for v in variants}
                diagnostics={v:{} for v in variants}
                for variant in variants:(args.out/f'{seed}-{variant}').mkdir()
                for scene in scenes:
                    budget()
                    with opener(scene) as source:frames=list(_scans(source))
                    trackers={v:Tracker(v) for v in variants}
                    caches={v:[] for v in variants}
                    times={v:[] for v in variants}
                    history=deque(maxlen=HISTORY-1)
                    for circle,points,_ in frames:
                        budget()
                        start=perf_counter()
                        with torch.no_grad():
                            scores=model(torch.as_tensor(point_features(points,list(history)),device='cuda')).sigmoid().cpu().tolist()
                        history.append(points)
                        feature_ms=(perf_counter()-start)*1000
                        for variant,tracker in trackers.items():
                            start=perf_counter()
                            caches[variant].append(tracker(points,scores))
                            times[variant].append(feature_ms+(perf_counter()-start)*1000)
                    for variant in variants:
                        out=args.out/f'{seed}-{variant}'
                        iterator=iter(caches[variant])
                        sink=((out/f'{scene}.jsonl').open('w',encoding='utf-8')
                              if seed==seeds[0] and scene in manifest['viewer_scenes'] else DiscardTrace())
                        class MeasuredTrace:
                            def write(self,line):
                                row=json.loads(line)
                                row['milliseconds']=round(times[variant][row['circle']],4)
                                sink.write(json.dumps(row)+'\n')
                        try:
                            result=replay_frames(iter(frames),MeasuredTrace(),lambda points,state:next(iterator))
                        finally:
                            if hasattr(sink,'close'):sink.close()
                        ordered=sorted(times[variant])
                        result['p95_handler_ms']=round(ordered[(95*len(ordered)-1)//100],4)
                        result['max_handler_ms']=round(ordered[-1],4)
                        if variant=='control':
                            for key,value in json.loads(json.dumps(result)).items():
                                if key not in ('p95_handler_ms','max_handler_ms'):
                                    assert value==expected['scenes'][scene][key],(seed,scene,key)
                        summaries[variant]['scenes'][scene]=result
                        diagnostics[variant][scene]=identity_diagnostics(frames,caches[variant])
                        save(out/'summary.json',summaries[variant])
                        save(out/'identity-diagnostics.json',diagnostics[variant])
                    log(event='scene',seed=seed,scene=scene,completed=len(summaries['control']['scenes']),
                        variants={v:{'hits':summaries[v]['scenes'][scene]['matched_target_points'],
                                     'clutter':summaries[v]['scenes'][scene]['clutter_outputs'],
                                     'switches':diagnostics[v][scene].get('switches',0)} for v in variants})
                report['by_seed'][seed]={}
                for variant in variants:
                    out=args.out/f'{seed}-{variant}'
                    card=build_scorecard(summaries[variant],out.name,expected,'frozen-8000-050-005')
                    write_scorecard(out,card)
                    total=Counter()
                    for row in diagnostics[variant].values():total.update(row)
                    report['by_seed'][seed][variant]={'metrics':card['overall']['values'],'identity':dict(total)}
                report['control_reproduced'][seed]=True
                save(args.out/'report.json',report)
                del model,caches,trackers,saved
        keys=tuple(next(iter(report['by_seed'].values()))['control']['metrics'])
        identity_keys=('switches','truth_carrying_ids','mixed_ids','mixed_minority_truth_points')
        report['means']={v:{'metrics':{k:mean(rows[v]['metrics'][k] for rows in report['by_seed'].values()) for k in keys},
                            'identity':{k:mean(rows[v]['identity'].get(k,0) for rows in report['by_seed'].values()) for k in identity_keys}}
                         for v in variants}
        report['checks']={}
        for variant in variants[1:]:
            report['checks'][variant]={
                'each_seed_recall_not_lower':all(rows[variant]['metrics']['point_recall']>=rows['control']['metrics']['point_recall'] for rows in report['by_seed'].values()),
                'each_seed_clutter_not_higher':all(rows[variant]['metrics']['clutter_outputs']<=rows['control']['metrics']['clutter_outputs'] for rows in report['by_seed'].values()),
                'mean_continuity_higher':report['means'][variant]['metrics']['continuous_score']>report['means']['control']['metrics']['continuous_score'],
                'mean_total_higher':report['means'][variant]['metrics']['weighted_total']>report['means']['control']['metrics']['weighted_total']}
        eligible=[v for v,checks in report['checks'].items() if all(checks.values())]
        report['selected']=max(eligible,key=lambda v:report['means'][v]['metrics']['weighted_total'],default=None)
        report['viewer_variant']=max(variants[1:],key=lambda v:report['means'][v]['metrics']['weighted_total'])
        report['complete']=True
    except TimeoutError as error:
        report['budget_stop']=str(error)
    finally:
        report['elapsed_seconds']=monotonic()-started
        save(args.out/'report.json',report)
    log(event='complete',complete=report['complete'],means=report.get('means'),selected=report.get('selected'))


if __name__=='__main__':
    main()
