"""Validate one frozen profile on all dev files using the shared runtime handler."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys
from time import monotonic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project','data','steps-run','calibration-run','reference-run','baseline','profile','decision','out'):
        p.add_argument(f'--{name}',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=720)
    a=p.parse_args();a.project=a.project.resolve();sys.path.insert(0,str(a.project))
    import torch
    from b02.point_tracker import Handler,load_tracker,sha256
    from b02.replay import open_scenes,replay_scene
    from b02.scorecard import build_scorecard,load_summary,write_scorecard
    from experiments.point_scorer.model import PointScorer
    from experiments.point_scorer_expanded.policy import Policy
    from experiments.association.compare_steps import DiscardTrace,save
    torch.set_num_threads(4);started=monotonic();a.out.mkdir(parents=True,exist_ok=False)
    split=json.loads((a.project/'splits/v1.json').read_text());scenes=split['dev']
    steps=json.loads((a.steps_run/'manifest.json').read_text());seeds=steps['seeds']
    decision=json.loads(a.decision.read_text());profile=json.loads(a.profile.read_text())
    assert decision['frozen_before_dev'] and sha256(a.profile)==decision['profile_sha256']
    assert len(scenes)==30 and not set(scenes)&set(steps['fit']+steps['calibration']+split['holdout'])
    assert sha256(a.data)==steps['data_sha256']
    baseline=load_summary(a.baseline);assert set(baseline['scenes'])==set(scenes)
    references={s:load_summary(a.reference_run/f'{s}-step8000-maintain005') for s in seeds}
    assert all(set(row['scenes'])==set(scenes) for row in references.values())
    sources=[a.project/'b02/point_tracker.py',a.project/'b02/algorithm.py',a.project/'b02/replay.py',
             a.project/'experiments/association/interventions.py',a.project/'experiments/association/assignment.py',
             a.project/'experiments/point_scorer/model.py',a.project/'experiments/point_scorer_expanded/policy.py',Path(__file__).resolve()]
    manifest={'dev':scenes,'seeds':seeds,'profile':profile,'profile_sha256':sha256(a.profile),'decision':decision,
              'data_sha256':steps['data_sha256'],'soft_seconds':a.seconds,'selection':'Single profile frozen before dev, no dev tuning.',
              'source_sha256':{str(path.relative_to(a.project)):sha256(path) for path in sources},
              'checkpoint_sha256':{str(s):sha256(a.steps_run/f'{s}-step8000.pt') for s in seeds}}
    save(a.out/'manifest.json',manifest);save(a.out/'frozen-profile.json',profile)
    report={'complete':False,'by_seed':{},'reference_reproduced':{},'candidate_calibration_reproduced':{},
            'baseline':build_scorecard(baseline,'original-baseline')['overall']['values']}
    def budget():
        if monotonic()-started>a.seconds:raise TimeoutError('Release dev validation soft budget reached')
    def log(**fields):
        row={'seconds':monotonic()-started,**fields};print(json.dumps(row),flush=True)
        with (a.out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    def check(observed,expected):
        assert all(v==expected[k] for k,v in json.loads(json.dumps(observed)).items() if k not in ('p95_handler_ms','max_handler_ms'))
    try:
        with open_scenes(a.data) as (_,opener):
            for seed in seeds:
                saved=torch.load(a.steps_run/f'{seed}-step8000.pt',map_location='cpu',weights_only=True)
                assert saved['seed']==seed and saved['step']==8000
                if seed==profile['seed']:
                    handler,metadata=load_tracker(a.profile,device='cuda');model=handler.model
                    assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in saved['model'].items())
                else:
                    model=PointScorer().cuda().eval();model.load_state_dict(saved['model'])
                    rule=profile['output_rule'];handler=Handler(model,profile['variant'],Policy(profile['name'],rule['enter'],rule['maintain'],rule['grace']))
                control=Handler(model,'control',Policy('original8000',.50,.05,3))
                with opener('train_015') as source:observed=replay_scene(source,DiscardTrace(),control)
                check(observed,references[seed]['scenes']['train_015']);report['reference_reproduced'][seed]=True
                cal=load_summary(a.calibration_run/f"{seed}-{decision['policy']}")
                with opener('train_032') as source:observed=replay_scene(source,DiscardTrace(),handler)
                check(observed,cal['scenes']['train_032']);report['candidate_calibration_reproduced'][seed]=True
                out=a.out/f'{seed}-candidate';out.mkdir()
                summary={'kind':'local_diagnostics','config':{'seed':seed,'profile':profile},'scenes':{}}
                for scene in scenes:
                    budget()
                    sink=((out/f'{scene}.jsonl').open('w',encoding='utf-8') if seed==seeds[0] and scene in ('train_015','train_123') else DiscardTrace())
                    try:
                        with opener(scene) as source:summary['scenes'][scene]=replay_scene(source,sink,handler)
                    finally:
                        if hasattr(sink,'close'):sink.close()
                    save(out/'summary.json',summary);log(event='scene',seed=seed,scene=scene,completed=len(summary['scenes']),total=30)
                card=build_scorecard(summary,out.name,references[seed],'original8000');write_scorecard(out,card)
                build_scorecard(summary,out.name,baseline,'original-baseline')
                report['by_seed'][seed]={'candidate':card['overall']['values'],'reference':card['overall']['baseline']}
                save(a.out/'report.json',report);del model,handler,control,saved
        keys=tuple(report['baseline'])
        report['means']={v:{k:mean(rows[v][k] for rows in report['by_seed'].values()) for k in keys} for v in ('candidate','reference')}
        report['promotion_checks']={
            'each_seed_clutter_not_higher':all(r['candidate']['clutter_outputs']<=r['reference']['clutter_outputs'] for r in report['by_seed'].values()),
            'each_seed_recall_within_010pp':all(r['candidate']['point_recall']>=r['reference']['point_recall']-.10 for r in report['by_seed'].values()),
            'each_seed_continuity_higher':all(r['candidate']['continuous_score']>r['reference']['continuous_score'] for r in report['by_seed'].values()),
            'each_seed_total_higher':all(r['candidate']['weighted_total']>r['reference']['weighted_total'] for r in report['by_seed'].values())}
        report['recommendation']='candidate' if all(report['promotion_checks'].values()) else 'reference'
        report['complete']=True
    except TimeoutError as error:report['budget_stop']=str(error)
    finally:report['elapsed_seconds']=monotonic()-started;save(a.out/'report.json',report)
    log(event='complete',complete=report['complete'],means=report.get('means'),recommendation=report.get('recommendation'))


if __name__=='__main__':main()
