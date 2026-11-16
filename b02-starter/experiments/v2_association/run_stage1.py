"""Four fixed state-maintenance ablations on training-only calibration scenes."""
import argparse
from collections import Counter, defaultdict, deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic, perf_counter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def historical_identity(history):
    positive={x for x in history if x>0}
    if len(positive)>1:return None
    if positive:return next(iter(positive))
    return None if any(x<0 for x in history) else 0


class Diagnostic:
    """Truth exists only here, outside the tracker and its learned input."""
    def __init__(self):
        self.histories={};self.last={};self.owners=defaultdict(Counter);self.counts=Counter()
        self.cases=[]

    def observe(self,points,labels,outputs,tracker,before_ids,scan):
        present={v for v in labels.values() if v>0}
        for event in tracker.events:
            anchor=historical_identity(self.histories.get(event['batch'],()))
            if anchor is None:continue
            if anchor>0 and anchor in present:
                self.counts['anchor_present_rows']+=1
                for key,indices in (('inside_geometry',event['gated_indices']),
                                    ('inside_cost_gate',event['eligible_indices']),
                                    ('inside_topk',event['indices'])):
                    self.counts[key]+=any(labels[points[i]['PntID']]==anchor for i in indices)
                index=tracker.last_matches.get(event['batch'])
                self.counts['assigned_same_identity']+=index is not None and labels[points[index]['PntID']]==anchor
        output_points={p['PntID'] for p in outputs}
        for batch,index in tracker.last_matches.items():
            label=labels[points[index]['PntID']]
            self.counts['internal_matches']+=1
            if label==0:
                self.counts['clutter_internal_matches']+=1
                self.counts['clutter_filtered_after_update']+=points[index]['PntID'] not in output_points
            self.histories.setdefault(batch,deque(maxlen=8)).append(label)
        for batch,index in tracker.last_births.items():
            self.histories[batch]=deque([labels[points[index]['PntID']]],maxlen=8)
        for batch in list(self.histories):
            if batch not in tracker.state['tracks']:del self.histories[batch]
        for point in outputs:
            label=labels[point['PntID']];batch=point['BatchID']
            if label<=0:continue
            previous=self.last.get(label)
            if previous is not None and previous!=batch:
                self.counts['switches']+=1
                self.counts['switch_old_id_alive_before']+=previous in before_ids
                if len(self.cases)<16:self.cases.append({'scan':scan,'truth_id':label,'from':previous,'to':batch})
            self.last[label]=batch;self.owners[batch][label]+=1

    def result(self):
        c=dict(self.counts)
        c.update(truth_carrying_ids=len(self.owners),mixed_ids=sum(len(v)>1 for v in self.owners.values()),
                 mixed_minority_truth_points=sum(sum(v.values())-max(v.values()) for v in self.owners.values()))
        n=c.get('anchor_present_rows',0)
        return {'counts':c,'candidate_recall':c.get('inside_topk',0)/n if n else None,'switch_examples':self.cases}


class Discard:
    def write(self,value):pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--seconds',type=int,default=1700)
    p.add_argument('--reference',type=Path)
    a=p.parse_args()
    import torch
    from b02.point_tracker import load_tracker
    from b02.replay import open_scenes,_scans,replay_frames
    from b02.scorecard import build_scorecard,write_scorecard
    from experiments.association.diagnose import oracle
    from experiments.v2_association.tracker import Tracker,VARIANTS
    started=monotonic();a.out.mkdir(parents=True,exist_ok=False)
    handler,profile=load_tracker(device=a.device)
    plan=json.loads((ROOT/'experiments/point_scorer_expanded/plan.json').read_text(encoding='utf-8'))
    split=json.loads((ROOT/'splits/v1.json').read_text(encoding='utf-8'))
    public=json.loads((ROOT/'splits/a_board.json').read_text(encoding='utf-8'))
    scenes=plan['calibration_scenes']
    assert len(scenes)==8 and set(scenes)<=set(split['train'])
    assert not set(scenes)&set(split['dev']+split['holdout']+public['scenes'])
    sources=[*Path(__file__).parent.glob('*.py'),ROOT/'b02/algorithm.py',ROOT/'b02/point_tracker.py',
             ROOT/'experiments/association/interventions.py',ROOT/'profiles/current.json',ROOT/'assets/point-scorer-8000.pt']
    manifest={'format':'b02-v2-stage1-v1','scenes':scenes,'variants':list(VARIANTS),'device':a.device,
              'point_profile':profile,'seconds_limit':a.seconds,'data_sha256':sha(a.data),
              'source_sha256':{str(x.relative_to(ROOT)):sha(x) for x in sources},
              'truth_usage':'diagnostics and local metrics only; not tracker inputs',
              'changes':{'position_trust':'position=prediction+min(1,score/.50)*innovation; existing velocity trust unchanged',
                         'trusted_confirmation':'require lifetime hits>=3 and cumulative min(1,score/.50)>=3; motion gain still uses lifetime hits',
                         'conservative_dummy':'dummy=1.15-.50*(1-min(strong_hits,6)/6); geometry and real-edge costs unchanged'},
              'preview_scene':'train_032','torch':str(torch.__version__),
              'promotion':{'switch_ratio_max':.75,'continuous_min_gain':3.,'recall_max_drop_pp':.3,'clutter_ratio_max':1.05}}
    save(a.out/'manifest.json',manifest)
    summaries={v:{'kind':'local_diagnostics','config':{'v2_variant':v,'profile':profile},'scenes':{}} for v in VARIANTS}
    diagnostics={v:{} for v in VARIANTS}
    report={'complete':False,'baseline_shadow_exact':True,'frozen_v1_promoted':False}
    def budget():
        if monotonic()-started>a.seconds:raise TimeoutError('Stage1 time limit reached')
    try:
        with open_scenes(a.data) as (_,opener):
            for scene in scenes:
                budget()
                with opener(scene) as file:frames=list(_scans(file))
                trackers={v:Tracker(v,capture=True) for v in VARIANTS}
                diags={v:Diagnostic() for v in VARIANTS}
                cache={v:[] for v in VARIANTS};times={v:[] for v in VARIANTS}
                history=deque(maxlen=5);shadow={}
                for circle,points,labels in frames:
                    budget();start=perf_counter()
                    with torch.no_grad():
                        scores=handler.model(torch.as_tensor(handler.features(points,list(history)),device=handler.device)).sigmoid().cpu().tolist()
                    history.append(points);feature_ms=(perf_counter()-start)*1000
                    frozen=handler(points,shadow)
                    for variant,tracker in trackers.items():
                        before_ids=set(tracker.state.get('tracks',{}));start=perf_counter()
                        outputs=tracker(points,scores)
                        times[variant].append(feature_ms+(perf_counter()-start)*1000)
                        if variant=='v1' and outputs!=frozen:
                            raise AssertionError(f'v1 replay mismatch {scene}/{circle}')
                        cache[variant].append(outputs)
                        diags[variant].observe(points,labels,outputs,tracker,before_ids,circle)
                for variant in VARIANTS:
                    folder=a.out/variant;folder.mkdir(exist_ok=True)
                    stream=(folder/f'{scene}.jsonl').open('w',encoding='utf-8') if scene=='train_032' else Discard()
                    try:
                        outputs=iter(cache[variant])
                        summary=replay_frames(frames,stream,lambda points,state:next(outputs))
                    finally:
                        if hasattr(stream,'close'):stream.close()
                    measured=sorted(times[variant])
                    summary.update(p95_handler_ms=round(measured[(95*len(measured)-1)//100],4),
                                   max_handler_ms=round(measured[-1],4))
                    summaries[variant]['scenes'][scene]=summary
                    diagnostics[variant][scene]=diags[variant].result()
                    save(folder/'summary.json',summaries[variant])
                print(json.dumps({'scene':scene,'seconds':round(monotonic()-started,2),
                                  'switches':{v:diagnostics[v][scene]['counts'].get('switches',0) for v in VARIANTS}}),flush=True)
        results={}
        for variant in VARIANTS:
            score=build_scorecard(summaries[variant],variant);write_scorecard(a.out/variant,score)
            total=Counter()
            for value in diagnostics[variant].values():total.update(value['counts'])
            results[variant]={'metrics':score['overall']['values'],'identity':dict(total)}
            save(a.out/variant/'diagnostics.json',diagnostics[variant])
        reference=results['v1'];checks={}
        for variant in VARIANTS[1:]:
            m=results[variant]['metrics'];b=reference['metrics']
            c=results[variant]['identity'];d=reference['identity']
            checks[variant]={'switches':c.get('switches',0)<=.75*d.get('switches',0),
                             'continuous':m['continuous_score']>=b['continuous_score']+3,
                             'recall':m['point_recall']>=b['point_recall']-.3,
                             'clutter':m['clutter_outputs']<=1.05*b['clutter_outputs']}
        oracles={}
        for name,kwargs in (('clean_output',{'clean':True}),('identity_only',{'identity':True})):
            oracles[name]=build_scorecard(oracle(summaries['v1'],**kwargs),name)['overall']['values']
        report.update(complete=True,results=results,diagnostic_oracles=oracles,
                      milestone_checks=checks,milestone_passed={v:all(c.values()) for v,c in checks.items()},
                      best_exploratory=max(VARIANTS,key=lambda v:results[v]['metrics']['weighted_total']))
        if a.reference:
            previous=json.loads((a.reference/'summary.json').read_text(encoding='utf-8'))
            previous_score=build_scorecard(previous,'saved-v1')['overall']['values']
            report['saved_reference_differences']={k:[previous_score[k],reference['metrics'][k]] for k in previous_score
                                                  if previous_score[k]!=reference['metrics'][k]}
    finally:
        report['wall_seconds']=monotonic()-started
        save(a.out/'report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
