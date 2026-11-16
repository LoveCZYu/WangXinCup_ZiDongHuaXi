"""Replay frozen v1 on declared train/calibration scenes; labels never enter features."""
import argparse
from collections import Counter,deque
import json
from pathlib import Path
import sys
from time import monotonic

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=3500)
    p.add_argument('--label-mode',choices=('physical','continuity'),default='physical')
    p.add_argument('--score-cache',type=Path)
    p.add_argument('--context',action='store_true')
    a=p.parse_args()
    import numpy as np
    import torch
    from b02.replay import open_scenes,_scans
    from b02.point_tracker import load_tracker
    from experiments.v2_association.tracker import Tracker
    from experiments.v2_association.features import TRACK_DIM,EDGE_DIM,MAX_EDGES,FEATURE_VERSION
    if a.context:
        from experiments.v2_association.context import TRACK_DIM,EDGE_DIM,FEATURE_VERSION
    from experiments.v2_association.run_stage1 import save,sha,historical_identity
    from experiments.v2_association.labels import continuity_owners
    start=monotonic();a.out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((ROOT/'experiments/point_scorer_expanded/plan.json').read_text(encoding='utf-8'))
    split=json.loads((ROOT/'splits/v1.json').read_text(encoding='utf-8'))
    public=json.loads((ROOT/'splits/a_board.json').read_text(encoding='utf-8'))
    fit,cal=plan['fit_scenes'],plan['calibration_scenes']
    assert set(fit+cal)<=set(split['train']) and not set(fit)&set(cal)
    assert not set(fit+cal)&set(split['dev']+split['holdout'])
    assert not set(cal)&set(public['scenes'])
    handler,profile=load_tracker(device='cpu')
    if a.score_cache:
        prior=json.loads((a.score_cache/'manifest.json').read_text(encoding='utf-8'))
        assert prior['fit']==fit and prior['calibration']==cal and prior['data_sha256']==sha(a.data)
    manifest={'format':'b02-edge-dataset-v1','feature_version':FEATURE_VERSION,
              'fit':fit,'calibration':cal,'dev':[],'holdout':[],
              'selection':'same 8 evenly spaced 8-scan clips per scene; full causal prefix replay',
              'track_dim':TRACK_DIM,'edge_dim':EDGE_DIM,'max_edges':MAX_EDGES,
              'profile':profile,'data_sha256':sha(a.data),'sources':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
              'labels':'unique positive identity in prior matched observations; mixed/unknown prefixes ignored; known clutter prefix -> null',
              'future_or_numeric_ids_as_features':False,'seconds_limit':a.seconds,
              'label_mode':a.label_mode,'context':a.context,'score_cache_manifest_sha256':sha(a.score_cache/'manifest.json') if a.score_cache else None,
              'continuity_owner_rule':'among same-identity pure history tracks with current true candidate: prior emitted support, prior positive history count, age; other duplicates -> null' if a.label_mode=='continuity' else None}
    save(a.out/'manifest.json',manifest)
    report={'complete':False,'scenes':{}}
    def budget():
        if monotonic()-start>a.seconds:raise TimeoutError('Pair preparation time limit')
    try:
        with open_scenes(a.data) as (_,opener):
            for role,names in (('fit',fit),('calibration',cal)):
                folder=a.out/role;folder.mkdir()
                for name in names:
                    budget()
                    with opener(name) as file:frames=list(_scans(file))
                    begins=np.linspace(0,len(frames)-8,8,dtype=int)
                    selected={i for b in begins for i in range(int(b),int(b)+8)}
                    tracker=Tracker('v1',capture=True,contextual=a.context);history=deque(maxlen=5);truth_histories={};output_counts={}
                    cached=None;cached_offsets=None
                    if a.score_cache:
                        with np.load(a.score_cache/role/f'{name}.npz') as data:
                            cached=data['point_scores'];cached_offsets=data['point_offsets']
                        assert len(cached_offsets)==len(frames)+1
                    tracks_x=[];edges_x=[];base_x=[];valid_x=[];target_x=[];counts=Counter()
                    score_parts=[];offsets=[0]
                    for frame_index,(_,points,labels) in enumerate(frames):
                        budget()
                        if cached is None:
                            with torch.no_grad():
                                scores=handler.model(torch.as_tensor(handler.features(points,list(history)))).sigmoid().tolist()
                        else:
                            scores=cached[cached_offsets[frame_index]:cached_offsets[frame_index+1]].tolist()
                            assert len(scores)==len(points)
                        history.append(points);score_parts.extend(scores);offsets.append(len(score_parts))
                        tracker.feature_capture=frame_index in selected
                        outputs=tracker(points,scores)
                        if frame_index in selected:
                            present={v for v in labels.values() if v>0}
                            owners=continuity_owners(tracker.events,truth_histories,output_counts,points,labels) if a.label_mode=='continuity' else {}
                            for event in tracker.events:
                                counts['rows_seen']+=1
                                anchor=historical_identity(truth_histories.get(event['batch'],()))
                                if anchor is None:counts['prefix_ignored']+=1;continue
                                if not event['indices']:counts['empty_candidates']+=1;continue
                                tx=event['track_features'];ex=event['edge_features']
                                n=len(event['indices'])
                                valid=np.zeros(MAX_EDGES+1,dtype=bool);target=np.zeros(MAX_EDGES+1,dtype=bool)
                                valid[-1]=True
                                for j,index in enumerate(event['indices']):
                                    label=labels[points[index]['PntID']]
                                    valid[j]=label>=0
                                    target[j]=anchor>0 and label==anchor and (a.label_mode=='physical' or owners.get(anchor)==event['batch'])
                                if not valid[:-1].any():counts['unknown_candidates_only']+=1;continue
                                if not target.any():
                                    target[-1]=True;counts['null_rows']+=1
                                    if anchor>0 and anchor in present:
                                        if any(labels[points[i]['PntID']]==anchor for i in event['indices']):counts['duplicate_rows_to_null']+=1
                                        else:counts['true_point_outside_candidates']+=1
                                else:counts['positive_rows']+=1
                                counts['other_target_edges']+=sum(labels[points[i]['PntID']]>0 and labels[points[i]['PntID']]!=anchor for i in event['indices'])
                                padded=np.zeros((MAX_EDGES,EDGE_DIM),dtype=np.float32);padded[:n]=ex
                                base=np.zeros(MAX_EDGES+1,dtype=np.float32);base[:n]=event['base_costs'];base[-1]=event['dummy_cost']
                                tracks_x.append(tx);edges_x.append(padded);base_x.append(base);valid_x.append(valid);target_x.append(target)
                        for point in outputs:
                            label=labels[point['PntID']]
                            if label>0:output_counts.setdefault(point['BatchID'],Counter())[label]+=1
                        for batch,index in tracker.last_matches.items():
                            truth_histories.setdefault(batch,deque(maxlen=8)).append(labels[points[index]['PntID']])
                        for batch,index in tracker.last_births.items():
                            truth_histories[batch]=deque([labels[points[index]['PntID']]],maxlen=8)
                        for batch in list(truth_histories):
                            if batch not in tracker.state['tracks']:del truth_histories[batch]
                        for batch in list(output_counts):
                            if batch not in tracker.state['tracks']:del output_counts[batch]
                    if not tracks_x:raise ValueError(f'No supervised groups in {name}')
                    np.savez_compressed(folder/f'{name}.npz',track=np.stack(tracks_x),edges=np.stack(edges_x),
                        base=np.stack(base_x),valid=np.stack(valid_x),target=np.stack(target_x),
                        point_scores=np.asarray(score_parts,dtype=np.float32),point_offsets=np.asarray(offsets,dtype=np.int64))
                    report['scenes'][name]={'role':role,'groups':len(tracks_x),'counts':dict(counts),
                        'frames':len(frames),'supervised_frames':len(selected),'source_file':f'{role}/{name}.npz'}
                    save(a.out/'report.json',report)
                    print(json.dumps({'scene':name,'role':role,'groups':len(tracks_x),
                                      'positives':counts['positive_rows'],'seconds':round(monotonic()-start,2)}),flush=True)
        report['complete']=True
    finally:
        report['wall_seconds']=monotonic()-start;save(a.out/'report.json',report)


if __name__=='__main__':main()
