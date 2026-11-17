"""Re-audit the current frozen candidate on old calibration only."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from b02.algorithm import POSITION
from b02.replay import _scans,open_scenes,replay_frames
from experiments.joint_matching.diagnose import FragmentationAudit
from experiments.robust_validation.frozen import Components
from experiments.robust_validation.oracles import clean_clutter,graph_repair,perfect_identity,shadow_identity
from experiments.v2_association.run_stage1 import Diagnostic,Discard,save,sha
from experiments.v3_tracking.diagnose import reference_equal


def scores_of(outputs,frames):
    stream=iter(outputs)
    return replay_frames(frames,Discard(),lambda _p,_s:next(stream))


def old_geometry(tracker,published):
    scan=tracker.state.get('scan',0)+1;result=[];used=set()
    for batch,track in tracker.state.get('tracks',{}).items():
        if batch not in published:continue
        age=scan-track.last_scan
        result.append((batch,np.asarray(track.position)+np.asarray(track.velocity)*age,1.75*min(1+.2*(age-1),1.6)))
        used.add(batch)
    for bank in (tracker.retired,tracker.short_retired):
        for batch,saved in bank.items():
            age=scan-saved.track.last_scan
            if batch in used or age>14:continue
            result.append((batch,np.asarray(saved.track.position)+np.asarray(saved.track.velocity)*age,1.75*min(1+.1*(age-1),1.8)))
    return result


def graphs_for(tracker,points,outputs,geometry):
    indexed=defaultdict(set)
    for event in tracker.events:
        for index in event['indices']:indexed[index].add(event['batch'])
    for batch,index in tracker.retired_edges:indexed[index].add(batch)
    for row in tracker.proposals:indexed[row['index']].add(row['old'])
    lookup={point['PntID']:i for i,point in enumerate(points)}
    actual=[indexed[lookup[p['PntID']]]|{p['BatchID']} for p in outputs]
    relaxed=[set(edges) for edges in actual]
    if outputs and geometry:
        xyz=np.asarray([[p[k]/scale for k,scale in zip(POSITION,tracker.config.gate)] for p in outputs])
        for batch,predicted,radius in geometry:
            for row in np.where(np.square(xyz-predicted).sum(1)<=radius**2)[0]:relaxed[int(row)].add(batch)
    return actual,relaxed,indexed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','cache','prior','affinity','head','fusion','freeze','reference','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--index',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args();torch.set_num_threads(1)
    plan=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    cache=json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    assert cache['data_sha256']==sha(args.data) and cache['calibration']==plan['calibration']
    components=Components(args.prior,args.affinity,args.head,args.fusion,args.freeze)
    assert cache['point_weights_sha256']==components.freeze['artifacts']['point']
    reference=json.loads(args.reference.read_text(encoding='utf-8'))['scenes']
    assert 0<=args.index<args.shards
    args.out.mkdir(parents=True,exist_ok=False);start=monotonic()
    with open_scenes(args.data) as (_,opener):
        for scene in plan['calibration'][args.index::args.shards]:
            with opener(scene) as source:frames=list(_scans(source))
            with np.load(args.cache/'calibration'/f'{scene}.npz') as payload:
                scores,offsets=payload['point_scores'],payload['point_offsets']
            tracker=components.build('candidate',probe=True)
            fragments,diagnostic=FragmentationAudit(),Diagnostic()
            published=set();seen=Counter();misses=Counter();outputs=[];actual=[];relaxed=[]
            for frame,(circle,points,labels) in enumerate(frames):
                before=fragments.snapshot(tracker);keys=set(fragments.first)
                active=set(tracker.state.get('tracks',{}));retired=set(tracker.retired)|set(tracker.short_retired)
                geometry=old_geometry(tracker,published)
                values=scores[offsets[frame]:offsets[frame+1]].tolist()
                prediction=tracker(points,values)
                graph,wide,indexed=graphs_for(tracker,points,prediction,geometry)
                actual.append(graph);relaxed.append(wide);outputs.append(prediction)
                fragments.observe(circle,points,labels,prediction,tracker,before)
                for key in set(fragments.first)-keys:
                    row=fragments.first[key];row['first_published_id']=key[1] not in published;row['old_retired']=row['old'] in retired
                matched={p['PntID'] for p in prediction};raw={p['PntID'] for p in tracker.last_raw}
                internal=set(tracker.last_matches.values());born=set(tracker.last_births.values())
                for index,point in enumerate(points):
                    truth=labels[point['PntID']]
                    if truth<=0:continue
                    seen[truth]+=1
                    if point['PntID'] in matched:continue
                    misses['all']+=1;misses['within_first_two_truth_observations']+=seen[truth]<=2
                    if point['PntID'] in raw:reason='output_gate'
                    elif index in internal:reason='matched_unconfirmed'
                    elif index in born:reason='new_internal_birth'
                    elif indexed[index]:reason='candidate_not_selected'
                    else:reason='no_recorded_candidate'
                    misses[reason]+=1
                    misses['point_score_below_maintain']+=values[index]<.05
                diagnostic.observe(points,labels,prediction,tracker,active,circle)
                published.update(p['BatchID'] for p in prediction)
            baseline=scores_of(outputs,frames)
            assert reference_equal(baseline,reference[scene]),scene
            variants={'frozen_candidate':baseline};operations={}
            perfect,operations['perfect_identity']=perfect_identity(outputs,frames)
            repaired,operations['actual_graph']=graph_repair(outputs,frames,actual)
            wider,operations['relaxed_geometry']=graph_repair(outputs,frames,relaxed)
            clean=clean_clutter(outputs,frames)
            perfect_clean,operations['perfect_identity_clean']=perfect_identity(outputs,frames,clean=True)
            for name,predictions in (('perfect_identity',perfect),('actual_graph',repaired),('relaxed_geometry',wider),('clean_known_clutter',clean),('perfect_identity_clean',perfect_clean)):
                variants[name]=scores_of(predictions,frames)
                assert variants[name]['matched_target_points']==baseline['matched_target_points']
                if name not in ('clean_known_clutter','perfect_identity_clean'):
                    for key in ('outputs','clutter_outputs','unknown_outputs'):assert variants[name][key]==baseline[key]
            breakdown=fragments.result();short=[row for row in breakdown['fragments'] if row['points']<=2]
            counts={'total_extra_ids':len(breakdown['fragments']),'short2_total':len(short),
                    'short2_first_published':sum(row['first_published_id'] for row in short),
                    'short2_already_published':sum(not row['first_published_id'] for row in short),
                    'short2_prior_other_truth':sum(row['new_id_prior_other_truth'] for row in short)}
            reasons=Counter(('fresh_' if row['first_published_id'] else 'published_')+('retired' if row['old_retired'] else row['cause'])+'_'+str(row['old_match']) for row in short)
            save(args.out/f'{scene}.json',{'scene':scene,'variants':variants,'operations':operations,'identity':diagnostic.result(),
                'misses':dict(misses),'fragment_counts':counts,'short_first_causes':dict(reasons),
                'all_first_causes':breakdown['first_causes'],'shadow':{name:shadow_identity(value) for name,value in variants.items()},
                'read_only_shadow_exact':True,'deployable_variants':['frozen_candidate']})
            print(scene,counts,dict(misses),round(monotonic()-start,1),flush=True)


if __name__=='__main__':main()
