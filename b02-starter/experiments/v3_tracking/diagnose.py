"""Attribute frozen v2 identity failures on calibration only, outside inference."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
from time import monotonic

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.v2_association.run_stage1 import Diagnostic,Discard,save,sha


def reference_equal(stats,reference):
    # replay_frames uses integer BatchID dictionary keys; persisted JSON uses strings.
    canonical=json.loads(json.dumps(stats))
    return all(canonical[k]==reference[k] for k in stats if not k.endswith('handler_ms'))


class Attribution:
    def __init__(self):
        self.last={};self.truth_last={};self.counts=Counter();self.examples=[]
        self.identity=defaultdict(Counter);self.first_cause={};self.id_scans=defaultdict(list)

    def observe(self,circle,points,labels,outputs,tracker,before):
        present=defaultdict(list)
        for i,p in enumerate(points):
            if labels[p['PntID']]>0:present[labels[p['PntID']]].append(i)
        events={e['batch']:e for e in tracker.events}
        for output in outputs:
            gt=labels[output['PntID']];batch=output['BatchID']
            if gt<=0:continue
            previous=self.last.get(gt);cause='first_output'
            if previous and previous[0]!=batch:
                old,old_circle=previous;self.counts['switches']+=1
                self.counts['switch_after_one_circle']+=circle-old_circle==1
                self.counts['switch_after_unemitted_truth']+=self.truth_last.get(gt,old_circle)>old_circle
                self.counts['new_id_already_born_before_previous_output']+=tracker.birth_scan.get(batch,10**12)<=old_circle+1
                if old not in before:cause='old_id_absent'
                elif old in tracker.last_matches:
                    label=labels[points[tracker.last_matches[old]]['PntID']]
                    cause='old_matched_other_truth' if label>0 and label!=gt else 'old_matched_clutter' if label==0 else 'old_matched_unknown' if label<0 else 'old_matched_same_truth'
                else:
                    position,velocity,last_scan=before[old];elapsed=tracker.state['scan']-last_scan
                    predicted=[x+v*elapsed for x,v in zip(position,velocity)]
                    radius=min(1+.2*(elapsed-1),1.6)
                    inside=any(sum((points[i][k]/g-x)**2 for k,g,x in zip(('Azimuth','Elevation','Distance'),tracker.config.gate,predicted))<=radius**2 for i in present[gt])
                    event=events.get(old,{});indices=event.get('indices',[])
                    candidate=any(labels[points[i]['PntID']]==gt for i in indices)
                    if not inside:cause='old_unmatched_outside_geometry'
                    elif candidate:
                        real=event.get('effective_costs',event['base_costs'])
                        best=min(cost for cost,i in zip(real,indices) if labels[points[i]['PntID']]==gt)
                        cause='old_unmatched_dummy_preferred' if best>=event.get('effective_dummy',event['dummy_cost']) else 'old_unmatched_competition'
                    elif any(labels[points[i]['PntID']]==gt for i in event.get('eligible_indices',[])):cause='old_unmatched_topk_excluded'
                    elif any(labels[points[i]['PntID']]==gt for i in event.get('gated_indices',[])):cause='old_unmatched_cost_gate_excluded'
                    else:cause='old_unmatched_occupied_in_earlier_stage'
                self.counts[cause]+=1
                if len(self.examples)<32:self.examples.append({'circle':circle,'truth':gt,'from':old,'to':batch,'cause':cause,'output_gap':circle-old_circle})
            self.first_cause.setdefault((gt,batch),cause)
            self.last[gt]=(batch,circle);self.identity[gt][batch]+=1;self.id_scans[(gt,batch)].append(circle)
        for gt in present:self.truth_last[gt]=circle

    def result(self):
        extra_causes=Counter();hist=Counter();short_extra=0;extras=[]
        for gt,ids in self.identity.items():
            main=max(ids,key=ids.get);n=len(ids);hist[str(n)]+=1
            if n>1 and ids[main]>=.9*sum(ids.values()):short_extra+=1
            for batch,count in ids.items():
                if batch==main:continue
                cause=self.first_cause[(gt,batch)];extra_causes[cause]+=1
                scans=self.id_scans[(gt,batch)];extras.append({'truth':gt,'id':batch,'points':count,'span':max(scans)-min(scans)+1,'first_cause':cause})
        return {'counts':dict(self.counts),'output_id_count_histogram':dict(hist),
                'multiple_ids_but_90pct_hits_in_one':short_extra,'extra_id_first_cause':dict(extra_causes),
                'extra_ids':extras,'examples':self.examples}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=3500);a=p.parse_args()
    import numpy as np
    import torch
    from b02.replay import open_scenes,_scans,replay_frames
    from experiments.v2_association.candidate import load_candidate
    from experiments.v2_association.tracker import Tracker
    plan=json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    reference=json.loads(a.reference.read_text(encoding='utf-8'))
    handler,spec=load_candidate(ROOT/'profiles/v2-association-candidate.json','cpu');torch.set_num_threads(2)
    a.out.mkdir(parents=True,exist_ok=False);start=monotonic();results={};total=Counter();exact={}
    save(a.out/'manifest.json',{'data_sha256':sha(a.data),'scenes':plan['calibration'],'split_sha256':sha(ROOT/'splits/v3_tracking.json'),
         'candidate_sha256':spec['checkpoint_sha256'],'diagnostic_source_sha256':sha(__file__),'truth_in_inference':False})
    with open_scenes(a.data) as (_,opener):
        for scene in plan['calibration']:
            with opener(scene) as file:frames=list(_scans(file))
            with np.load(a.cache/'calibration'/f'{scene}.npz') as cache:scores=cache['point_scores'];offsets=cache['point_offsets']
            tracker=Tracker('v1',contextual=True,scorer=handler.scorer,capture=True);attr=Attribution();diag=Diagnostic();predictions=[]
            for i,(circle,points,labels) in enumerate(frames):
                if monotonic()-start>a.seconds:raise TimeoutError('Diagnostic time limit')
                needed={x[0] for x in attr.last.values()}
                before={b:(t.position,t.velocity,t.last_scan) for b,t in tracker.state.get('tracks',{}).items() if b in needed}
                all_before=set(tracker.state.get('tracks',{}))
                output=tracker(points,scores[offsets[i]:offsets[i+1]].tolist());predictions.append(output)
                attr.observe(circle,points,labels,output,tracker,before);diag.observe(points,labels,output,tracker,all_before,circle)
            iterator=iter(predictions);stats=replay_frames(frames,Discard(),lambda p,s:next(iterator))
            old=reference['scenes'][scene]
            assert reference_equal(stats,old),scene
            exact[scene]=True;results[scene]={'attribution':attr.result(),'diagnostics':diag.result()};total.update(attr.counts)
            save(a.out/'per_scene.json',results)
            print(json.dumps({'scene':scene,'seconds':round(monotonic()-start,2),'counts':dict(attr.counts)}),flush=True)
    save(a.out/'report.json',{'complete':True,'v2_reference_exact':exact,'counts':dict(total),'wall_seconds':monotonic()-start})


if __name__=='__main__':main()
