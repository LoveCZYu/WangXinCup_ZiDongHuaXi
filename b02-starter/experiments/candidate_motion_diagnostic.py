"""Past-only prediction with oracle historical identities, diagnostic only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from b02.algorithm import DEFAULT_CONFIG,Track
from b02.replay import _scans,open_scenes
from experiments.v2_association.features import normalized
from experiments.v2_association.run_stage1 import save,sha


def metrics(rows):
    if not rows:return {'count':0}
    r=np.asarray(rows,float)
    return dict(count=len(r),gate_coverage=float(r[:,0].mean()),base_cost_coverage=float(r[:,1].mean()),
        residual_norm_quantiles=np.quantile(np.sqrt(r[:,2]),[.5,.9,.95,.99]).tolist(),
        squared_error_axis_share=(r[:,3:6].sum(0)/r[:,2].sum()).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','cache','protocol','out'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.protocol.read_text(encoding='utf-8'))
    assert sha(Path(__file__))==plan['source_sha256'] and sha(args.data)==plan['data_sha256']
    names=plan['training4'];all_rows=defaultdict(list);by_scene={};second=[]
    with open_scenes(args.data) as (_,opener):
        for scene in names:
            with opener(scene) as source:frames=list(_scans(source))
            with np.load(args.cache/'fit'/f'{scene}.npz') as stored:scores,offsets=stored['point_scores'],stored['point_offsets']
            assert len(offsets)==len(frames)+1
            tracks={};history=defaultdict(list);rows=defaultdict(list)
            for frame,(_,points,labels) in enumerate(frames):
                scan=frame+1;by_truth=defaultdict(list)
                for i,pt in enumerate(points):
                    if labels[pt['PntID']]>0:by_truth[labels[pt['PntID']]].append(i)
                for truth,indices in by_truth.items():
                    if len(indices)!=1:continue
                    i=indices[0];pt=points[i];x=normalized(pt).astype(float);score=float(scores[offsets[frame]+i])
                    old=tracks.get(truth);past=history[truth]
                    if old is not None:
                        elapsed=scan-old.last_scan;radius=min(1+.2*(elapsed-1),1.6)
                        predictors={'cv035':np.asarray(old.position)+np.asarray(old.velocity)*elapsed}
                        if len(past)>=4:
                            selected=past[-16:];time=np.asarray([when-scan for when,_ in selected],float)
                            xyz=np.stack([z for _,z in selected]);dt=time-time.mean()
                            v=(dt[:,None]*(xyz-xyz.mean(0))).sum(0)/max(float(np.square(dt).sum()),1e-6)
                            predictors['ols16']=xyz.mean(0)-v*time.mean()
                        if old.hits>=4 and elapsed<=3:
                            for name,prediction in predictors.items():
                                error=np.square(x-prediction);norm=float(error.sum());ratio=norm/radius**2
                                record=[float(ratio<=1),float(ratio+.5*(1-score)<1.15),norm,*error]
                                rows[name+'_all'].append(record);all_rows[name+'_all'].append(record)
                                if score>=.525:
                                    rows[name+'_strong'].append(record);all_rows[name+'_strong'].append(record)
                        gain=1. if old.hits==1 else DEFAULT_CONFIG.velocity_gain
                        trust=min(1.,score/.5)
                        updated=(1-gain)*np.asarray(old.velocity)+gain*(x-np.asarray(old.position))/elapsed
                        old.velocity=tuple(np.asarray(old.velocity)+trust*(updated-np.asarray(old.velocity)))
                        old.position=tuple(x);old.last_scan=scan;old.hits+=1
                    else:tracks[truth]=Track(tuple(x),scan)
                    if len(past)>=2 and past[-1][0]==scan-1 and past[-2][0]==scan-2:
                        second.append(x-2*past[-1][1]+past[-2][1])
                    past.append((scan,x));history[truth]=past[-16:]
            by_scene[scene]={k:metrics(v) for k,v in rows.items()}
    delta=np.stack(second)
    report=dict(protocol_sha256=sha(args.protocol),prediction={k:metrics(v) for k,v in all_rows.items()},scenes=by_scene,
        contiguous_second_difference=dict(count=len(delta),median_abs=np.median(np.abs(delta),axis=0).tolist(),
            q95_abs=np.quantile(np.abs(delta),.95,axis=0).tolist()),
        scope='Uses true past identity association for diagnostics, never deployable tracking scores. Predictor inputs are past only; current label solely groups/evaluates. Oracle updates ignore actual gate rejection.',
        limitations='Second differences mix measurement variation and real acceleration; these data alone do not separate sensor noise from target manoeuvres.',
        new_training=False,independent_validation=False)
    save(args.out,report);print(json.dumps({k:v for k,v in report.items() if k!='scenes'},indent=2))


if __name__=='__main__':main()
