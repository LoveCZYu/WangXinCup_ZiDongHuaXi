"""Frozen-weight probes for confidence, localization, teacher forcing and path parity."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--project',type=Path,required=True)
p.add_argument('--upstream',type=Path,required=True)
p.add_argument('--data',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
sys.path[:0]=[str(a.upstream),str(a.project),str(a.project/'experiments/track_mt3_stage2')]
import torch
from adapter import make_model,make_window,associate,coordinates,IgnoreUnknownCriterion,HISTORY,GATE
from b02.replay import open_scenes,_scans
from track_mt3.tracking import OnlineTracker
from track_mt3.matching.cta import CrossFrameTargetAlignment
from scipy.optimize import linear_sum_assignment

torch.set_num_threads(4);torch.manual_seed(20261111)
model=make_model(a.upstream)
model.load_state_dict(torch.load(a.checkpoint,map_location='cpu')['model'])
model.eval()
scenes={}
with open_scenes(a.data) as (_,opener):
    for name in ('train_001','train_002','train_015','train_016'):
        with opener(name) as source:
            scenes[name]=[]
            for row in _scans(source):
                scenes[name].append(row)
                if row[0]>=139:break


def stats(values):
    return {'count':len(values),'median':float(np.median(values)) if values else None,
            'p90':float(np.quantile(values,.9)) if values else None,
            'max':float(np.max(values)) if values else None}


@torch.no_grad()
def evaluate(detection,tracking,teacher=False):
    model.config.model.detection_threshold=detection
    model.config.model.tracking_threshold=tracking
    results={}
    for name,frames in scenes.items():
        online=OnlineTracker(model);tracks=model.empty_tracks();history=deque(maxlen=HISTORY)
        alignment=CrossFrameTargetAlignment(model.config.loss)
        truth=raw=outputs=hits=clutter=unknown=all_hits=all_outputs=oracle_covered=0
        near_probs=[];far_probs=[];new_probs=[];old_probs=[];errors=[]
        for circle,points,labels in frames:
            history.append((circle,points,labels if teacher else None))
            window=make_window(history,circle,labels if teacher else None)
            if teacher:
                prediction=model.forward_window(window,tracks)
                assignment=alignment(prediction,tracks,window.target_positions,window.target_ids)
                tracks,selected=model.qtm(prediction,tracks,query_target_ids=assignment.query_target_ids,
                                          query_track_ids=assignment.query_target_ids,teacher_force=True)
                positions=prediction.positions[selected].cpu().numpy()
            else:
                result=online.step(window);prediction=result.prediction
                positions=result.positions.numpy()
            if circle<100:continue
            result=associate(positions,np.arange(len(positions)),points)
            all_result=associate(prediction.positions.cpu().numpy(),np.arange(len(prediction.positions)),points)
            tp=[point for point in points if labels[point['PntID']]>0]
            truth+=len(tp);raw+=len(positions);outputs+=len(result)
            hits+=sum(labels[r['PntID']]>0 for r in result)
            clutter+=sum(labels[r['PntID']]==0 for r in result)
            unknown+=sum(labels[r['PntID']]==-1 for r in result)
            all_hits+=sum(labels[r['PntID']]>0 for r in all_result);all_outputs+=len(all_result)
            predicted=prediction.positions.cpu().numpy()
            probs=prediction.probabilities.cpu().numpy().ravel()
            new_probs.extend(probs[prediction.num_track_queries:].tolist())
            old_probs.extend(probs[:prediction.num_track_queries].tolist())
            if tp:
                residual=np.abs(predicted[:,None,:]-coordinates(tp)[None,:,:])
                d=(residual/np.asarray(GATE)).max(axis=2)
                near=(d<=1).any(axis=1)
                near_probs.extend(probs[near].tolist());far_probs.extend(probs[~near].tolist())
                rows,cols=linear_sum_assignment(np.concatenate([np.where(d<=1,d,1e6),np.full((len(d),len(d)),10)],axis=1))
                oracle_covered+=sum(j<len(tp) and d[i,j]<=1 for i,j in zip(rows,cols))
                ri,ci=linear_sum_assignment(np.linalg.norm(residual,axis=2))
                errors.extend(residual[ri,ci].tolist())
        results[name]={'truth':truth,'raw_tracks':raw,'outputs':outputs,'hits':hits,'clutter':clutter,'unknown':unknown,
                       'recall':hits/truth,'precision':hits/(hits+clutter) if hits+clutter else 0,
                       'all_query_original_point_recall':all_hits/truth,'all_query_outputs':all_outputs,
                       'oracle_geometry_recall':oracle_covered/truth,
                       'near_truth_probabilities':stats(near_probs),'away_from_truth_probabilities':stats(far_probs),
                       'new_query_probabilities':stats(new_probs),'old_query_probabilities':stats(old_probs),
                       'median_assignment_abs_error_azi_dist_el':np.median(errors,axis=0).tolist()}
    return results


report={}
for name,detection,tracking,teacher in [('original',.75,.9,False),('detect_05',.5,.9,False),
    ('detect_025',.25,.9,False),('detect_01',.1,.9,False),('detect025_track05',.25,.5,False),
    ('oracle_teacher_forced',.75,.9,True)]:
    report[name]=evaluate(detection,tracking,teacher)
    print(json.dumps({'probe':name,'summary':{s:{k:v[k] for k in ('recall','precision','oracle_geometry_recall')} for s,v in report[name].items()}}),flush=True)

# Single and batched forward must agree in eval mode on identical state/input.
frames=scenes['train_015'];history=deque(frames[95:101],maxlen=HISTORY)
window=make_window(history,100,frames[100][2])
with torch.no_grad():
    single=model.forward_window(window,model.empty_tracks())
    batch=model.forward_windows([window],[model.empty_tracks()])[0]
    report['path_parity']={'position_max_abs':float((single.positions-batch.positions).abs().max()),
                           'logit_max_abs':float((single.logits-batch.logits).abs().max())}
    clean=make_window(deque([(i,pts,None) for i,pts,_ in history],maxlen=HISTORY),100)
    prediction=model.forward_window(clean,model.empty_tracks())
    report['label_parity']={'position_max_abs':float((single.positions-prediction.positions).abs().max()),
                           'logit_max_abs':float((single.logits-prediction.logits).abs().max())}
    changed=make_window(deque([(i,[{**pt,'Elevation':.9} for pt in pts],None) for i,pts,_ in history],maxlen=HISTORY),100)
    perturbed=model.forward_window(changed)
    report['elevation_perturbation']={'mean_position_abs_change':float((prediction.positions-perturbed.positions).abs().mean()),
                                     'mean_probability_abs_change':float((prediction.probabilities-perturbed.probabilities).abs().mean())}

calls=[]
hook=model.qtm.register_forward_pre_hook(lambda module,args,kwargs:calls.append(kwargs.get('teacher_force')),with_kwargs=True)
criterion=IgnoreUnknownCriterion(model.config.loss)
with torch.no_grad():
    model([[window]*4],criterion=criterion,teacher_forcing_probability=0.)
hook.remove()
report['requested_zero_teacher_forcing_actual_calls']=calls
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['original','detect_05','detect_025','detect_01','detect025_track05','oracle_teacher_forced']}),flush=True)
