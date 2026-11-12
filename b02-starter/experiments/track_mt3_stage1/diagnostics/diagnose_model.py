"""Fixed-seed diagnostics; threshold sweeps are not selected competition settings."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from track_mt3.config import ExperimentConfig, _construct
from track_mt3.config_merge import load_merged_config
from track_mt3.data import MultiTargetSimulator, build_sliding_windows
from track_mt3.models import TrackMT3
from track_mt3.tracking import OnlineTracker
from track_mt3.metrics import gospa


@torch.no_grad()
def evaluate(model, config, seeds=(20261111,20261112,20261113), frames_count=100):
    rows=[]
    for seed in seeds:
        frames=MultiTargetSimulator(config.simulation,seed).simulate(frames_count)
        windows=build_sliding_windows(frames,config.model.window_size,include_partial=True)
        tracker=OnlineTracker(model)
        for frame,window in zip(frames,windows,strict=True):
            output=tracker.step(window.to('cuda'))
            pred=output.prediction
            raw=pred.positions.cpu().numpy()
            probs=pred.probabilities.flatten().cpu().numpy()
            truth=frame.states[:,:2]
            distances=np.linalg.norm(raw[:,None]-truth[None,:],axis=-1)
            rr,cc=linear_sum_assignment(distances)
            match_dist=distances[rr,cc]
            g=gospa(output.positions.numpy(),truth)
            rows.append({'seed':seed,'frame':frame.index,'truth':len(truth),'outputs':len(output.track_ids),
                         'query_count':len(probs),'max_probability':float(probs.max()),
                         'mean_probability':float(probs.mean()),'matched_query_probabilities':probs[rr].tolist(),
                         'raw_assignment_distances':match_dist.tolist(),'gospa':g.value,
                         'matched':len(g.matched_pairs),'missed':g.missed,'false':g.false,
                         'query_position_std':raw.std(axis=0).tolist()})
    scored=[r for r in rows if r['frame']>=20]
    def avg(k): return float(np.mean([r[k] for r in scored]))
    dist=[v for r in scored for v in r['raw_assignment_distances']]
    prob=[v for r in scored for v in r['matched_query_probabilities']]
    return {'scored_frames':len(scored),'total_outputs_all_frames':sum(r['outputs'] for r in rows),
            'mean_targets':avg('truth'),'mean_outputs':avg('outputs'),'mean_gospa':avg('gospa'),
            'empty_predictor_gospa':avg('truth'),'mean_matched_within_cutoff2':avg('matched'),
            'mean_false_component':avg('false'),'max_probability_all_frames':max(r['max_probability'] for r in rows),
            'mean_frame_max_probability':avg('max_probability'),
            'median_raw_assignment_distance':float(np.median(dist)),
            'matched_query_probability_quantiles':np.quantile(prob,[0,.5,.9,1]).tolist(),
            'mean_query_position_std':np.mean([r['query_position_std'] for r in scored],axis=0).tolist()}, rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sweep',action='store_true')
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    torch.set_num_threads(2)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    config=_construct(ExperimentConfig,saved['config'])
    upstream=load_merged_config('configs/paper.yaml','configs/training/track_mt3.yaml')
    config_diff={k:{'checkpoint':config.to_dict()[k],'upstream':upstream.to_dict()[k]} for k in ('model','simulation','loss') if config.to_dict()[k]!=upstream.to_dict()[k]}
    model=TrackMT3(config)
    model.load_state_dict(saved['model'])
    model.to('cuda').eval()
    choices=[(config.model.detection_threshold,config.model.tracking_threshold)]
    if a.sweep: choices += [(0.5,0.9),(0.25,0.9),(0.25,0.5)]
    report={'checkpoint':a.checkpoint,'step':saved['training_state']['step'],
            'model_simulation_loss_config_differences':config_diff,'variants':[]}
    for detection,tracking in choices:
        config.model.detection_threshold=detection
        config.model.tracking_threshold=tracking
        summary,rows=evaluate(model,config)
        record={'detection':detection,'tracking':tracking,**summary}
        report['variants'].append(record)
        (a.out/f'frames-d{detection}-t{tracking}.json').write_text(json.dumps(rows),encoding='utf-8')
        print(json.dumps(record),flush=True)
    (a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__': main()
