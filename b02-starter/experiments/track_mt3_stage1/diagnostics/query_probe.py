"""Inspect truth-assisted versus autonomous query behavior; never deployment logic."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import torch
import numpy as np
from track_mt3.config import ExperimentConfig, _construct
from track_mt3.data import MultiTargetSimulator, build_sliding_windows
from track_mt3.models import TrackMT3
from track_mt3.losses import CollectiveAverageCriterion

torch.set_num_threads(2)
checkpoint='/root/shared-nvme/b02/runs/stage1-20261111-02/restore/checkpoints/resumed.pt'
saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
c=_construct(ExperimentConfig,saved['config'])
m=TrackMT3(c)
m.load_state_dict(saved['model'])
m.to('cuda').eval()
windows=build_sliding_windows(MultiTargetSimulator(c.simulation,20261111).simulate(21),20)
report={}
with torch.no_grad():
    single=m.forward_window(windows[0])
    batched=m.forward_windows([windows[0]],[m.empty_tracks()])[0]
    report['single_vs_batch']={'positions_max_abs_diff':float((single.positions-batched.positions).abs().max()),
                              'logits_max_abs_diff':float((single.logits-batched.logits).abs().max())}
    no_truth=replace(windows[0],target_positions=torch.empty(0,2),target_ids=torch.empty(0,dtype=torch.long),
                     measurement_ids=torch.full_like(windows[0].measurement_ids,-1))
    blinded=m.forward_window(no_truth)
    report['truth_labels_removed']={'positions_max_abs_diff':float((single.positions-blinded.positions).abs().max()),
                                   'logits_max_abs_diff':float((single.logits-blinded.logits).abs().max())}
    no_signal=replace(windows[0],measurements=torch.zeros_like(windows[0].measurements))
    zeroed=m.forward_window(no_signal)
    report['all_measurements_zeroed']={'positions_mean_l2_change':float((single.positions-zeroed.positions).norm(dim=-1).mean()),
                                     'probabilities_max_change':float((single.probabilities-zeroed.probabilities).abs().max())}
    groups=[]
    for tf in (False,True):
        predictions,_,_,assignments=m.forward_clip(windows,criterion=CollectiveAverageCriterion(c.loss),teacher_forcing=tf)
        for k,(p,a) in enumerate(zip(predictions,assignments,strict=True)):
            probabilities=p.probabilities.flatten().cpu().numpy()
            matched=a.query_target_ids.cpu().numpy()>=0
            is_track=np.arange(len(probabilities))<p.num_track_queries
            r={'teacher_forcing':tf,'window':k,'track_queries':p.num_track_queries}
            for name,mask in [('matched_track',matched & is_track),('unmatched_track',~matched & is_track),
                              ('matched_detection',matched & ~is_track),('unmatched_detection',~matched & ~is_track)]:
                values=probabilities[mask]
                r[name]={'count':len(values),'mean_probability':float(values.mean()) if len(values) else None}
            groups.append(r)
    report['truth_assistance_diagnostic']=groups
print(json.dumps(report,indent=2))
Path('/root/shared-nvme/b02/runs/diagnosis-query401.json').write_text(json.dumps(report,indent=2))
