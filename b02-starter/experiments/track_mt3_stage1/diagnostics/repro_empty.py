"""Deterministic diagnostic: fails if the saved model emits no tracks."""
import argparse
import json
import torch
from track_mt3.config_merge import load_merged_config
from track_mt3.data import MultiTargetSimulator, build_sliding_windows
from track_mt3.models import TrackMT3
from track_mt3.tracking import OnlineTracker

p=argparse.ArgumentParser()
p.add_argument('--frames',type=int,default=100)
p.add_argument('--start',type=int,default=0)
p.add_argument('--checkpoint',default='/root/shared-nvme/b02/runs/stage1-20261111-02/restore/checkpoints/resumed.pt')
a=p.parse_args()
torch.set_num_threads(2)
torch.manual_seed(0)
c=load_merged_config('configs/paper.yaml','configs/training/track_mt3.yaml')
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
model=TrackMT3(c)
model.load_state_dict(saved['model'])
model.to('cuda').eval()
tracker=OnlineTracker(model)
frames=MultiTargetSimulator(c.simulation,20261111).simulate(a.frames)
windows=build_sliding_windows(frames,c.model.window_size,include_partial=True)
count=0
for window in windows[a.start:]:
    count+=len(tracker.step(window.to('cuda')).track_ids)
print(json.dumps({'frames_evaluated':a.frames-a.start,'start':a.start,'outputs':count,'checkpoint_step':saved['training_state']['step']}),flush=True)
assert count > 0, 'REPRODUCED: zero tracks emitted by saved model'
