"""Regression: a non-DDP batch requesting zero TF must never force queries."""
import argparse
from pathlib import Path
import sys
p=argparse.ArgumentParser()
p.add_argument('--project',type=Path,required=True)
p.add_argument('--upstream',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True)
a=p.parse_args()
sys.path[:0]=[str(a.upstream),str(a.project),str(a.project/'experiments/track_mt3_stage2')]
import torch
from adapter import make_model,make_window,IgnoreUnknownCriterion
torch.set_num_threads(4);torch.manual_seed(1)
model=make_model(a.upstream)
model.load_state_dict(torch.load(a.checkpoint,map_location='cpu')['model'])
model.train()
point=dict(Time=0,Azimuth=.2,Distance=.3,Elevation=.1,Amp=.5,SNR=.5,CircleID=0,PntID=0)
window=make_window([(0,[point],{0:1})],0,{0:1})
criterion=IgnoreUnknownCriterion(model.config.loss)
calls=[]
hook=model.qtm.register_forward_pre_hook(lambda module,args,kwargs:calls.append(kwargs['teacher_force']),with_kwargs=True)
with torch.no_grad():
    model([[window]*4],criterion=criterion,teacher_forcing_probability=0.)
hook.remove()
print({'requested_probability':0.,'actual_calls':calls},flush=True)
assert calls == [False]*4, 'Single-GPU training silently overrides requested zero teacher forcing'
