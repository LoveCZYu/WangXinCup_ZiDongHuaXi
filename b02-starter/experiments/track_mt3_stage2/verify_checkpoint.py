"""Check final checkpoint through the public replay entry and bounded dense input."""
import argparse
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--project',type=Path,required=True)
p.add_argument('--upstream',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
sys.path.insert(0,str(a.project.resolve()))
from b02.neural import load_tracker
from b02.replay import _checked_outputs,_identity
import torch
import numpy as np
import json

torch.set_num_threads(4)
handler,config=load_tracker(a.checkpoint,a.upstream,'cuda')
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
assert config['history_frames']==6
rng=np.random.default_rng(42)
state={}
torch.cuda.reset_peak_memory_stats()
started=time.perf_counter()
counts=[]
# Exercise a fully populated bounded history. Synthetic points have no labels.
for circle in range(6):
    xyz=rng.uniform(0,1,(8812,3))
    points=[dict(Time=circle*5000,Azimuth=float(v[0]),Distance=float(v[1]),Elevation=float(v[2]),
                 Amp=.5,SNR=.5,CircleID=circle,PntID=i) for i,v in enumerate(xyz)]
    output=handler(points,state)
    _checked_outputs(output,{_identity(point) for point in points})
    assert len({p['BatchID'] for p in output})==len(output)
    counts.append(len(output))
torch.cuda.synchronize()
report={'dense_input_points_per_circle':8812,'circles':6,'outputs':counts,
        'seconds':time.perf_counter()-started,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
        'stage2_checkpoint_step':saved['step'], 'original_point_checks':'passed',
        'note':'Synthetic resource check only, not detection accuracy or full deployment VRAM certification.'}
a.out.write_text(json.dumps(report,indent=2))
print(json.dumps(report))
