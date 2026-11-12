"""Integration checks against the isolated upstream fork (CPU or CUDA)."""
import argparse
from pathlib import Path
import sys

p=argparse.ArgumentParser()
p.add_argument('--upstream',type=Path,required=True)
p.add_argument('--project',type=Path,required=True)
p.add_argument('--pretrained',type=Path,required=True)
p.add_argument('--device',default='cuda')
a=p.parse_args()
sys.path[:0]=[str(a.upstream.resolve()),str(a.project.resolve())]
import numpy as np
import torch
from adapter import make_model,transfer,make_window,associate,coordinates,select_points,IgnoreUnknownCriterion
from track_mt3.matching.cta import CTAResult
from b02.replay import _checked_outputs,_identity

torch.set_num_threads(4);torch.manual_seed(1)
points=[dict(Time=0,Azimuth=.2,Distance=.4,Elevation=e,Amp=.5,SNR=.5,CircleID=0,PntID=i)
        for i,e in enumerate([.1,.7])]
out=associate(coordinates(points),[0,9],points)
assert [v['BatchID'] for v in out]==[1,10]
_checked_outputs(out,{_identity(point) for point in points})
assert not associate(np.array([[.9,.9,.9]]),[0],points)
assert len(associate(coordinates([points[0],points[0]]),[0,1],points))==1
assert select_points(points)==select_points(list(reversed(points)))

model=make_model(a.upstream,a.device).eval()
report=transfer(model,a.pretrained)
print({'transfer_fraction':report['copied_parameter_elements']/report['total_parameters']},flush=True)
labelled=make_window([(0,points,{0:7,1:-1})],0,{0:7,1:-1})
clean=make_window([(0,points,None)],0)
assert torch.equal(labelled.measurements,clean.measurements)
with torch.no_grad():
    first=model.forward_window(clean)
    second=model.forward_window(labelled)
    torch.testing.assert_close(first.positions,second.positions,rtol=0,atol=0)
    torch.testing.assert_close(first.logits,second.logits,rtol=0,atol=0)
    empty=model.forward_window(make_window([(0,[],None)],0))
    assert empty.positions.shape[1]==3 and torch.isfinite(empty.positions).all()
    # Ensure adding a future frame to the dataset cannot alter an earlier window.
    assert clean.time_indices.tolist()==[19,19]

criterion=IgnoreUnknownCriterion(model.config.loss)
criterion.set_window(labelled)
positions=torch.tensor([[.2,.4,.7],[.8,.8,.8]],device=a.device,requires_grad=True)
logits=torch.zeros(2,1,device=a.device,requires_grad=True)
emptyidx=torch.empty(0,dtype=torch.long,device=a.device)
assignment=CTAResult(emptyidx,emptyidx,torch.full((2,),-1,device=a.device),0,0,emptyidx)
_,loss=criterion._layer_loss(positions,logits,positions[:0],assignment)
loss.backward()
assert logits.grad[0]==0 and logits.grad[1]>0

model.train()
criterion.set_window(labelled)
total,*_=model([[labelled,labelled]],criterion=criterion,teacher_forcing_probability=.5)
assert torch.isfinite(total)
total.backward()
assert model.encoder.input_projection.weight.grad is not None
assert torch.isfinite(model.encoder.input_projection.weight.grad).all()
assert model.encoder.input_projection.weight.grad[:,2].abs().sum()>0
print('PASS: 3D transfer, label-free inference, empty scans, original-point uniqueness, unknown masking, backward to elevation.',flush=True)
