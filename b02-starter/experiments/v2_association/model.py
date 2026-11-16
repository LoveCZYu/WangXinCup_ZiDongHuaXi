"""A small residual edge-cost model initialized to reproduce frozen v1 exactly."""
import numpy as np
import torch
from torch import nn

from experiments.v2_association.features import TRACK_DIM,EDGE_DIM,MAX_EDGES


class AssociationModel(nn.Module):
    def __init__(self,track_dim=TRACK_DIM,edge_dim=EDGE_DIM):
        super().__init__()
        self.register_buffer('track_mean',torch.zeros(track_dim))
        self.register_buffer('track_std',torch.ones(track_dim))
        self.register_buffer('edge_mean',torch.zeros(edge_dim))
        self.register_buffer('edge_std',torch.ones(edge_dim))
        self.edge=nn.Sequential(nn.Linear(track_dim+edge_dim,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
        self.null=nn.Sequential(nn.Linear(track_dim,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.zeros_(self.edge[-1].weight);nn.init.zeros_(self.edge[-1].bias)
        nn.init.zeros_(self.null[-1].weight);nn.init.zeros_(self.null[-1].bias)

    def forward(self,track,edges):
        track=((track-self.track_mean)/self.track_std).clamp(-10,10)
        edges=((edges-self.edge_mean)/self.edge_std).clamp(-10,10)
        context=track[:,None,:].expand(-1,edges.shape[1],-1)
        real=self.edge(torch.cat((context,edges),dim=-1)).squeeze(-1)
        null=self.null(track).squeeze(-1)
        return torch.cat((real,null[:,None]),dim=1)


def association_loss(model,track,edges,base,valid,target,weights=None):
    residual=model(track,edges)
    logits=(-base/.25-residual).masked_fill(~valid,-1e9)
    correct=logits.masked_fill(~target,-1e9)
    loss=torch.logsumexp(logits,dim=1)-torch.logsumexp(correct,dim=1)
    if weights is not None:loss=loss*weights
    regularizer=(residual.square()*valid).sum()/valid.sum().clamp_min(1)
    return loss.mean()+1e-4*regularizer


class Scorer:
    def __init__(self,model,device='cpu',scale=1.,strategy='residual'):
        if strategy not in ('residual','rank_only','trusted_rank'):raise ValueError(strategy)
        self.model=model.to(device).eval();self.device=device;self.scale=scale;self.strategy=strategy

    @torch.no_grad()
    def __call__(self,events):
        n=len(events)
        tracks=np.stack([e['track_features'] for e in events])
        edges=np.zeros((n,MAX_EDGES,self.model.edge_mean.numel()),dtype=np.float32)
        for i,event in enumerate(events):edges[i,:len(event['indices'])]=event['edge_features']
        delta=self.model(torch.as_tensor(tracks,device=self.device),
                         torch.as_tensor(edges,device=self.device)).cpu().numpy()
        result=[]
        for i,event in enumerate(events):
            real=delta[i,:len(event['indices'])];null=float(delta[i,-1])
            if self.strategy!='residual':
                # Preserve the dummy and sole-candidate decisions. Penalize only
                # alternatives relative to the model's preferred candidate.
                real=np.clip(real-real.min(),0,4) if len(real)>1 else np.zeros_like(real)
                null=0.
                if self.strategy=='trusted_rank' and event['track_features'][37]*6<3:
                    real=np.zeros_like(real)
            result.append((real*self.scale,null*self.scale))
        return result
