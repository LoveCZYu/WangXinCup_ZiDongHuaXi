"""Geometric neighborhoods stay separate from loss/label masks."""
import numpy as np
import torch
from torch.nn import functional as F

from experiments.competition_learning.model import set_loss
from experiments.competition_learning.samples import grouped_index
from experiments.set_association.model import geometry


def neighborhoods(data):
    ri, rm, supervised, positive = geometry(data)
    ci, cm = grouped_index(data['edge_col'],len(data['col_allowed']),32)
    assert np.all(np.bincount(data['edge_col'],minlength=len(ci)) <= 32), 'No hidden column truncation'
    unknown = np.bincount(data['edge_col'][~data['edge_known']],minlength=len(ci))
    cols = data['col_allowed'] & (unknown == 0) & cm.any(1)
    return dict(ri=ri,rm=rm,ci=ci,cm=cm,
        positive_rows=np.flatnonzero(supervised & positive),absent_rows=np.flatnonzero(supervised & ~positive),
        supervised_columns=np.flatnonzero(cols))


class Tables:
    def __init__(self, data, device):
        self.data = {k:torch.as_tensor(v,device=device) for k,v in data.items()}
        self.neighbors = {k:torch.as_tensor(v,device=device) for k,v in neighborhoods(data).items()}

    def batch(self):
        d,n = self.data,self.neighbors
        def choose(pool,size):return pool[torch.randint(len(pool),(size,),device=pool.device)]
        e = choose(d['known_ids'],512)
        r = torch.cat((choose(n['positive_rows'],128),choose(n['absent_rows'],128)))
        c = choose(n['supervised_columns'],128)
        ri,ci = n['ri'][r],n['ci'][c]
        edges,inverse = torch.unique(torch.cat((e,ri.flatten(),ci.flatten())),return_inverse=True)
        return e,r,c,edges,inverse,ri,ci

    def loss(self, model, batch, base):
        d,n = self.data,self.neighbors
        e,r,c,edges,inverse,ri,ci = batch
        er,ec = d['edge_row'][edges],d['edge_col'][edges]
        pair = model.edges(d['pair'][edges],d['pair'][n['ri'][er]],n['rm'][er],
            d['pair'][n['ci'][ec]],n['cm'][ec],base[edges])
        logits = pair[inverse]
        flat = logits[:len(e)]
        row = logits[len(e):len(e)+ri.numel()].reshape_as(ri)
        col = logits[len(e)+ri.numel():].reshape_as(ci)
        rv,cv = n['rm'][r],n['cm'][c]
        rp,cp = d['edge_positive'][ri] & rv,d['col_target'][ci] & cv
        binary = F.binary_cross_entropy_with_logits(torch.cat((flat,row[rv],col[cv])),
            torch.cat((d['edge_positive'][e].float(),rp[rv].float(),d['edge_positive'][ci][cv].float())))
        null = model.rejection(d['row'][r],d['pair'][ri],rv)
        row_loss = set_loss(torch.cat((row,null[:,None]),1),
            torch.cat((rv,torch.ones_like(null[:,None],dtype=torch.bool)),1),
            torch.cat((rp,~rp.any(1,keepdim=True)),1))
        col_loss = set_loss(torch.cat((col,col.new_zeros(len(col),1)),1),
            torch.cat((cv,torch.ones(len(col),1,device=col.device,dtype=torch.bool)),1),
            torch.cat((cp,~cp.any(1,keepdim=True)),1))
        return binary+.5*row_loss+.5*col_loss
