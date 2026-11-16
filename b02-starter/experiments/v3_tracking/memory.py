"""Bounded causal identity memory; selection uses pair evidence, never truth."""
from collections import deque
import numpy as np

from experiments.v2_association.features import normalized

FEATURE_VERSION='b02-association-memory-v3'
TRACK_DIM=98
EDGE_DIM=85
MEMORY_LENGTH=16


class Memory:
    def __init__(self):
        self.anchor=None;self.entries=deque(maxlen=MEMORY_LENGTH)

    def add(self,point,context,scan,score,margin):
        entry=(normalized(point),np.asarray(context,dtype=np.float32).copy(),scan,float(score),float(margin))
        fresh=self.anchor is None
        if fresh:self.anchor=entry
        self.entries.append(entry)
        return fresh

    def augment(self,track,edges,current,scan):
        n=len(self.entries);anchor=self.anchor[1] if self.anchor is not None else np.zeros(10,dtype=np.float32)
        if n:
            context=np.stack([e[1] for e in self.entries]);mean=context.mean(0);std=context.std(0)
            xyz=np.stack([e[0] for e in self.entries]);times=np.asarray([e[2]-scan for e in self.entries],dtype=np.float32)
            w=np.asarray([min(1.,e[4]) for e in self.entries],dtype=np.float32);w/=w.sum()
            dt=times-(times*w).sum();center=xyz-(xyz*w[:,None]).sum(0)
            velocity=(dt[:,None]*center*w[:,None]).sum(0)/max(float((dt**2*w).sum()),1e-5)
            last_age=(scan-self.entries[-1][2])/8;anchor_age=(scan-self.anchor[2])/100
            confidence=sum(min(1.,e[4]) for e in self.entries)/n
        else:
            mean=track[40:50];std=np.full(10,.1,dtype=np.float32);velocity=np.zeros(3,dtype=np.float32)
            last_age=4.;anchor_age=0.;confidence=0.
        metadata=np.asarray([n/MEMORY_LENGTH,min(4,last_age),min(4,anchor_age),confidence,float(n>0)],dtype=np.float32)
        tx=np.concatenate((track,anchor,mean,std,np.clip(velocity/4,-4,4),metadata)).astype(np.float32)
        elapsed=max(1.,float(track[38]*4))
        here=current
        extra=np.column_stack((here-anchor,here-mean,np.clip((here-mean)/(std+.02),-4,4),
            np.clip(edges[:,3:6]/elapsed-velocity[None,:]/4,-4,4),
            here[:,7]-anchor[7],here[:,8]*anchor[8]))
        ex=np.concatenate((edges,extra),axis=1).astype(np.float32)
        assert tx.shape==(TRACK_DIM,) and ex.shape==(len(edges),EDGE_DIM)
        assert np.isfinite(tx).all() and np.isfinite(ex).all()
        return tx,ex


def reliable(event,index,score,hits):
    if score<.5 or hits<3:return False,0.
    indices=event.get('indices',[])
    if index not in indices:return False,0.
    j=indices.index(index);costs=event.get('effective_costs',event['base_costs'])
    alternatives=[event.get('effective_dummy',event['dummy_cost'])]+[c for k,c in enumerate(costs) if k!=j]
    margin=min(alternatives)-costs[j]
    innovation=float(event.get('short_edge_features',event['edge_features'])[j,9])
    return margin>=.10 and innovation<=.5,float(max(0,margin))
