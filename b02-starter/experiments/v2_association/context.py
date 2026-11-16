"""Causal local group descriptors from confident current points; no numeric IDs."""
import numpy as np
from experiments.v2_association.features import normalized

FEATURE_VERSION='b02-association-context-v2'
TRACK_DIM=60
EDGE_DIM=50
CONTEXT_DIM=10


def descriptors(points,scores):
    result=np.zeros((len(points),CONTEXT_DIM),dtype=np.float32)
    selected=[i for i,s in enumerate(scores) if s>=.5]
    if not selected:return result
    xyz=np.stack([normalized(points[i]) for i in selected]).astype(np.float64)
    distance=((xyz[:,None,:]-xyz[None,:,:])**2).sum(-1)
    links=distance<=1.
    visited=set()
    for start in range(len(selected)):
        if start in visited:continue
        group=[];pending=[start];visited.add(start)
        while pending:
            j=pending.pop();group.append(j)
            for k in np.flatnonzero(links[j]):
                k=int(k)
                if k not in visited:visited.add(k);pending.append(k)
        # Geometric/signal reductions, with no input-order or point-ID feature.
        group.sort(key=lambda j:tuple(xyz[j])+tuple(points[selected[j]][k] for k in ('Amp','SNR')))
        indices=[selected[j] for j in group];positions=xyz[group]
        signal=np.asarray([[points[i]['Amp'],points[i]['SNR']] for i in indices])
        relative=positions-positions.mean(0);centered=signal-signal.mean(0)
        rms=float(np.sqrt(np.square(relative).sum(1).mean()))
        for j,index in enumerate(indices):
            rank_amp=float(np.sign(signal[j,0]-signal[:,0]).sum())/max(1,len(group)-1)
            rank_el=float(np.sign(positions[j,1]-positions[:,1]).sum())/max(1,len(group)-1)
            result[index]=[*relative[j],*centered[j],rank_amp,rank_el,min(1,len(group)/8),float(len(group)>1),rms]
    assert np.isfinite(result).all()
    return result


def augment(track_features,edge_features,history,current,indices):
    previous=np.asarray(history[-4:],dtype=np.float32)
    last=previous[-1];mean=previous.mean(0)
    tx=np.concatenate((track_features,last,mean))
    if indices:
        here=current[indices]
        ex=np.concatenate((edge_features,here,here-last,here-mean),axis=1)
    else:ex=np.empty((0,EDGE_DIM),dtype=np.float32)
    assert tx.shape==(TRACK_DIM,) and ex.shape==(len(indices),EDGE_DIM)
    return tx,ex
