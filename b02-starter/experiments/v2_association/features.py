"""Causal track/point edge features. No labels, IDs, or absolute time as features."""
import numpy as np

from b02.algorithm import DEFAULT_CONFIG, POSITION

FEATURE_VERSION='b02-association-features-v1'
TRACK_DIM=40
EDGE_DIM=20
MAX_EDGES=12


def normalized(point):
    return np.asarray([point[k]/g for k,g in zip(POSITION,DEFAULT_CONFIG.gate)],dtype=np.float32)


def features(track, history, scan, birth_scan, strong_hits, quality_hits,
             points, scores, indices, base_costs, radius):
    """history contains only already matched observations before this scan."""
    t=[];masks=[]
    last=history[-1][0]
    last_position=normalized(last)
    selected=history[-4:]
    for i in range(4):
        if i < 4-len(selected):
            t.extend([0.]*7);masks.append(0.)
        else:
            p,score,observed=selected[i-(4-len(selected))]
            t.extend(np.clip((normalized(p)-last_position)/4,-4,4).tolist())
            t.extend((min(4,(scan-observed)/8),score,p['Amp'],p['SNR']))
            masks.append(1.)
    t.extend(masks)
    t.extend(np.clip(np.asarray(track.velocity)/4,-4,4).tolist())
    t.extend((min(4,(scan-birth_scan)/100),min(4,track.hits/10),strong_hits/6,
              min(4,(scan-track.last_scan)/4),min(4,quality_hits/3)))
    track_x=np.asarray(t,dtype=np.float32)
    if not indices:
        return track_x,np.empty((0,EDGE_DIM),dtype=np.float32)
    position=np.asarray(track.position,dtype=np.float32)
    velocity=np.asarray(track.velocity,dtype=np.float32)
    elapsed=scan-track.last_scan
    xyz=np.stack([normalized(points[i]) for i in indices])
    error=xyz-(position+velocity*elapsed)
    displacement=xyz-position
    current=np.asarray([[points[i]['Amp'],points[i]['SNR']] for i in indices],dtype=np.float32)
    last_signal=np.asarray([last['Amp'],last['SNR']],dtype=np.float32)
    mean_signal=np.mean([[p['Amp'],p['SNR']] for p,_,_ in history],axis=0)
    dt=np.asarray([(points[i]['Time']-last['Time'])/5000/4 for i in indices],dtype=np.float32)
    edge=np.column_stack((np.clip(error/4,-4,4),np.clip(displacement/4,-4,4),
        np.clip((displacement/elapsed-velocity)/4,-4,4),
        np.sum(error**2,axis=1)/radius**2,np.asarray(base_costs)/1.5,
        np.asarray([scores[i] for i in indices]),current,current-last_signal,
        current-mean_signal,np.clip(dt,-4,4),np.full(len(indices),len(indices)/MAX_EDGES))).astype(np.float32)
    assert track_x.shape==(TRACK_DIM,) and edge.shape==(len(indices),EDGE_DIM)
    assert np.isfinite(track_x).all() and np.isfinite(edge).all()
    return track_x,edge
