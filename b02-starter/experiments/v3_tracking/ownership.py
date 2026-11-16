"""Learn competition across tracks for one point, preserving prior column strength."""
from collections import defaultdict
import numpy as np

from experiments.v3_tracking.tracker import Tracker

FEATURE_VERSION='b02-column-ownership-v1'
TRACK_DIM=101
EDGE_DIM=88


def extended_features(events,tracker,prior):
    references=defaultdict(list)
    for r,event in enumerate(events):
        for j,index in enumerate(event['indices']):references[index].append((r,j))
    tracks=[];edges=[];costs=[]
    for r,event in enumerate(events):
        batch=event['batch'];emitted=tracker.emitted.get(batch,0);hits=tracker.state['tracks'][batch].hits
        age=(tracker.state['scan']-tracker.last_emitted.get(batch,tracker.state['scan']-16))/4
        tx=np.concatenate((event['track_features'],np.asarray([min(4,emitted/50),min(4,age),emitted/max(1,hits)],dtype=np.float32)))
        real=[c+.25*float(d) for c,d in zip(event['base_costs'],prior[r][0])]
        extra=[]
        for j,index in enumerate(event['indices']):
            competitors=references[index]
            maximum=max(tracker.emitted.get(events[k]['batch'],0) for k,_ in competitors)
            extra.append([min(4,len(competitors)/8),emitted/max(1,maximum),real[j]/1.5])
        ex=np.column_stack((event['edge_features'],np.asarray(extra,dtype=np.float32)))
        assert tx.shape==(TRACK_DIM,) and ex.shape==(len(real),EDGE_DIM)
        tracks.append(tx);edges.append(ex);costs.append(real)
    return np.stack(tracks),edges,costs,references


def ownership_costs(prior,delta,references,scores):
    """Column translation preserves best feasible prior cost, including at zero init."""
    penalties=[np.zeros(len(x),dtype=np.float64) for x in prior]
    shifts={}
    for index,refs in references.items():
        if len(refs)<2 or scores[index]<.5:continue
        minimum=min(float(delta[r][j]) for r,j in refs)
        for r,j in refs:penalties[r][j]=.25*min(4.,max(0.,float(delta[r][j])-minimum))
        old=min(prior[r][j] for r,j in refs)
        new=min(prior[r][j]+penalties[r][j] for r,j in refs)
        shifts[index]=new-old
    adjusted=[np.asarray(x,dtype=np.float64).copy() for x in prior]
    corrections=[np.zeros(len(x),dtype=np.float64) for x in prior]
    for index,shift in shifts.items():
        for r,j in references[index]:
            corrections[r][j]=penalties[r][j]-shift;adjusted[r][j]+=corrections[r][j]
    return adjusted,corrections


class OwnershipTracker(Tracker):
    def __init__(self,prior,model=None,**kwargs):
        super().__init__(prior,memory=True,**kwargs)
        self.ownership_model=model;self.emitted={};self.last_emitted={};self.graph_capture=False;self.graphs=[];self.current_scores=[]

    def score_events(self,events):
        capture=self.feature_capture
        need=self.graph_capture or self.ownership_model is not None
        if need:self.feature_capture=True
        try:prior=super().score_events(events)
        finally:self.feature_capture=capture
        if not need:return prior
        tx,ex,costs,references=extended_features(events,self,prior)
        if self.graph_capture:
            self.graphs.append({'track':tx,'edges':ex,'prior':costs,'batches':[e['batch'] for e in events],
                'indices':[list(e['indices']) for e in events]})
        if self.ownership_model is None:return prior
        import torch
        padded=np.zeros((len(events),12,EDGE_DIM),dtype=np.float32)
        for r,array in enumerate(ex):padded[r,:len(array)]=array
        with torch.no_grad():delta=self.ownership_model(torch.as_tensor(tx),torch.as_tensor(padded)).numpy()[:,:-1]
        adjusted,correction=ownership_costs(costs,delta,references,self.current_scores)
        return [(np.asarray(old[0],dtype=np.float64)+correction[r]*4,old[1]) for r,old in enumerate(prior)]

    def __call__(self,points,scores):
        self.graphs=[];self.current_scores=scores
        outputs=super().__call__(points,scores)
        for p in outputs:
            batch=p['BatchID'];self.emitted[batch]=self.emitted.get(batch,0)+1;self.last_emitted[batch]=self.state['scan']
        for mapping in (self.emitted,self.last_emitted):
            for batch in list(mapping):
                if batch not in self.state['tracks']:del mapping[batch]
        return outputs


class OwnershipBundle:
    """Shared immutable weights; each build creates fresh tracking/identity state."""
    def __init__(self,prior,model):self.prior=prior;self.model=model.eval()

    def build_tracker(self):return OwnershipTracker(self.prior,self.model)
