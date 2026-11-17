"""Construct the three fixed candidates, with optional non-mutating probes."""
from pathlib import Path
import json

import numpy as np

from experiments.continuation.fusion import MemoryFusionTracker
from experiments.continuation.learn import load_model
from experiments.continuation.memory_guard import MemoryGuardTracker
from experiments.continuation.tracker import ContinuationTracker
from experiments.id_recovery.affinity import load_affinity
from experiments.v2_association.run_stage1 import sha
from experiments.v3_tracking.runtime import load_scorer

ROOT=Path(__file__).resolve().parents[2]
MODES=('wide','memory','candidate')


class Probe:
    """Record retired candidates before decisions; no labels or state changes."""
    def __call__(self,points,scores):
        self.retired_edges=[]
        return super().__call__(points,scores)

    def reconnect_retired(self,positions,occupied,points,scores,scan):
        available=[i for i,s in enumerate(scores) if s>=self.gate.policy.enter and i not in occupied]
        if available:
            observations=np.asarray([positions[i] for i in available],dtype=np.float32)
            for batch,saved in sorted(self.retired.items()):
                age=scan-saved.track.last_scan
                predicted=np.asarray(saved.track.position)+np.asarray(saved.track.velocity)*age
                radius=min(1.+.10*(age-1),1.8)
                errors=np.square(observations-predicted).sum(1)/radius**2
                selected=[j for j in np.argsort(errors,kind='stable')[:6] if errors[j]<=.75]
                self.retired_edges.extend((batch,available[j]) for j in selected)
        return super().reconnect_retired(positions,occupied,points,scores,scan)


class Components:
    def __init__(self,prior,affinity,head,fusion,freeze):
        self.freeze=json.loads(Path(freeze).read_text(encoding='utf-8'))
        for key,path in (('prior',prior),('affinity',affinity),('head',head),('fusion',fusion)):
            assert sha(path)==self.freeze['artifacts'][key],key
        for name,digest in self.freeze['sources'].items():assert sha(ROOT/name)==digest,name
        self.scorer,memory,_,_=load_scorer(prior)
        assert memory
        self.affinity=load_affinity(affinity,prior)
        self.head,_=load_model(head,prior,affinity)
        self.fusion=json.loads(Path(fusion).read_text(encoding='utf-8'))
        assert self.fusion['amount']==1. and self.fusion['threshold']==.8

    def build(self,mode,probe=False):
        assert mode in MODES
        base={'wide':ContinuationTracker,'memory':MemoryGuardTracker,'candidate':MemoryFusionTracker}[mode]
        cls=type('Probed'+base.__name__,(Probe,base),{}) if probe else base
        if mode=='candidate':return cls(self.scorer,self.affinity,self.head,self.fusion,guard=.90)
        if mode=='memory':return cls(self.scorer,self.affinity,guard=.90)
        return cls(self.scorer,self.affinity)
