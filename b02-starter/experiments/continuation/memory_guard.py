"""Separate existing-ID memory quarantine; current outputs are not filtered."""
import numpy as np

from experiments.continuation.tracker import ContinuationTracker
from experiments.id_recovery.affinity import pack
from experiments.id_recovery.tracker import cutoff
from experiments.v3_tracking.memory import Memory,reliable


class MemoryGuardTracker(ContinuationTracker):
    def __init__(self,*args,guard=.90,**kwargs):
        super().__init__(*args,**kwargs)
        self.memory_guard=guard
        self.memory_decisions=[]

    def __call__(self,points,scores):
        self.memory_decisions=[]
        return super().__call__(points,scores)

    def update_memory(self,points,scores):
        self.new_anchors=set();self.trusted_updates={}
        events={event['batch']:event for event in self.events}
        pending=[];guarded=[];track_x=[];edge_x=[]
        for batch,index in self.last_matches.items():
            event=events.get(batch)
            if event is None:continue
            good,margin=reliable(event,index,scores[index],self.state['tracks'][batch].hits)
            if not good:continue
            pending.append((batch,index,margin))
            memory=self.memories.get(batch)
            if self.emissions[batch]>=3 and memory is not None and len(memory.entries)>=4:
                guarded.append((batch,index))
                track_x.append(event['track_features'])
                edge_x.append(event['edge_features'][event['indices'].index(index)])
        blocked=set()
        if guarded:
            logits=self.affinity.score(pack(np.stack(track_x),np.stack(edge_x)))
            self.counters['memory_updates_considered']+=len(guarded)
            for (batch,index),logit in zip(guarded,logits,strict=True):
                accepted=logit>=cutoff(self.memory_guard)
                self.memory_decisions.append((batch,index,bool(accepted)))
                if not accepted:blocked.add(batch)
        self.counters['memory_updates_blocked']+=len(blocked)
        for batch,index,margin in pending:
            if batch in blocked:continue
            memory=self.memories.setdefault(batch,Memory())
            if memory.add(points[index],self.current_context[index],self.state['scan'],scores[index],margin):
                self.new_anchors.add(batch)
            self.trusted_updates[batch]=index
        for batch in list(self.memories):
            if batch not in self.state['tracks']:del self.memories[batch]
