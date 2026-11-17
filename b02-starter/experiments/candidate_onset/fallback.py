"""Supplement only unmatched zero-candidate IDs; protect existing matches."""
from collections import Counter,defaultdict
import numpy as np

from experiments.candidate_onset.observe import ObservedTracker
from experiments.candidate_onset.policy import correction
from experiments.continuation.fusion import MemoryFusionTracker
from experiments.id_recovery.affinity import pack
from experiments.id_recovery.tracker import cutoff
from experiments.v2_association.tracker import Tracker as CoreTracker

MODES=('baseline','backup_motion','backup_guarded')


class BackupScorer:
    def __init__(self,prior,affinity):
        self.model=prior.model;self.affinity=affinity;self.counts=Counter()

    def __call__(self,events):
        packed=np.concatenate([pack(np.repeat(e['track_features'][None,:],len(e['indices']),axis=0),e['edge_features']) for e in events])
        raw=self.affinity.score(packed)
        rows=[];columns=defaultdict(list);offset=0
        for r,e in enumerate(events):
            n=len(e['indices'])
            strength=np.clip(raw[offset:offset+n],-6,6)-1.2*e['edge_features'][:,9]
            offset+=n;rows.append(strength)
            for j,index in enumerate(e['indices']):columns[index].append((r,j,float(strength[j])))
        result=[]
        for r,(e,strength) in enumerate(zip(events,rows,strict=True)):
            desired=np.full(len(strength),1.,np.float32)
            rank=np.argsort(-strength,kind='stable');j=int(rank[0]);best=float(strength[j])
            row_margin=best-float(strength[rank[1]]) if len(rank)>1 else float('inf')
            rivals=[s for other,_,s in columns[e['indices'][j]] if other!=r]
            col_margin=best-max(rivals,default=-1e4)
            self.counts['candidate_edges']+=len(strength)
            if best>=cutoff(.95) and row_margin>=.25 and col_margin>=.25:
                desired[j]=cutoff(.95)-best;self.counts['accepted_proposals']+=1
            result.append(((desired-np.asarray(e['base_costs']))/.25,-e['dummy_cost']/.25))
        return result


class BackupMixin:
    def assign(self,track_ids,scan,positions,grid,occupied,points,scores):
        stage=self.assignment_stage
        matched=super().assign(track_ids,scan,positions,grid,occupied,points,scores)
        if stage or self.backup_mode=='baseline':return matched
        recorded={e['batch']:e for e in self.events}
        candidates=[];saved={}
        for batch in track_ids:
            if batch in matched or batch not in self.state['tracks']:continue
            if recorded.get(batch,{}).get('indices'):continue
            motion=correction(self,batch,scan)
            if motion is None:continue
            track=self.state['tracks'][batch]
            saved[batch]=track.position;track.position=tuple(motion[0]);candidates.append(batch)
        if not candidates:return matched
        self.motion_counts['backup_attempted_ids']+=len(candidates)
        original_occupied=set(occupied)
        restricted=original_occupied|{i for i,s in enumerate(scores) if s<self.gate.policy.enter}
        original_scorer=self.model_scorer
        guard=BackupScorer(original_scorer,self.affinity) if self.backup_mode=='backup_guarded' else None
        if guard is not None:self.model_scorer=guard
        try:
            extra=CoreTracker.assign(self,candidates,scan,positions,grid,restricted,points,scores)
        finally:self.model_scorer=original_scorer
        if guard is not None:self.motion_counts.update({'guard_'+k:v for k,v in guard.counts.items()})
        assert not set(extra)&set(matched) and not set(extra.values())&original_occupied
        for batch,position in saved.items():
            if batch not in extra:self.state['tracks'][batch].position=position
        self.motion_counts['backup_recovered']+=len(extra)
        occupied.update(extra.values());matched.update(extra)
        return matched


class BackupTracker(BackupMixin,MemoryFusionTracker):pass
class BackupObservedTracker(BackupMixin,ObservedTracker):pass


def install(tracker,mode):
    assert mode in MODES
    if isinstance(tracker,ObservedTracker):tracker.__class__=BackupObservedTracker
    else:
        assert type(tracker) is MemoryFusionTracker
        tracker.__class__=BackupTracker
    tracker.backup_mode=mode;tracker.motion_counts=Counter()
    return tracker
