"""Past-only physical-state correction after an untrusted observation."""
from collections import Counter
import numpy as np

from experiments.continuation.fusion import MemoryFusionTracker
from experiments.candidate_onset.observe import ObservedTracker
from experiments.identity_competition.motion import memory_motion

MODES=('baseline','position','velocity','both')


def correction(tracker,batch,scan):
    track=tracker.state['tracks'][batch]
    memory=tracker.memories.get(batch)
    if tracker.emissions[batch]<3 or memory is None or len(memory.entries)<4:return None
    last=memory.entries[-1][2]
    if last>=track.last_scan or not 1<=scan-last<=3:return None
    motion=memory_motion(memory,track.last_scan)
    if motion is None:return None
    xyz=np.stack([e[0] for e in memory.entries]).astype(float)
    time=np.asarray([e[2]-track.last_scan for e in memory.entries],dtype=float)
    if np.ptp(time)<3:return None
    residual=xyz-motion[0]-time[:,None]*motion[1]
    if float(np.square(residual).sum(1).mean())>.5:return None
    if not np.isfinite(np.concatenate(motion)).all():return None
    return motion


class RebaseMixin:
    def assign(self,track_ids,scan,positions,grid,occupied,points,scores):
        if not self.assignment_stage and self.rebase_mode!='baseline':
            if not hasattr(self,'motion_counts'):self.motion_counts=Counter()
            for batch in track_ids:
                motion=correction(self,batch,scan)
                if motion is None:continue
                track=self.state['tracks'][batch]
                if self.rebase_mode in ('position','both'):track.position=tuple(motion[0])
                if self.rebase_mode in ('velocity','both'):track.velocity=tuple(motion[1])
                self.motion_counts['rebased']+=1
        return super().assign(track_ids,scan,positions,grid,occupied,points,scores)


class RebaseTracker(RebaseMixin,MemoryFusionTracker):pass
class RebaseObservedTracker(RebaseMixin,ObservedTracker):pass


def install(tracker,mode):
    assert mode in MODES
    if isinstance(tracker,ObservedTracker):tracker.__class__=RebaseObservedTracker
    else:
        assert type(tracker) is MemoryFusionTracker
        tracker.__class__=RebaseTracker
    tracker.rebase_mode=mode;tracker.motion_counts=Counter()
    return tracker
