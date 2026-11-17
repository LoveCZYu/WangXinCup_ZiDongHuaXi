"""Non-mutating stage instrumentation; no truth labels enter this module."""
from experiments.continuation.fusion import MemoryFusionTracker
from experiments.robust_validation.frozen import Probe


def track_snapshot(tracker,batch,track,*,saved=None,bank='active'):
    history = tuple(saved.history if saved is not None else tracker.history.get(batch,()))
    memory = saved.memory if saved is not None else tracker.memories.get(batch)
    return dict(position=tuple(track.position),velocity=tuple(track.velocity),last_scan=track.last_scan,hits=track.hits,
        emissions=int(saved.emission_count if saved is not None else tracker.emissions[batch]),
        last_emission=saved.last_emission if saved is not None else tracker.last_emission.get(batch),
        history=history,memory=tuple(memory.entries) if memory is not None else (),bank=bank)


def snapshot(tracker):
    result={b:track_snapshot(tracker,b,t) for b,t in tracker.state.get('tracks',{}).items()}
    for name in ('retired','short_retired'):
        for b,saved in getattr(tracker,name).items():
            if b not in result:result[b]=track_snapshot(tracker,b,saved.track,saved=saved,bank=name)
    return result


class ObservedTracker(Probe,MemoryFusionTracker):
    def __call__(self,points,scores):
        self.stage_records=[];self.retired_calls=[]
        return super().__call__(points,scores)

    def assign(self,track_ids,scan,positions,grid,occupied,points,scores):
        record=dict(stage=self.assignment_stage,track_ids=tuple(track_ids),occupied_before=tuple(occupied),
            states={b:track_snapshot(self,b,self.state['tracks'][b]) for b in track_ids})
        start=len(self.events)
        result=super().assign(track_ids,scan,positions,grid,occupied,points,scores)
        record.update(events=list(self.events[start:]),matches=dict(result))
        self.stage_records.append(record)
        return result

    def reconnect_retired(self,positions,occupied,points,scores,scan):
        self.retired_calls.append(dict(occupied=tuple(occupied),ids=tuple(self.retired),
            states={b:track_snapshot(self,b,s.track,saved=s,bank='retired') for b,s in self.retired.items()}))
        return super().reconnect_retired(positions,occupied,points,scores,scan)


def build(components):
    return ObservedTracker(components.scorer,components.affinity,components.head,components.fusion,guard=.90)
