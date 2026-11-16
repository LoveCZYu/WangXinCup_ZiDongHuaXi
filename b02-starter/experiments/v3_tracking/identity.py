"""Causal logical identities over an unchanged set of selected original points."""
from collections import defaultdict
from math import floor

from b02.algorithm import POSITION,Track
from experiments.v3_tracking.tracker import Tracker


class IdentityManager:
    """Detection/output selection stays in the upstream tracker; only IDs change."""
    def __init__(self,scorer=None,max_gap=4,memory=False):
        if max_gap not in (4,8):raise ValueError('Preregistered identity gaps are 4 or 8')
        self.engine=Tracker(scorer,memory=memory);self.max_gap=max_gap

    def __call__(self,points,scores,selected,context):
        engine=self.engine;scan=engine.state.get('scan',0)+1;engine.state['scan']=scan
        tracks=engine.state.setdefault('tracks',{});next_id=engine.state.get('next_id',1)
        engine.events=[];engine.last_matches={};engine.last_births={};engine.deleted=[]
        engine.current_context=context
        lookup={p['PntID']:i for i,p in enumerate(points)}
        if len(lookup)!=len(points):raise ValueError('Input point IDs must be unique within scan')
        indices=[lookup[p['PntID']] for p in selected]
        if len(set(indices))!=len(indices):raise ValueError('Selected original points must be unique')
        positions=[tuple(p[k]/g for k,g in zip(POSITION,engine.config.gate)) for p in points]
        grid=defaultdict(list)
        for i in indices:grid[tuple(map(floor,positions[i]))].append(i)
        # A single joint assignment; histories belong to persistent logical IDs.
        occupied=set();matches=engine.assign(list(tracks),scan,positions,grid,occupied,points,scores)
        engine.last_matches=matches.copy();assigned={}
        for batch,track in list(tracks.items()):
            index=matches.get(batch);elapsed=scan-track.last_scan
            if index is None:
                if elapsed>self.max_gap:del tracks[batch];engine.deleted.append(batch)
                continue
            position=positions[index];gain=1. if track.hits==1 else engine.config.velocity_gain
            updated=tuple((1-gain)*v+gain*(x-previous)/elapsed for v,x,previous in zip(track.velocity,position,track.position))
            trust=min(1.,scores[index]/.5)
            track.velocity=tuple(v+trust*(new-v) for v,new in zip(track.velocity,updated))
            track.position=position;track.last_scan=scan;track.hits+=1
            engine.strong_hits[batch]=min(6,engine.strong_hits.get(batch,0)+int(scores[index]>=.5))
            engine.quality_hits[batch]=engine.quality_hits.get(batch,0)+trust
            engine.history[batch].append((points[index],scores[index],scan));engine.history[batch]=engine.history[batch][-8:]
            engine.context_history[batch].append(context[index]);engine.context_history[batch]=engine.context_history[batch][-8:]
            assigned[index]=batch
        for index in indices:
            if index in occupied:continue
            batch=next_id;next_id+=1;tracks[batch]=Track(positions[index],scan)
            engine.strong_hits[batch]=int(scores[index]>=.5);engine.quality_hits[batch]=min(1.,scores[index]/.5)
            engine.birth_scan[batch]=scan;engine.history[batch]=[(points[index],scores[index],scan)]
            engine.context_history[batch]=[context[index]];engine.last_births[batch]=index;assigned[index]=batch
        for mapping in (engine.strong_hits,engine.quality_hits,engine.birth_scan,engine.history,engine.context_history):
            for batch in list(mapping):
                if batch not in tracks:del mapping[batch]
        engine.state['next_id']=next_id
        if engine.extra_memory:engine.update_memory(points,scores)
        output=[{**point,'BatchID':assigned[index]} for point,index in zip(selected,indices)]
        assert len({p['BatchID'] for p in output})==len(output)
        engine.last_raw=output
        return output
