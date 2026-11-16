"""v1-compatible tracking seam with four preregistered causal interventions."""
from collections import defaultdict
from heapq import nsmallest
from itertools import product
from math import floor

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from b02.algorithm import DEFAULT_CONFIG, POSITION, Track, _output
from experiments.point_scorer_expanded.policy import OutputGate, Policy
from experiments.v2_association.features import features

VARIANTS=('v1','position_trust','trusted_confirmation','conservative_dummy')


class Tracker:
    def __init__(self,variant='v1',policy=None,*,capture=False,feature_capture=False,scorer=None,contextual=False):
        if variant not in VARIANTS:raise ValueError(variant)
        self.variant=variant;self.config=DEFAULT_CONFIG
        self.state={};self.strong_hits={};self.quality_hits={};self.birth_scan={};self.history={}
        self.gate=OutputGate(policy or Policy('frozen_v1',.525,.05,3))
        self.capture=capture;self.feature_capture=feature_capture;self.scorer=scorer
        self.contextual=contextual;self.context_history={};self.current_context=None
        self.events=[];self.last_matches={};self.last_births={};self.last_raw=[];self.deleted=[]

    def confirmed(self,batch):
        t=self.state['tracks'][batch]
        if self.variant=='trusted_confirmation':
            return t.hits>=3 and self.quality_hits.get(batch,0)>=3
        return t.hits>=self.config.confirm_hits

    def assign(self,track_ids,scan,positions,grid,occupied,points,scores):
        tracks=self.state['tracks'];rows=[];columns=[];costs=[];pending=[];point_columns={};events=[]
        for batch in track_ids:
            track=tracks[batch];elapsed=scan-track.last_scan
            predicted=tuple(x+v*elapsed for x,v in zip(track.position,track.velocity))
            radius=min(1+.2*(elapsed-1),1.6)
            neighborhood=[range(floor(x-radius),floor(x+radius)+1) for x in predicted]
            edges=[];gated=[]
            for cell in product(*neighborhood):
                for index in grid.get(cell,()):
                    if index in occupied:continue
                    distance=sum((a-b)**2 for a,b in zip(predicted,positions[index]))
                    if distance<=radius**2:
                        gated.append(index)
                        cost=distance/radius**2+.50*(1-scores[index])
                        if cost<1.15:edges.append((cost,index))
            penalty=.15*(1-min(self.strong_hits.get(batch,0),6)/6)
            chosen=nsmallest(self.config.candidates_per_track,edges)
            indices=[i for _,i in chosen];base=[c+penalty for c,_ in chosen]
            dummy=1.15
            if self.variant=='conservative_dummy':
                dummy-=.50*(1-min(self.strong_hits.get(batch,0),6)/6)
            event={'batch':batch,'indices':indices,'base_costs':base,'dummy_cost':dummy,
                   'gated_indices':gated,'eligible_indices':[i for _,i in edges]}
            if self.feature_capture or self.scorer is not None:
                tx,ex=features(track,self.history[batch],scan,self.birth_scan[batch],
                               self.strong_hits.get(batch,0),self.quality_hits.get(batch,0),
                               points,scores,indices,base,radius)
                if self.contextual:
                    from experiments.v2_association.context import augment
                    tx,ex=augment(tx,ex,self.context_history[batch],self.current_context,indices)
                event.update(track_features=tx,edge_features=ex)
            if self.capture:self.events.append(event)
            if not chosen:continue
            pending.append(batch);events.append(event)
        if not pending:return {}
        deltas=self.scorer(events) if self.scorer is not None else None
        for row,event in enumerate(events):
            real=event['base_costs'];dummy=event['dummy_cost']
            if deltas is not None:
                edge_delta,null_delta=deltas[row]
                real=[c+.25*float(d) for c,d in zip(real,edge_delta,strict=True)]
                dummy+=.25*float(null_delta)
            shift=max(0.,-min([dummy,*real]))
            event['effective_costs']=real;event['effective_dummy']=dummy
            for cost,index in zip(real,event['indices'],strict=True):
                rows.append(row);columns.append(point_columns.setdefault(index,len(point_columns)))
                costs.append(cost+shift+1e-8)
            event['shift']=shift
        point_count,count=len(point_columns),len(pending)
        rows.extend(range(count));columns.extend(range(point_count,point_count+count))
        costs.extend(e['effective_dummy']+e['shift']+1e-8 for e in events)
        matrix=coo_matrix((costs,(rows,columns)),shape=(count,point_count+count)).tocsr()
        ri,ci=min_weight_full_bipartite_matching(matrix)
        indices=list(point_columns)
        matched={pending[r]:indices[c] for r,c in zip(ri,ci) if c<point_count}
        occupied.update(matched.values())
        return matched

    def __call__(self,points,scores):
        if len(points)!=len(scores):raise ValueError('Point/score count mismatch')
        scan=self.state.get('scan',0)+1;self.state['scan']=scan
        tracks=self.state.setdefault('tracks',{});next_id=self.state.get('next_id',1)
        self.events=[];self.last_births={};self.deleted=[]
        if self.contextual:
            from experiments.v2_association.context import descriptors
            self.current_context=descriptors(points,scores)
        positions=[tuple(p[k]/scale for k,scale in zip(POSITION,self.config.gate)) for p in points]
        grid=defaultdict(list)
        for i,pos in enumerate(positions):grid[tuple(map(floor,pos))].append(i)
        confirmed=[b for b in tracks if self.confirmed(b)]
        tentative=[b for b in tracks if not self.confirmed(b)]
        occupied=set()
        matches=self.assign(confirmed,scan,positions,grid,occupied,points,scores)
        matches.update(self.assign(tentative,scan,positions,grid,occupied,points,scores))
        self.last_matches=matches.copy();outputs=[]
        for batch,track in list(tracks.items()):
            index=matches.get(batch);elapsed=scan-track.last_scan
            if index is None:
                if not self.confirmed(batch) or elapsed>self.config.max_missed:
                    del tracks[batch];self.deleted.append(batch)
                continue
            position=positions[index];old_velocity=track.velocity;old_position=track.position
            gain=1.0 if track.hits==1 else self.config.velocity_gain
            updated=tuple((1-gain)*v+gain*(x-previous)/elapsed
                          for v,x,previous in zip(old_velocity,position,old_position))
            trust=min(1,scores[index]/.50)
            track.velocity=tuple(v+trust*(new-v) for v,new in zip(old_velocity,updated))
            if self.variant=='position_trust':
                predicted=tuple(x+v*elapsed for x,v in zip(old_position,old_velocity))
                position=tuple(x+trust*(z-x) for x,z in zip(predicted,position))
            track.position=position;track.last_scan=scan;track.hits+=1
            self.quality_hits[batch]=self.quality_hits.get(batch,0)+trust
            self.strong_hits[batch]=min(6,self.strong_hits.get(batch,0)+int(scores[index]>=.50))
            self.history[batch].append((points[index],scores[index],scan))
            self.history[batch]=self.history[batch][-8:]
            if self.contextual:
                self.context_history[batch].append(self.current_context[index])
                self.context_history[batch]=self.context_history[batch][-8:]
            if self.confirmed(batch):outputs.append(_output(points[index],batch))
        capacity=self.config.max_tentative-sum(not self.confirmed(b) for b in tracks)
        free=(i for i in range(len(points)) if i not in occupied)
        for index in nsmallest(capacity,free,key=lambda i:(-points[i]['Amp'],i)):
            tracks[next_id]=Track(positions[index],scan)
            self.strong_hits[next_id]=int(scores[index]>=.50)
            self.quality_hits[next_id]=min(1,scores[index]/.50)
            self.birth_scan[next_id]=scan;self.history[next_id]=[(points[index],scores[index],scan)]
            if self.contextual:self.context_history[next_id]=[self.current_context[index]]
            self.last_births[next_id]=index;next_id+=1
        for mapping in (self.strong_hits,self.quality_hits,self.birth_scan,self.history,self.context_history):
            for batch in list(mapping):
                if batch not in tracks:del mapping[batch]
        self.state['next_id']=next_id;self.last_raw=outputs
        score_map=dict(zip((p['PntID'] for p in points),scores))
        return self.gate.apply(outputs,score_map,scan-1)
