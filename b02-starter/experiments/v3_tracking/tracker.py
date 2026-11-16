"""Frozen-v2 tracking with optional extra memory features; physical state unchanged."""
import numpy as np
from experiments.v2_association.tracker import Tracker as V2Tracker
from experiments.v3_tracking.memory import Memory,TRACK_DIM,reliable


class Tracker(V2Tracker):
    def __init__(self,scorer=None,*,memory=False,capture=True,feature_capture=False):
        self.extra_memory=memory;self.model_scorer=scorer;self.memories={}
        self.new_anchors=set();self.trusted_updates={}
        super().__init__('v1',contextual=True,capture=capture,feature_capture=feature_capture,
            scorer=self.score_events if scorer is not None or memory else None)

    def score_events(self,events):
        use_memory=self.model_scorer is not None and self.model_scorer.model.track_mean.numel()==TRACK_DIM
        short=[]
        for event in events:
            tx,ex=event['track_features'],event['edge_features']
            event['short_track_features']=tx;event['short_edge_features']=ex
            short.append({**event,'track_features':tx,'edge_features':ex})
            if self.extra_memory and (self.feature_capture or use_memory):
                m=self.memories.setdefault(event['batch'],Memory())
                event['track_features'],event['edge_features']=m.augment(tx,ex,self.current_context[event['indices']],self.state['scan'])
        if self.model_scorer is None:return [(np.zeros(len(e['indices']),dtype=np.float32),0.) for e in events]
        if use_memory and not self.extra_memory:raise ValueError('Memory model needs memory-enabled tracker')
        return self.model_scorer(events if use_memory else short)

    def update_memory(self,points,scores):
        self.new_anchors=set();self.trusted_updates={}
        events={e['batch']:e for e in self.events}
        for batch,index in self.last_matches.items():
            event=events.get(batch)
            if event is None:continue
            good,margin=reliable(event,index,scores[index],self.state['tracks'][batch].hits)
            if good:
                m=self.memories.setdefault(batch,Memory())
                if m.add(points[index],self.current_context[index],self.state['scan'],scores[index],margin):self.new_anchors.add(batch)
                self.trusted_updates[batch]=index
        for batch in list(self.memories):
            if batch not in self.state['tracks']:del self.memories[batch]

    def __call__(self,points,scores):
        outputs=super().__call__(points,scores)
        if self.extra_memory:self.update_memory(points,scores)
        return outputs
