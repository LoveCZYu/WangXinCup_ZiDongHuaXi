"""External episode outcomes: distinguish recoverable gaps from ID replacement."""
from collections import Counter,defaultdict
from experiments.candidate_onset.audit import Audit


class OutcomeAudit(Audit):
    def __init__(self,last_presence):
        super().__init__(last_presence)
        self.other_hits=defaultdict(Counter)

    def update_record(self,episode):
        record=episode['first_visible']
        if record is None:return
        hits=self.other_hits[(episode['batch'],episode['start_frame'])]
        record['other_id_points_before_recovery']=sum(hits.values())
        record['max_other_id_hits_before_recovery']=max(hits.values(),default=0)
        record['any_other_id_before_recovery']=bool(hits)
        record['replacement_three_plus']=max(hits.values(),default=0)>=3

    def observe(self,frame,circle,points,labels,outputs,tracker,before,scores):
        previous=dict(self.open)
        result=super().observe(frame,circle,points,labels,outputs,tracker,before,scores)
        by_truth=defaultdict(list)
        for point in outputs:
            truth=labels[point['PntID']]
            if truth>0:by_truth[truth].append(point['BatchID'])
        for batch,episode in self.open.items():
            self.other_hits[(batch,episode['start_frame'])].update(b for b in by_truth[episode['truth']] if b!=batch)
            self.update_record(episode)
        for batch,episode in previous.items():
            if batch in self.open:continue
            self.update_record(episode)
            record=episode['first_visible']
            if record is not None:
                record.update(recovered=True,recovered_at_frame=frame,
                    frames_from_first_visible_to_recovery=frame-record['frame'])
            self.other_hits.pop((batch,episode['start_frame']),None)
        return result

    def finalize(self):
        for episode in self.open.values():
            self.update_record(episode)
            if episode['first_visible'] is not None:episode['first_visible']['recovered']=False
        for record in self.events:
            record['had_later_truth_observation']=self.last_presence[record['truth']]>record['frame']
            reason=record['reason']
            if record['any_other_id_before_recovery']:
                self.counts['visible_any_other_id']+=1;self.counts['other_id_reason_'+reason]+=1
            if record['replacement_three_plus']:
                self.counts['visible_replacement_three_plus']+=1;self.counts['replacement3_reason_'+reason]+=1
            if record['recovered']:self.counts['visible_recovered_same_id']+=1
