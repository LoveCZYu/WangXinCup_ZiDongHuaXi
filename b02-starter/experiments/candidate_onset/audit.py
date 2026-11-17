"""External label-dependent observer; never changes tracker decisions."""
from collections import Counter,defaultdict
import numpy as np

from experiments.identity_competition.preflight import prefix_anchor


def classify(batch,right,before,tracker,scores):
    """Classify an observed failed opportunity, not its unproven root cause."""
    right=set(right)
    main=[e for e in tracker.events if e['batch']==batch]
    main_edges={i for e in main for i in e['indices']}
    retired={i for b,i in tracker.retired_edges if b==batch}
    prebirth={r['index'] for r in tracker.proposals if r['old']==batch}
    selected=tracker.last_matches.get(batch)
    if selected in right:return 'output_gate',dict(main=bool(main_edges&right),retired=bool(retired&right),prebirth=bool(prebirth&right))
    if main_edges&right:return 'main_candidate_not_selected',dict(main=True,retired=bool(retired&right),prebirth=bool(prebirth&right))
    if retired&right or prebirth&right:return 'recovery_candidate_not_selected',dict(main=False,retired=bool(retired&right),prebirth=bool(prebirth&right))
    source=dict(main=False,retired=False,prebirth=False)
    stages=[s for s in tracker.stage_records if batch in s['track_ids']]
    if stages:
        available=[i for i in right if any(i not in s['occupied_before'] for s in stages)]
        if not available:return 'prior_stage_occupied',source
        gated={i for e in main for i in e['gated_indices']}
        eligible={i for e in main for i in e['eligible_indices']}
        if eligible&right:return 'main_topk',source
        if gated&right:return 'main_base_cost',source
        return 'main_motion_gate',source
    state=before.get(batch)
    if state is None:return 'outside_banks',source
    if state['bank']=='retired':
        calls=[c for c in tracker.retired_calls if batch in c['ids']]
        if not calls:return 'retired_not_offered_or_expired',source
        unoccupied=[i for i in right if any(i not in c['occupied'] for c in calls)]
        if not unoccupied:return 'retired_occupied',source
        if not any(scores[i]>=tracker.gate.policy.enter for i in unoccupied):return 'retired_point_score',source
        return 'retired_geometry_or_topk',source
    return 'banked_no_main_or_recovery_edge',source


class Audit:
    def __init__(self,last_presence):
        self.last_presence=last_presence
        self.history=defaultdict(Counter);self.anchors={};self.last_good={};self.open={}
        self.last_truth_output=defaultdict(dict)
        self.since_good=defaultdict(Counter)
        self.counts=Counter();self.events=[]

    def observe(self,frame,circle,points,labels,outputs,tracker,before,scores):
        # Anchor is locked at first reliable past prefix; later impurity cannot erase the failure.
        for b,h in self.history.items():
            if b not in self.anchors:
                truth=prefix_anchor(h)
                if truth is not None:
                    self.anchors[b]=truth
                    self.last_good[b]=self.last_truth_output[b][truth]
        right=defaultdict(list)
        for i,p in enumerate(points):
            if labels[p['PntID']]>0:right[labels[p['PntID']]].append(i)
        published={p['BatchID']:labels[p['PntID']] for p in outputs}
        new=[]
        for batch,truth in self.anchors.items():
            if batch not in self.last_good:continue
            if published.get(batch)==truth:
                if batch in self.open:
                    episode=self.open.pop(batch)
                    episode['recovered_at_frame']=frame
                    episode['recovery_gap']=frame-episode['start_frame']
                    self.counts['episodes_recovered']+=1
                self.since_good[batch].clear()
                continue
            if not right[truth] and self.last_presence.get(truth,-1)<=frame:
                self.counts['post_truth_end_rows_excluded']+=1
                continue
            if batch not in self.open:
                episode=dict(batch=batch,truth=truth,start_frame=frame,start_circle=circle,
                    start_raw_present=bool(right[truth]),last_good_frame=self.last_good[batch],first_visible=None)
                self.open[batch]=episode
                self.counts['episodes_started']+=1
                self.counts['starts_observation_present' if right[truth] else 'starts_observation_gap']+=1
            episode=self.open[batch]
            if not right[truth] or episode['first_visible'] is not None:continue
            reason,source=classify(batch,right[truth],before,tracker,scores)
            state=before.get(batch)
            emitted=state['emissions'] if state else 0
            pure=prefix_anchor(self.history[batch])==truth
            scored=any(e['batch']==batch and len(e['indices']) and e.get('track_features',np.zeros(0)).shape==(98,)
                for e in tracker.events)
            rule_eligible=bool(emitted>=3 and pure and scored)
            recent=self.since_good[batch].copy()
            record=dict(frame=frame,circle=circle,batch=batch,truth=truth,point_id=points[right[truth][0]]['PntID'],
                reason=reason,candidate_sources=source,bank=state['bank'] if state else 'outside',
                age=tracker.state['scan']-state['last_scan'] if state else None,
                gap_since_good=frame-self.last_good[batch],episode_started_with_observation_gap=not episode['start_raw_present'],
                prior_sampling_rule_eligible=rule_eligible,history_still_pure=pure,scored_main_row=bool(scored),
                intervening_internal_updates=dict(recent),
                raw_target_score=float(max(scores[i] for i in right[truth])),
                raw_target_outputs=[p['BatchID'] for p in outputs if labels[p['PntID']]==truth])
            episode['first_visible']=record;self.events.append(record);new.append(record)
            self.counts['first_visible_failures']+=1;self.counts['reason_'+reason]+=1
            self.counts['sampling_eligible' if rule_eligible else 'sampling_excluded']+=1
            self.counts['first_visible_with_prior_other_target_update']+=recent['internal_other_target']>0
            self.counts['first_visible_with_prior_clutter_update']+=recent['internal_clutter']>0
            self.counts['first_visible_with_prior_wrong_trusted_update']+=recent['trusted_other_target']+recent['trusted_clutter']>0
        for batch,truth in self.anchors.items():
            if published.get(batch)==truth:continue
            i=tracker.last_matches.get(batch)
            if i is not None:
                value=labels[points[i]['PntID']]
                kind='correct' if value==truth else 'other_target' if value>0 else 'clutter' if value==0 else 'unknown'
                self.since_good[batch]['internal_'+kind]+=1
                if batch in tracker.trusted_updates:self.since_good[batch]['trusted_'+kind]+=1
        for p in outputs:
            batch,truth=p['BatchID'],labels[p['PntID']]
            if truth>=0:self.history[batch][truth]+=1
            if truth>0:
                self.last_truth_output[batch][truth]=frame
                if self.anchors.get(batch)==truth:self.last_good[batch]=frame
        return new
