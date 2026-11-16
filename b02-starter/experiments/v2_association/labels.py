"""Training-only ownership targets. These helpers never run in inference."""
from experiments.v2_association.run_stage1 import historical_identity


def continuity_owners(events,histories,output_counts,points,labels):
    """Choose among pure-prefix tracks using only prior emitted support and age.

    Current labels establish whether the true point is an available candidate;
    history and output_counts must contain observations strictly before this scan.
    No future labels or final full-sequence identity remapping is used.
    """
    chosen={};ranks={}
    for event in events:
        batch=event['batch'];history=histories.get(batch,())
        anchor=historical_identity(history)
        if anchor is None or anchor<=0:continue
        if not any(labels[points[i]['PntID']]==anchor for i in event['indices']):continue
        rank=(output_counts.get(batch,{}).get(anchor,0),sum(v==anchor for v in history),
              float(event['track_features'][35]))
        if anchor not in ranks or rank>ranks[anchor]:
            ranks[anchor]=rank;chosen[anchor]=batch
    return chosen
