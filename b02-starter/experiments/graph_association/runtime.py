"""Research integration of a causal joint graph head into frozen matching."""
from collections import Counter
import numpy as np
import torch

from experiments.competition_learning.features import inputs


class GraphScorer:
    def __init__(self, prior, learned, tracker, offset):
        self.prior,self.model = prior,prior.model
        self.learned,self.tracker,self.offset = learned,tracker,offset
        self.counts = Counter()

    def __call__(self, events):
        result = [(np.asarray(edge).copy(),null) for edge,null in self.prior(events)]
        selected,rows,pairs,er,ec,columns = [],[],[],[],[],{}
        for i,event in enumerate(events):
            if self.tracker.emissions.get(event['batch'],0) < 3:continue
            row,pair = inputs(event,self.tracker)
            er.extend([len(rows)]*len(pair))
            for point in event['indices']:
                ec.append(columns.setdefault(point,len(columns)))
            selected.append(i);rows.append(row);pairs.append(pair)
        if not selected:return result
        device = self.learned.row_mean.device
        with torch.inference_mode():
            pair_logits,null_logits = self.learned.graph(
                torch.as_tensor(np.concatenate(pairs),device=device),
                torch.as_tensor(np.stack(rows),device=device),
                torch.as_tensor(er,device=device,dtype=torch.long),torch.as_tensor(ec,device=device,dtype=torch.long))
        pair_logits,null_logits = pair_logits.cpu().numpy(),null_logits.cpu().numpy()
        start = 0
        for row,i in enumerate(selected):
            event = events[i];count = len(event['indices'])
            result[i] = (-pair_logits[start:start+count]-np.asarray(event['base_costs'])/.25,
                -float(null_logits[row])-self.offset-event['dummy_cost']/.25)
            start += count
            self.counts['replaced_rows'] += 1
            self.counts['single_candidate_rows'] += count == 1
        return result
