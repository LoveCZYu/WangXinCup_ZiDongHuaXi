"""Integrate learned evidence into mature IDs, without labels or future input."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.competition_learning.features import inputs
from experiments.competition_learning.model import Model
from experiments.continuation.fusion import MemoryFusionTracker
from experiments.identity_competition.scoring import ColumnScorer
from experiments.v2_association.run_stage1 import sha

ROOT = Path(__file__).resolve().parents[2]
MODES = ('baseline', 'old_column_identity', 'learned_column', 'learned_joint')


def load_model(path, protocol):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['protocol_sha256'] == sha(protocol)
    assert saved['kind'] == 'joint'
    assert saved['model_source_sha256'] == sha(Path(__file__).with_name('model.py'))
    assert saved['feature_source_sha256'] == sha(Path(__file__).with_name('features.py'))
    model = Model().eval()
    model.load_state_dict(saved['model'], strict=True)
    return model, saved


class LearnedScorer:
    def __init__(self, prior, learned, tracker, mode):
        self.prior, self.model = prior, prior.model
        self.learned, self.tracker, self.mode = learned, tracker, mode
        self.counts = Counter()

    def __call__(self, events):
        original = self.prior(events)
        result = [(np.asarray(edge).copy(), null) for edge, null in original]
        rows, row_x, pairs = [], [], []
        for row, event in enumerate(events):
            if self.tracker.emissions.get(event['batch'], 0) < 3:continue
            rx, ex = inputs(event, self.tracker)
            rows.append(row);row_x.append(rx);pairs.append(ex)
        if not rows:return result
        flat = self.learned.score(np.concatenate(pairs))
        logits, offset = {}, 0
        for row, ex in zip(rows, pairs, strict=True):
            logits[row] = flat[offset:offset+len(ex)]
            offset += len(ex)
        if self.mode == 'learned_joint':
            with torch.inference_mode():
                nulls = self.learned.rejection(torch.from_numpy(np.stack(row_x))).numpy()
            for row, null in zip(rows, nulls, strict=True):
                event = events[row]
                # The parent's fixed .25 residual scale converts these deltas
                # to calibrated costs -.25*logit, including the learned null.
                result[row] = (-logits[row]-np.asarray(event['base_costs'])/.25,
                               -float(null)-event['dummy_cost']/.25)
                self.counts['replaced_rows'] += 1
                self.counts['replaced_single_candidate_rows'] += len(event['indices']) == 1
            return result
        assert self.mode == 'learned_column'
        groups = defaultdict(list)
        for row in rows:
            for j, index in enumerate(events[row]['indices']):
                if events[row]['edge_features'][j, 11] >= self.tracker.gate.policy.enter:
                    groups[index].append((row, j))
        for index, choices in groups.items():
            if len(choices) < 2:continue
            best = max(float(logits[row][j]) for row, j in choices)
            self.counts['conflict_columns'] += 1
            for row, j in choices:
                penalty = float(np.clip(best-logits[row][j], 0., 4.))
                result[row][0][j] += penalty
                self.counts['penalized_edges'] += penalty > 0
        return result


class Tracker(MemoryFusionTracker):
    def __init__(self, components, learned, mode):
        assert mode in MODES
        super().__init__(components.scorer, components.affinity, components.head, components.fusion, guard=.90)
        if mode == 'old_column_identity':
            self.model_scorer = ColumnScorer(self.model_scorer, components.affinity, self, 'identity')
        elif mode in ('learned_column', 'learned_joint'):
            self.model_scorer = LearnedScorer(self.model_scorer, learned, self, mode)
