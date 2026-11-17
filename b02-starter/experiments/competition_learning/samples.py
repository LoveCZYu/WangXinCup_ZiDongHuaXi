"""Load private rollout samples and build bounded row/column index tables."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from experiments.competition_learning.features import MAX_ROW, MAX_COLUMN
from experiments.v2_association.run_stage1 import sha


def grouped_index(groups, count, width):
    order = np.argsort(groups, kind='stable')
    counts = np.bincount(groups, minlength=count)
    starts = np.cumsum(counts)-counts
    slot = np.arange(len(order))-np.repeat(starts, counts)
    valid = slot < width
    index = np.zeros((count, width), dtype=np.int64)
    mask = np.zeros((count, width), dtype=bool)
    index[groups[order[valid]], slot[valid]] = order[valid]
    mask[groups[order[valid]], slot[valid]] = True
    return index, mask


def load(root, names, protocol_path):
    parts = defaultdict(list)
    rows = columns = 0
    metadata = []
    for scene in names:
        meta = json.loads((root/f'{scene}.json').read_text(encoding='utf-8'))
        path = root/f'{scene}.npz'
        assert meta['sample_sha256'] == sha(path) and meta['protocol_sha256'] == sha(protocol_path), scene
        metadata.append(meta)
        with np.load(path) as data:
            assert data['protocol_sha256'].item() == sha(protocol_path)
            for key in ('pair', 'row', 'edge_known', 'edge_positive', 'row_allowed', 'col_target', 'col_allowed', 'col_eval'):
                parts[key].append(data[key])
            parts['edge_row'].append(data['edge_row']+rows)
            parts['edge_col'].append(data['edge_col']+columns)
            rows += len(data['row']);columns += len(data['col_allowed'])
    out = {key: np.concatenate(value) for key, value in parts.items()}
    out['row_index'], out['row_mask'] = grouped_index(out['edge_row'], rows, MAX_ROW)
    out['col_index'], out['col_mask'] = grouped_index(out['edge_col'], columns, MAX_COLUMN)
    assert np.all(np.bincount(out['edge_row'], minlength=rows) <= MAX_ROW)
    assert np.isfinite(out['pair']).all() and np.isfinite(out['row']).all()
    out['row_mask'] &= out['edge_known'][out['row_index']]
    out['col_mask'] &= out['edge_known'][out['col_index']]
    out['known_ids'] = np.flatnonzero(out['edge_known'])
    out['row_ids'] = np.flatnonzero(out['row_allowed'])
    out['col_ids'] = np.flatnonzero(out['col_allowed'])
    return out, metadata


class Tensors:
    def __init__(self, arrays, device):
        self.data = {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}

    def batch(self, flat=512, rows=128, columns=128):
        d = self.data
        def choose(key, size):
            ids = d[key]
            return ids[torch.randint(len(ids), (size,), device=ids.device)]
        e = choose('known_ids', flat)
        r = choose('row_ids', rows)
        c = choose('col_ids', columns)
        ri, ci = d['row_index'][r], d['col_index'][c]
        return (d['pair'][e], d['edge_positive'][e].float(), d['row'][r], d['pair'][ri],
            d['row_mask'][r], d['edge_positive'][ri] & d['row_mask'][r],
            d['pair'][ci], d['col_mask'][c], d['col_target'][ci] & d['col_mask'][c],
            d['edge_positive'][ci] & d['col_mask'][c])
