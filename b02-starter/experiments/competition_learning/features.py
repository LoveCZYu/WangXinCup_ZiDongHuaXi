"""Causal shared inputs; no truth, raw IDs or future observations."""
import numpy as np
from experiments.id_recovery.affinity import pack

PAIR_DIM = 96
ROW_DIM = 101
MAX_ROW = 12
MAX_COLUMN = 32


def metadata(tracker, batch):
    scan = tracker.state['scan']
    count = tracker.emissions.get(batch, 0)
    last = tracker.last_emission.get(batch, scan-14)
    memory = tracker.memories.get(batch)
    return np.asarray([np.log1p(count)/5., min(4., (scan-last)/14.),
        min(1., len(memory.entries)/16.) if memory is not None else 0.], dtype=np.float32)


def inputs(event, tracker):
    tx = event['track_features']
    ex = event['edge_features']
    assert tx.shape == (98,) and ex.shape == (len(event['indices']), 85)
    extra = metadata(tracker, event['batch'])
    row = np.concatenate((tx, extra)).astype(np.float32)
    pair = np.concatenate((pack(np.repeat(tx[None, :], len(ex), axis=0), ex),
        np.repeat(extra[None, :], len(ex), axis=0)), axis=1).astype(np.float32)
    assert row.shape == (ROW_DIM,) and pair.shape == (len(ex), PAIR_DIM)
    return row, pair
