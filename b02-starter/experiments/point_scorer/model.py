"""Label-free, causal point features and a small trainable point classifier."""
from collections import deque

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

FIELDS = ('Azimuth', 'Distance', 'Elevation')
GATE = np.asarray((.025, .020, .065), dtype=np.float32)
HISTORY = 6
SPATIAL_DIM = 6
FEATURE_DIM = SPATIAL_DIM + 5 * 14


def coordinates(points):
    return np.asarray([[p[k] for k in FIELDS] for p in points], dtype=np.float32).reshape(-1, 3)


def point_features(current, previous):
    """Previous contains up to five point lists, oldest first; no labels or IDs used.

    Each past scan contributes three nearest offsets/distances and two density
    counts. These summarize motion evidence without a hard target association.
    All feature scales are fixed beforehand, with no dev-set normalization.
    """
    xyz = coordinates(current)
    scaled = xyz / GATE
    n = len(xyz)
    if not n:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)
    tree = cKDTree(scaled)
    nearest = tree.query(scaled, k=2)[0][:, 1]
    local = np.stack([
        np.minimum(nearest, 4) / 4,
        np.log1p(np.maximum(0, tree.query_ball_point(scaled, 1., return_length=True)-1))/8,
        np.log1p(np.maximum(0, tree.query_ball_point(scaled, 2., return_length=True)-1))/8,
    ], axis=1)
    blocks = [xyz, local]
    for lag in range(1, HISTORY):
        past = coordinates(previous[-lag]) if len(previous) >= lag else np.empty((0, 3), np.float32)
        block = np.zeros((n, 14), np.float32)
        # Missing neighbours have capped distance; signed offset remains zero.
        block[:, [3, 7, 11]] = 1
        if len(past):
            past_scaled = past/GATE
            previous_tree = cKDTree(past_scaled)
            distance, indices = previous_tree.query(scaled, k=3)
            valid = indices < len(past)
            safe = np.minimum(indices, len(past)-1)
            offset = np.clip((scaled[:, None, :] - past_scaled[safe])/4, -1, 1)
            offset[~valid] = 0
            neighbours = np.concatenate([offset, np.minimum(distance, 4)[:, :, None]/4], axis=2)
            block[:, :12] = neighbours.reshape(n, 12)
            block[:, 12] = np.log1p(previous_tree.query_ball_point(scaled, 1., return_length=True))/8
            block[:, 13] = np.log1p(previous_tree.query_ball_point(scaled, 2., return_length=True))/8
        blocks.append(block)
    result = np.concatenate(blocks, axis=1).astype(np.float32)
    assert result.shape == (n, FEATURE_DIM) and np.isfinite(result).all()
    return result


class PointScorer(nn.Module):
    def __init__(self, temporal=True):
        super().__init__()
        self.temporal = temporal
        self.layers = nn.Sequential(nn.Linear(FEATURE_DIM, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, features):
        if not self.temporal:
            features = features.clone()
            features[:, SPATIAL_DIM:] = 0
        return self.layers(features).flatten()


class Handler:
    """Score all current originals, then reuse the unchanged baseline association."""
    def __init__(self, model, threshold=.5):
        self.model = model.eval()
        self.threshold = threshold

    @torch.no_grad()
    def __call__(self, points, state):
        from b02.algorithm import process_circle
        history = state.setdefault('point_history', deque(maxlen=HISTORY-1))
        features = point_features(points, list(history))
        device = next(self.model.parameters()).device
        scores = self.model(torch.as_tensor(features, device=device)).sigmoid().cpu().numpy()
        selected = [p for p, score in zip(points, scores) if score >= self.threshold]
        history.append(points)
        return process_circle(selected, state.setdefault('association', {}))
