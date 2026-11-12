"""Small semantic checks for causal feature extraction and optional inference."""
import numpy as np
from model import point_features, PointScorer, FEATURE_DIM, SPATIAL_DIM
import torch


def point(x, i):
    return dict(Azimuth=x, Distance=.3+x/10, Elevation=.5, PntID=i)


current = [point(.1, 1), point(.2, 2), point(.4, 3)]
past = [[point(.09, 99), point(.21, 98)], [], [point(.39, 97)]]
expected = point_features(current, past)
assert expected.shape == (3, FEATURE_DIM)
assert np.isfinite(point_features([current[0]], [])).all()
assert point_features([], past).shape == (0, FEATURE_DIM)
np.testing.assert_allclose(point_features(current[::-1], [p[::-1] for p in past])[::-1], expected)
# IDs and optional truth/unused scalar fields must not change the learned input.
changed = [{**p, 'PntID': -50, 'Batch_ID': 999, 'Amp': 999} for p in current]
np.testing.assert_array_equal(point_features(changed, past), expected)
assert not np.allclose(point_features(current, [])[:, SPATIAL_DIM:], expected[:, SPATIAL_DIM:])
torch.manual_seed(1)
model = PointScorer(temporal=False).eval()
x = torch.tensor(expected)
other = x.clone(); other[:, SPATIAL_DIM:] = torch.randn_like(other[:, SPATIAL_DIM:])
torch.testing.assert_close(model(x), model(other), rtol=0, atol=0)
assert model(torch.empty((0, FEATURE_DIM))).shape == (0,)
print('Feature permutation, empty/singleton, label/ID isolation and spatial-ablation checks passed')
