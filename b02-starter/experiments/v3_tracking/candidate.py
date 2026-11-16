"""Replay a confirmed candidate with the frozen point scorer."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.v2_association.run_stage1 import sha
from experiments.v3_tracking.runtime import Pipeline, load_scorer


class CandidateHandler:
    def __init__(self, point, scorer, memory, strategy):
        self.point = point
        self.scorer = scorer
        self.memory = memory
        self.strategy = strategy

    def __call__(self, points, state):
        import torch
        if 'pipeline' not in state:
            state['pipeline'] = Pipeline(self.scorer, self.memory, self.strategy)
            state['history'] = deque(maxlen=5)
        history = state['history']
        with torch.no_grad():
            features = self.point.features(points, list(history))
            scores = self.point.model(torch.as_tensor(features)).sigmoid().tolist()
        history.append(points)
        return state['pipeline'](points, scores)


def load_candidate(profile):
    import torch
    from b02.point_tracker import load_tracker
    profile = Path(profile).resolve()
    spec = json.loads(profile.read_text(encoding='utf-8'))
    assert spec['format'] == 'b02-v3-candidate-profile-v1'
    point_profile = (profile.parent/spec['point_profile']).resolve()
    assert sha(point_profile) == spec['point_profile_sha256']
    point, point_spec = load_tracker(point_profile, device='cpu')
    assert point_spec['checkpoint_sha256'] == spec['point_weights_sha256']
    path = (profile.parent/spec['checkpoint']).resolve()
    scorer, memory, metadata, step = load_scorer(path, spec['checkpoint_sha256'])
    assert metadata['seed'] == spec['seed'] and step == spec['step']
    torch.set_num_threads(2)
    return CandidateHandler(point, scorer, memory, spec['identity']), spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'profile', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--scene', nargs='+', required=True)
    args = parser.parse_args()
    from b02.replay import replay
    plan = json.loads((ROOT/'splits/v3_tracking.json').read_text(encoding='utf-8'))
    assert not set(args.scene) & set(plan['sealed_excluded'])
    handler, spec = load_candidate(args.profile)
    replay(args.data, args.out, handler, spec, args.scene)


if __name__ == '__main__':
    main()
